"""
main_pipeline.py
=================
Master orchestration script for the Money Mule Detection System.

Runs all phases in sequence:
  Phase 1  â€” Data ingestion (load or generate dataset)
  Phase 2  â€” Graph construction
  Phase 3  â€” Feature engineering
  Phase 4  â€” Community detection
  Phase 5  â€” Machine learning models (Isolation Forest, RF, GB)
  Phase 6  â€” Lifecycle stage classification
  Phase 7  â€” Evaluation & reporting
  Phase 8  â€” Visualisations

Usage
-----
  # Use your PaySim CSV:
  python main_pipeline.py --source paysim --path data/raw/paysim.csv

  # Sample 50K rows from PaySim:
  python main_pipeline.py --source paysim --sample 50000
"""

import argparse
import time
import sys
import os
import pandas as pd
import numpy as np

# Make src importable from project root
sys.path.insert(0, os.path.dirname(__file__))

from config.config import *
from src.ingestion.data_loader       import load_synthetic, load_paysim, load_amlsim
from src.graph.graph_builder         import build_transaction_graph, graph_summary
from src.graph.neo4j_connector       import export_graph_to_neo4j, NEO4J_AVAILABLE
from src.features.feature_engineering import build_feature_matrix, get_feature_columns
from src.community.community_detector import (
    detect_communities, score_communities, get_suspicious_accounts
)
from src.models.ml_models            import (
    train_isolation_forest, train_random_forest,
    train_gradient_boosting, build_ensemble_scores, save_model
)
from src.gnn.gnn_models              import train_gnn, TORCH_AVAILABLE, plot_embeddings
from src.lifecycle.lifecycle_detector import (
    detect_lifecycle_stages, get_early_stage_accounts
)
from src.evaluation.evaluator        import (
    evaluate_model, evaluate_early_detection,
    evaluate_community_detection, compile_report
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="paysim",
                   choices=["synthetic", "paysim", "amlsim", "custom", "advanced_synthetic"],
                   help="Data source")
    p.add_argument("--path",   default=None, help="Path to CSV file")
    p.add_argument("--sample", type=int, default=None,
                   help="Sample N rows (for large datasets)")
    p.add_argument("--regen",  action="store_true",
                   help="Regenerate synthetic dataset")
    p.add_argument("--skip-viz", action="store_true",
                   help="Skip visualisation (faster)")
    p.add_argument("--with-gnn", action="store_true",
                   help="Train the optional GNN model in Phase 5")
    p.add_argument("--gnn-model", default=GNN_DEFAULT_MODEL,
                   choices=["sage", "gat"],
                   help="GNN architecture when --with-gnn is enabled")
    p.add_argument("--gnn-epochs", type=int, default=GNN_DEFAULT_EPOCHS,
                   help="Maximum epochs for GNN training")
    p.add_argument("--export-neo4j", action="store_true",
                   help="Export the summary graph and account metadata to Neo4j")
    p.add_argument("--neo4j-uri", default=None, help="Neo4j Bolt URI")
    p.add_argument("--neo4j-user", default=None, help="Neo4j username")
    p.add_argument("--neo4j-password", default=None, help="Neo4j password")
    p.add_argument("--neo4j-database", default=None, help="Neo4j database name")
    return p.parse_args()


def banner(text):
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def derive_score_threshold(scores: pd.Series, labels: pd.Series | np.ndarray | None) -> float:
    """Tune the final suspicious-account threshold when labels are available."""
    if labels is None:
        return float(np.percentile(scores.values, 95))

    label_series = pd.Series(labels, index=scores.index).reindex(scores.index).fillna(0).astype(int)
    if label_series.sum() == 0:
        return float(np.percentile(scores.values, 95))

    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(label_series.values, scores.values)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-9
    )
    return float(thresholds[int(np.nanargmax(f1_scores))])


def run_pipeline(args=None):
    if args is None:
        args = parse_args()

    t_total = time.time()

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 1 â€” DATA INGESTION
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    banner("Phase 1 â€” Data Ingestion")
    t0 = time.time()

    if args.source == "paysim":
        df = load_paysim(path=args.path, sample_n=args.sample)
    elif args.source == "amlsim":
        df = load_amlsim(path=args.path, sample_n=args.sample)
    elif args.source == "custom":
        from src.ingestion.data_loader import load_csv
        if not args.path:
            raise ValueError("Must provide --path when using --source custom")
        df = load_csv(path=args.path)
        if args.sample and len(df) > args.sample:
            df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
    elif args.source == "advanced_synthetic":
        from src.ingestion.advanced_generator import generate_advanced_synthetic_dataset
        df = generate_advanced_synthetic_dataset()
        df["source"] = "advanced_synthetic"
        if args.sample and len(df) > args.sample:
            df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df = load_synthetic(regenerate=args.regen)
        if args.sample and len(df) > args.sample:
            df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)

    # Save canonical dataset
    df.to_csv(DATA_PROCESSED / "canonical_transactions.csv", index=False)
    print(f"  Phase 1 done in {time.time()-t0:.1f}s")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 2 â€” GRAPH CONSTRUCTION
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    banner("Phase 2 â€” Graph Construction")
    t0 = time.time()

    G_summary, G_full = build_transaction_graph(df)

    stats = graph_summary(G_summary)
    print("\nGraph statistics:")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")

    print(f"  Phase 2 done in {time.time()-t0:.1f}s")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 3 â€” FEATURE ENGINEERING
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    banner("Phase 3 â€” Feature Engineering")
    t0 = time.time()

    feature_matrix = build_feature_matrix(df, G_summary)
    feat_cols      = get_feature_columns(feature_matrix)
    print(f"  Feature matrix : {feature_matrix.shape}")
    print(f"  Feature columns: {len(feat_cols)}")
    print(f"  Phase 3 done in {time.time()-t0:.1f}s")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 4 â€” COMMUNITY DETECTION
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    banner("Phase 4 â€” Community Detection")
    t0 = time.time()

    partition = detect_communities(G_summary, method="louvain")
    comm_df   = score_communities(partition, feature_matrix, G_summary)
    susp_accs = get_suspicious_accounts(partition, comm_df)

    # Attach community ID to feature matrix
    feature_matrix["community_id"] = feature_matrix["account"].map(partition)

    # Save community results
    comm_save = comm_df.drop(columns=["accounts"], errors="ignore")
    comm_save.to_csv(DATA_PROCESSED / "communities.csv", index=False)
    pd.Series(susp_accs, name="account").to_csv(
        OUTPUTS_RESULTS / "suspicious_from_communities.csv", index=False
    )
    print(f"  Suspicious accounts from communities: {len(susp_accs):,}")
    print(f"  Phase 4 done in {time.time()-t0:.1f}s")

    if args.export_neo4j:
        banner("Neo4j Export")
        t0 = time.time()
        try:
            export_info = export_graph_to_neo4j(
                G_summary,
                feature_matrix=feature_matrix,
                communities=comm_df,
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
                database=args.neo4j_database,
            )
            print(f"  Exported {export_info['nodes_exported']:,} nodes and "
                  f"{export_info['edges_exported']:,} edges")
            print(f"  Neo4j target        : {export_info['uri']} / {export_info['database']}")
        except Exception as exc:
            print(f"  Neo4j export skipped: {exc}")
            if not NEO4J_AVAILABLE:
                print("  Install the driver with: python3 -m pip install neo4j")
        print(f"  Neo4j step done in {time.time()-t0:.1f}s")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 5 â€” MACHINE LEARNING MODELS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    banner("Phase 5 â€” Machine Learning Models")
    t0 = time.time()

    has_labels = (
        feature_matrix["is_fraud"].sum() > 0 and
        feature_matrix["is_fraud"].nunique() > 1
    )

    # 5a. Isolation Forest (always runs)
    iso_model, iso_scaler, iso_scores, iso_preds = train_isolation_forest(feature_matrix)
    save_model({"model": iso_model, "scaler": iso_scaler}, "isolation_forest")

    model_results = {}

    # Isolation Forest eval
    if has_labels:
        accs_indexed = feature_matrix.set_index("account")
        common_iso   = iso_scores.index.intersection(accs_indexed.index)
        iso_res = evaluate_model(
            accs_indexed.loc[common_iso, "is_fraud"].values,
            iso_preds.loc[common_iso].values,
            iso_scores.loc[common_iso].values,
            "IsolationForest",
        )
        model_results["IsolationForest"] = {
            "y_test":      accs_indexed.loc[common_iso, "is_fraud"].values,
            "y_prob_test": iso_scores.loc[common_iso].values,
            "y_pred_test": iso_preds.loc[common_iso].values,
        }
    else:
        iso_res = {"Model": "IsolationForest"}

    # 5b. Random Forest
    rf_model = rf_scaler = rf_res_dict = fi = None
    rf_results_eval = {"Model": "RandomForest"}

    if has_labels:
        rf_model, rf_scaler, rf_res_dict = train_random_forest(feature_matrix)
        save_model({"model": rf_model, "scaler": rf_scaler, "feat_cols": feat_cols}, "random_forest")
        fi = rf_res_dict["feat_importance"]
        rf_results_eval = evaluate_model(
            rf_res_dict["y_test"],
            rf_res_dict["y_pred_test"],
            rf_res_dict["y_prob_test"],
            "RandomForest",
        )
        model_results["RandomForest"] = {
            "y_test":      rf_res_dict["y_test"],
            "y_prob_test": rf_res_dict["y_prob_test"],
            "y_pred_test": rf_res_dict["y_pred_test"],
        }

    # 5c. Gradient Boosting
    gb_model = gb_scaler = gb_res_dict = None
    gb_results_eval = {"Model": "GradientBoosting"}
    gnn_results_eval = {"Model": f"GNN-{args.gnn_model.upper()}"}
    gnn_res_dict = None

    if has_labels:
        gb_model, gb_scaler, gb_res_dict = train_gradient_boosting(feature_matrix)
        save_model({"model": gb_model, "scaler": gb_scaler, "feat_cols": feat_cols}, "gradient_boosting")
        gb_results_eval = evaluate_model(
            gb_res_dict["y_test"],
            gb_res_dict["y_pred_test"],
            gb_res_dict["y_prob_test"],
            "GradientBoosting",
        )
        model_results["GradientBoosting"] = {
            "y_test":      gb_res_dict["y_test"],
            "y_prob_test": gb_res_dict["y_prob_test"],
            "y_pred_test": gb_res_dict["y_pred_test"],
        }

    # 5d. GraphSAGE (optional)
    if args.with_gnn:
        if TORCH_AVAILABLE and has_labels:
            gnn_res_dict = train_gnn(
                feature_matrix,
                df,
                feat_cols,
                model_type=args.gnn_model,
                n_epochs=args.gnn_epochs,
                patience=GNN_DEFAULT_PATIENCE,
                hidden1=GNN_HIDDEN1,
                hidden2=GNN_HIDDEN2,
            )
            if gnn_res_dict:
                gnn_results_eval = evaluate_model(
                    gnn_res_dict["test_results"]["y_test"],
                    gnn_res_dict["test_results"]["y_pred_test"],
                    gnn_res_dict["test_results"]["y_prob_test"],
                    f"GNN-{args.gnn_model.upper()}",
                )
                model_results[f"GNN-{args.gnn_model.upper()}"] = {
                    "y_test":      gnn_res_dict["test_results"]["y_test"],
                    "y_prob_test": gnn_res_dict["test_results"]["y_prob_test"],
                    "y_pred_test": gnn_res_dict["test_results"]["y_pred_test"],
                }
        elif not TORCH_AVAILABLE:
            print(f"\n  {args.gnn_model.upper()} requested, but torch/torch_geometric are unavailable.")
            print("  Skipping GNN training and continuing with classical models.")
        else:
            print(f"\n  {args.gnn_model.upper()} requested, but labels are unavailable.")
            print("  Skipping GNN training because supervised node classification needs labels.")

    # 5e. Ensemble score
    if rf_res_dict and gb_res_dict:
        ensemble_scores = build_ensemble_scores(
            iso_scores,
            rf_res_dict["proba"],
            gb_res_dict["proba"],
            gnn_proba=gnn_res_dict["proba_series"] if gnn_res_dict else None,
        )
    else:
        ensemble_scores = iso_scores.rename("ensemble_score")

    # Build final suspicious account list (ensemble threshold)
    threshold = derive_score_threshold(
        ensemble_scores,
        feature_matrix.set_index("account").loc[ensemble_scores.index, "is_fraud"]
        if has_labels else None,
    )
    suspicious_ml    = ensemble_scores[ensemble_scores > threshold].index.tolist()
    suspicious_all   = list(set(susp_accs) | set(suspicious_ml))

    # Save
    result_df = feature_matrix[["account", "is_fraud"]].copy()
    result_df["iso_score"]      = iso_scores.reindex(result_df["account"]).values
    result_df["iso_pred"]       = iso_preds.reindex(result_df["account"]).values
    if rf_res_dict:
        result_df["rf_proba"]   = rf_res_dict["proba"].reindex(result_df["account"]).values
        result_df["rf_pred"]    = rf_res_dict["preds"].reindex(result_df["account"]).values
    if gb_res_dict:
        result_df["gb_proba"]   = gb_res_dict["proba"].reindex(result_df["account"]).values
        result_df["gb_pred"]    = gb_res_dict["preds"].reindex(result_df["account"]).values
    if gnn_res_dict:
        result_df["gnn_proba"]  = gnn_res_dict["proba_series"].reindex(result_df["account"]).values
    result_df["ensemble_score"] = ensemble_scores.reindex(result_df["account"]).values
    result_df["ml_suspicious"]  = result_df["ensemble_score"] > threshold
    result_df.to_csv(OUTPUTS_RESULTS / "model_predictions.csv", index=False)

    pd.Series(suspicious_all, name="account").to_csv(
        OUTPUTS_RESULTS / "all_suspicious_accounts.csv", index=False
    )
    print(f"\n  Total suspicious accounts (ensemble + community): {len(suspicious_all):,}")
    print(f"  Phase 5 done in {time.time()-t0:.1f}s")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 6 â€” LIFECYCLE DETECTION
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    banner("Phase 6 â€” Lifecycle Stage Detection")
    t0 = time.time()

    lifecycle_df    = detect_lifecycle_stages(feature_matrix)
    early_stage_df  = get_early_stage_accounts(lifecycle_df)
    early_stage_df[["account","lifecycle_stage","risk_level"]].to_csv(
        OUTPUTS_RESULTS / "early_stage_accounts.csv", index=False
    )
    print(f"  Phase 6 done in {time.time()-t0:.1f}s")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 7 â€” EVALUATION & REPORTING
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    banner("Phase 7 â€” Evaluation & Reporting")
    t0 = time.time()

    model_metrics = [iso_res, rf_results_eval, gb_results_eval]
    if gnn_res_dict:
        model_metrics.append(gnn_results_eval)
    early_eval    = evaluate_early_detection(lifecycle_df)
    comm_eval     = evaluate_community_detection(partition, feature_matrix, comm_df)

    comparison_df = pd.DataFrame([
        m for m in model_metrics if "Precision" in m
    ])

    compile_report(model_metrics, early_eval, comm_eval, suspicious_all)
    print(f"  Phase 7 done in {time.time()-t0:.1f}s")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PHASE 8 â€” VISUALISATIONS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if not args.skip_viz:
        from src.visualization.visualizer import (
            plot_transaction_network, plot_mule_cluster,
            plot_feature_importance, plot_pr_curves,
            plot_model_comparison, plot_lifecycle_distribution,
            plot_temporal_drift, plot_community_suspicion,
            plot_anomaly_scores,
        )

        banner("Phase 8 â€” Visualisations")
        t0 = time.time()

        plot_transaction_network(G_summary)

        # Pick largest suspicious community for cluster plot
        if len(comm_df) > 0:
            top_comm     = comm_df.iloc[0]
            cluster_nodes = top_comm["accounts"][:50]   # limit for clarity
            plot_mule_cluster(G_summary, cluster_nodes)

        if fi is not None:
            plot_feature_importance(fi)

        if model_results:
            plot_pr_curves(model_results)

        if len(comparison_df) > 0:
            plot_model_comparison(comparison_df)

        if gnn_res_dict is not None:
            plot_embeddings(gnn_res_dict["embeddings"], feature_matrix)

        plot_lifecycle_distribution(lifecycle_df)

        # Temporal drift: pick one mule + one normal account
        if has_labels:
            mule_acc   = feature_matrix[feature_matrix["is_fraud"] == 1]["account"].iloc[0]
            normal_senders = feature_matrix[
                (feature_matrix["is_fraud"] == 0) &
                (feature_matrix["account"].isin(df["sender_account"]))
            ]["account"]
            if len(normal_senders) > 0:
                normal_acc = normal_senders.iloc[0]
            else:
                normal_acc = feature_matrix[feature_matrix["is_fraud"] == 0]["account"].iloc[0]
            plot_temporal_drift(df, mule_acc, normal_acc)

        plot_community_suspicion(comm_df)

        iso_lab = feature_matrix["is_fraud"] if has_labels else None
        plot_anomaly_scores(iso_scores, labels=iso_lab, threshold=threshold)

        print(f"  Phase 8 done in {time.time()-t0:.1f}s")
    else:
        print("\nVisualisations skipped (--skip-viz flag set)")

    # â”€â”€ Final summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    banner("Pipeline Complete")
    print(f"  Total time           : {time.time()-t_total:.1f}s")
    print(f"  Suspicious accounts  : {len(suspicious_all):,}")
    print(f"  Early-stage flagged  : {len(early_stage_df):,}")
    print(f"\nOutputs written to:")
    print(f"  {DATA_PROCESSED}")
    print(f"  {OUTPUTS_RESULTS}")
    print(f"  {OUTPUTS_PLOTS}")
    print(f"  {OUTPUTS_REPORTS}")

    return {
        "df":               df,
        "G":                G_summary,
        "feature_matrix":   feature_matrix,
        "communities":      comm_df,
        "lifecycle_df":     lifecycle_df,
        "gnn_results":      gnn_res_dict,
        "suspicious_all":   suspicious_all,
        "ensemble_scores":  ensemble_scores,
    }


if __name__ == "__main__":
    run_pipeline()
