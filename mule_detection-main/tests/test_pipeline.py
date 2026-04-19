"""
tests/test_pipeline.py
=======================
Unit and integration tests for the mule detection system.

Tests cover:
  - Data generation and loading
  - Graph construction correctness
  - Feature engineering sanity checks
  - Model training and prediction shapes
  - Lifecycle classification rules
  - Community detection
  - API scoring logic (without running the server)

Run with:
  python -m pytest tests/test_pipeline.py -v
  python tests/test_pipeline.py         # standalone runner
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import networkx as nx
import unittest
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ── Minimal synthetic data fixtures ──────────────────────────────────────────

def make_mini_transactions(n: int = 200, fraud_rate: float = 0.10) -> pd.DataFrame:
    """Create a tiny transaction DataFrame for fast unit testing."""
    rng   = np.random.default_rng(42)
    n_acc = 50
    accs  = [f"ACC{i:04d}" for i in range(n_acc)]
    n_f   = int(n * fraud_rate)

    rows = []
    for i in range(n):
        src = rng.choice(accs)
        dst = rng.choice([a for a in accs if a != src])
        amt = float(rng.lognormal(7, 1.5))
        ts  = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=int(i * 1.2))
        rows.append({
            "sender_account":     src,
            "receiver_account":   dst,
            "transaction_amount": round(amt, 2),
            "timestamp":          ts,
            "is_fraud":           int(i < n_f),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataGenerator(unittest.TestCase):

    def test_generate_returns_dataframe(self):
        from src.ingestion.data_generator import generate_synthetic_dataset
        df = generate_synthetic_dataset(
            n_accounts=200, n_transactions=500,
            mule_fraction=0.05, n_mule_networks=3,
            save=False,
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_required_columns_present(self):
        from src.ingestion.data_generator import generate_synthetic_dataset
        df = generate_synthetic_dataset(
            n_accounts=100, n_transactions=200, save=False
        )
        required = {"sender_account", "receiver_account",
                    "transaction_amount", "timestamp", "is_fraud"}
        self.assertTrue(required.issubset(set(df.columns)),
                        f"Missing: {required - set(df.columns)}")

    def test_fraud_accounts_exist(self):
        from src.ingestion.data_generator import generate_synthetic_dataset
        df = generate_synthetic_dataset(
            n_accounts=200, n_transactions=1000,
            mule_fraction=0.05, save=False,
        )
        self.assertGreater(df["is_fraud"].sum(), 0,
                           "Expected some fraud transactions")

    def test_amounts_positive(self):
        from src.ingestion.data_generator import generate_synthetic_dataset
        df = generate_synthetic_dataset(
            n_accounts=100, n_transactions=300, save=False
        )
        self.assertTrue((df["transaction_amount"] > 0).all(),
                        "All amounts should be positive")

    def test_no_self_loops(self):
        from src.ingestion.data_generator import generate_synthetic_dataset
        df = generate_synthetic_dataset(
            n_accounts=100, n_transactions=200, save=False
        )
        self.assertTrue(
            (df["sender_account"] != df["receiver_account"]).all(),
            "Self-loop transactions found"
        )


class TestPaySimLoader(unittest.TestCase):

    def _write_paysim_fixture(self, rows: list[dict]) -> str:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        path = Path(tmp_dir.name) / "paysim_fixture.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_load_paysim_keeps_all_fraud_rows_when_sampling(self):
        from src.ingestion.data_loader import load_paysim

        rows = [
            {
                "step": i + 1,
                "type": "TRANSFER" if i % 2 == 0 else "CASH_OUT",
                "amount": float(100 + i),
                "nameOrig": f"C{i}",
                "oldbalanceOrg": float(1000 + i),
                "newbalanceOrig": float(900 + i),
                "nameDest": f"D{i}",
                "oldbalanceDest": 0.0,
                "newbalanceDest": float(100 + i),
                "isFraud": int(i < 2),
                "isFlaggedFraud": 0,
            }
            for i in range(10)
        ]
        path = self._write_paysim_fixture(rows)

        df = load_paysim(path=path, sample_n=5)

        self.assertEqual(int(df["is_fraud"].sum()), 2)
        self.assertEqual(len(df), 5)

    def test_load_paysim_adds_balance_features(self):
        from src.ingestion.data_loader import load_paysim

        rows = [{
            "step": 1,
            "type": "TRANSFER",
            "amount": 200.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 200.0,
            "newbalanceOrig": 0.0,
            "nameDest": "D1",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 1,
            "isFlaggedFraud": 0,
        }]
        path = self._write_paysim_fixture(rows)

        df = load_paysim(path=path)

        for col in [
            "sender_drained", "dest_no_increase", "origin_balance_error",
            "dest_balance_error", "balance_error_flag", "type",
        ]:
            self.assertIn(col, df.columns)


class TestGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.df = make_mini_transactions()

    def test_graph_has_correct_node_count(self):
        from src.graph.graph_builder import build_transaction_graph
        G, _ = build_transaction_graph(self.df)
        all_accounts = set(self.df["sender_account"]) | set(self.df["receiver_account"])
        self.assertEqual(G.number_of_nodes(), len(all_accounts))

    def test_graph_has_edges(self):
        from src.graph.graph_builder import build_transaction_graph
        G, _ = build_transaction_graph(self.df)
        self.assertGreater(G.number_of_edges(), 0)

    def test_summary_graph_is_digraph(self):
        from src.graph.graph_builder import build_transaction_graph
        G_sum, G_full = build_transaction_graph(self.df)
        self.assertIsInstance(G_sum,  nx.DiGraph)
        self.assertIsInstance(G_full, nx.MultiDiGraph)

    def test_edge_has_amount_attr(self):
        from src.graph.graph_builder import build_transaction_graph
        G, _ = build_transaction_graph(self.df)
        for u, v, data in G.edges(data=True):
            self.assertIn("total_amount", data,
                          f"Edge {u}->{v} missing 'total_amount'")
            break   # check first edge only for speed

    def test_fraud_nodes_labelled(self):
        from src.graph.graph_builder import build_transaction_graph
        G, _ = build_transaction_graph(self.df)
        fraud_senders = set(self.df[self.df["is_fraud"] == 1]["sender_account"])
        for n in fraud_senders:
            if n in G:
                self.assertEqual(G.nodes[n].get("is_fraud"), 1,
                                 f"Node {n} should be labelled as fraud")

    def test_graph_summary_returns_dict(self):
        from src.graph.graph_builder import build_transaction_graph, graph_summary
        G, _ = build_transaction_graph(self.df)
        stats = graph_summary(G)
        self.assertIsInstance(stats, dict)
        for key in ["n_nodes", "n_edges", "density"]:
            self.assertIn(key, stats)

    def test_get_subgraph(self):
        from src.graph.graph_builder import build_transaction_graph, get_subgraph
        G, _ = build_transaction_graph(self.df)
        nodes = list(G.nodes())[:5]
        sub   = get_subgraph(G, nodes)
        self.assertIsInstance(sub, nx.DiGraph)
        self.assertGreaterEqual(sub.number_of_nodes(), len(nodes))


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.df = make_mini_transactions(n=300)
        from src.graph.graph_builder import build_transaction_graph
        self.G, _ = build_transaction_graph(self.df)

    def test_node_features_returns_dataframe(self):
        from src.features.feature_engineering import compute_node_features
        feats = compute_node_features(self.df)
        self.assertIsInstance(feats, pd.DataFrame)
        self.assertGreater(len(feats), 0)

    def test_node_features_no_nan(self):
        from src.features.feature_engineering import compute_node_features
        feats = compute_node_features(self.df)
        nan_cols = feats.isnull().any()
        nan_cols = nan_cols[nan_cols].index.tolist()
        self.assertEqual(nan_cols, [], f"NaN found in columns: {nan_cols}")

    def test_graph_features_returns_dataframe(self):
        from src.features.feature_engineering import compute_graph_features
        feats = compute_graph_features(self.G)
        self.assertIsInstance(feats, pd.DataFrame)
        for col in ["pagerank", "betweenness", "clustering", "out_degree"]:
            self.assertIn(col, feats.columns, f"Missing column: {col}")

    def test_pagerank_sums_to_one(self):
        from src.features.feature_engineering import compute_graph_features
        feats = compute_graph_features(self.G)
        pr_sum = feats["pagerank"].sum()
        self.assertAlmostEqual(pr_sum, 1.0, places=3,
                               msg=f"PageRank sum = {pr_sum}, expected ≈ 1.0")

    def test_temporal_features_shape(self):
        from src.features.feature_engineering import compute_temporal_features
        feats = compute_temporal_features(self.df)
        all_accounts = set(self.df["sender_account"]) | set(self.df["receiver_account"])
        self.assertEqual(len(feats), len(all_accounts))

    def test_build_feature_matrix_columns(self):
        from src.features.feature_engineering import build_feature_matrix
        fm = build_feature_matrix(self.df, self.G)
        self.assertIn("account",  fm.columns)
        self.assertIn("is_fraud", fm.columns)
        self.assertGreater(len(fm.columns), 10)

    def test_feature_matrix_no_inf(self):
        from src.features.feature_engineering import build_feature_matrix, get_feature_columns
        fm        = build_feature_matrix(self.df, self.G)
        feat_cols = get_feature_columns(fm)
        has_inf   = np.isinf(fm[feat_cols].values).any()
        self.assertFalse(has_inf, "Feature matrix contains Inf values")


class TestMLModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.ingestion.data_generator import generate_synthetic_dataset
        from src.graph.graph_builder       import build_transaction_graph
        from src.features.feature_engineering import build_feature_matrix
        cls.df = generate_synthetic_dataset(
            n_accounts=300, n_transactions=1500,
            mule_fraction=0.06, n_mule_networks=3,
            save=False,
        )
        G, _ = build_transaction_graph(cls.df)
        cls.fm = build_feature_matrix(cls.df, G)

    def test_isolation_forest_runs(self):
        from src.models.ml_models import train_isolation_forest
        iso, scaler, scores, preds = train_isolation_forest(self.fm)
        self.assertEqual(len(scores), len(self.fm))
        self.assertEqual(len(preds),  len(self.fm))
        self.assertTrue(set(preds.unique()).issubset({0, 1}))

    def test_random_forest_runs(self):
        from src.models.ml_models import train_random_forest
        rf, scaler, results = train_random_forest(self.fm)
        self.assertIn("proba",           results)
        self.assertIn("feat_importance", results)
        self.assertIn("pr_auc",          results)
        self.assertGreater(results["pr_auc"], 0.0)

    def test_gradient_boosting_runs(self):
        from src.models.ml_models import train_gradient_boosting
        gb, scaler, results = train_gradient_boosting(self.fm)
        self.assertIn("proba",  results)
        self.assertIn("pr_auc", results)

    def test_ensemble_scores_shape(self):
        from src.models.ml_models import (
            train_isolation_forest, train_random_forest,
            train_gradient_boosting, build_ensemble_scores,
        )
        _, _, iso_scores, _ = train_isolation_forest(self.fm)
        _, _, rf_r          = train_random_forest(self.fm)
        _, _, gb_r          = train_gradient_boosting(self.fm)
        ens = build_ensemble_scores(iso_scores, rf_r["proba"], gb_r["proba"])
        self.assertEqual(len(ens), len(self.fm))
        self.assertTrue((ens >= 0).all() and (ens <= 1).all(),
                        "Ensemble scores should be in [0, 1]")

    def test_feature_importance_length(self):
        from src.models.ml_models import train_random_forest, _get_feature_cols
        rf, _, results = train_random_forest(self.fm)
        feat_cols = _get_feature_cols(self.fm)
        self.assertEqual(len(results["feat_importance"]), len(feat_cols))

    def test_graph_sage_runs_when_available(self):
        from src.features.feature_engineering import get_feature_columns
        from src.gnn.gnn_models import TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            self.skipTest("torch/torch_geometric not installed")

        from src.gnn.gnn_models import train_graph_sage

        feat_cols = get_feature_columns(self.fm)
        results = train_graph_sage(
            self.fm,
            self.df,
            feat_cols,
            n_epochs=10,
            patience=5,
            hidden1=32,
            hidden2=16,
        )
        self.assertIn("proba_series", results)
        self.assertEqual(len(results["proba_series"]), len(self.fm))
        self.assertIn("test_results", results)


class TestLifecycleDetector(unittest.TestCase):

    def _make_row(self, **kwargs) -> pd.Series:
        defaults = {
            "tx_count": 10, "n_sent": 5, "n_recv": 5,
            "max_gap_days": 5, "burst_ratio": 2.0, "weekly_cv": 0.5,
            "passthrough_ratio": 0.3, "small_round_txns": 0,
            "sudden_wakeup": 0, "tx_velocity_7d": 2, "tx_velocity_30d": 5,
            "active_days": 30, "max_weekly_txns": 5, "mean_weekly_txns": 2.5,
            "fanout_ratio": 0.5, "degree_ratio": 1.0, "is_fraud": 0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_dormant_detection(self):
        from src.lifecycle.lifecycle_detector import classify_lifecycle_stage
        row = self._make_row(tx_count=2, max_gap_days=45)
        self.assertEqual(classify_lifecycle_stage(row), "Dormant")

    def test_laundering_detection_high_burst(self):
        from src.lifecycle.lifecycle_detector import classify_lifecycle_stage
        row = self._make_row(
            burst_ratio=12, tx_count=50, weekly_cv=2.0,
            tx_velocity_7d=8, passthrough_ratio=0.5
        )
        self.assertEqual(classify_lifecycle_stage(row), "Laundering")

    def test_recruitment_detection(self):
        from src.lifecycle.lifecycle_detector import classify_lifecycle_stage
        row = self._make_row(
            sudden_wakeup=1, n_sent=2, n_recv=8, tx_count=10
        )
        self.assertEqual(classify_lifecycle_stage(row), "Recruitment")

    def test_activation_detection(self):
        from src.lifecycle.lifecycle_detector import classify_lifecycle_stage
        row = self._make_row(
            small_round_txns=2, tx_count=5, burst_ratio=2.5
        )
        self.assertEqual(classify_lifecycle_stage(row), "Activation")

    def test_normal_detection(self):
        from src.lifecycle.lifecycle_detector import classify_lifecycle_stage
        row = self._make_row(
            tx_count=50, burst_ratio=1.5, weekly_cv=0.3,
            max_gap_days=2, sudden_wakeup=0
        )
        self.assertEqual(classify_lifecycle_stage(row), "Normal")

    def test_detect_stages_returns_dataframe(self):
        from src.ingestion.data_generator    import generate_synthetic_dataset
        from src.graph.graph_builder          import build_transaction_graph
        from src.features.feature_engineering import build_feature_matrix
        from src.lifecycle.lifecycle_detector import detect_lifecycle_stages

        df  = generate_synthetic_dataset(n_accounts=150, n_transactions=500,
                                          save=False)
        G,_ = build_transaction_graph(df)
        fm  = build_feature_matrix(df, G)
        lc  = detect_lifecycle_stages(fm, use_ml=False)

        self.assertIn("lifecycle_stage", lc.columns)
        self.assertIn("early_flag",      lc.columns)
        self.assertIn("risk_level",      lc.columns)

    def test_risk_levels_valid(self):
        from src.ingestion.data_generator    import generate_synthetic_dataset
        from src.graph.graph_builder          import build_transaction_graph
        from src.features.feature_engineering import build_feature_matrix
        from src.lifecycle.lifecycle_detector import detect_lifecycle_stages, STAGE_RISK

        df  = generate_synthetic_dataset(n_accounts=100, n_transactions=300,
                                          save=False)
        G,_ = build_transaction_graph(df)
        fm  = build_feature_matrix(df, G)
        lc  = detect_lifecycle_stages(fm, use_ml=False)

        valid_levels = set(STAGE_RISK.values())
        actual = set(lc["risk_level"].unique())
        self.assertTrue(actual.issubset(valid_levels),
                        f"Unexpected risk levels: {actual - valid_levels}")


class TestCommunityDetector(unittest.TestCase):

    def setUp(self):
        from src.ingestion.data_generator    import generate_synthetic_dataset
        from src.graph.graph_builder          import build_transaction_graph
        from src.features.feature_engineering import build_feature_matrix

        df       = generate_synthetic_dataset(n_accounts=200, n_transactions=800,
                                               n_mule_networks=3, save=False)
        G, _     = build_transaction_graph(df)
        self.G   = G
        self.fm  = build_feature_matrix(df, G)

    def test_detect_returns_partition(self):
        from src.community.community_detector import detect_communities
        partition = detect_communities(self.G, method="components")
        self.assertIsInstance(partition, dict)
        self.assertEqual(set(partition.keys()), set(self.G.nodes()))

    def test_partition_covers_all_nodes(self):
        from src.community.community_detector import detect_communities
        partition = detect_communities(self.G, method="components")
        for node in self.G.nodes():
            self.assertIn(node, partition,
                          f"Node {node} missing from partition")

    def test_score_communities_returns_dataframe(self):
        from src.community.community_detector import (
            detect_communities, score_communities
        )
        partition = detect_communities(self.G, method="components")
        comm_df   = score_communities(partition, self.fm, self.G)
        self.assertIsInstance(comm_df, pd.DataFrame)
        self.assertIn("suspicion_score", comm_df.columns)
        self.assertIn("is_suspicious",   comm_df.columns)

    def test_suspicion_scores_in_range(self):
        from src.community.community_detector import (
            detect_communities, score_communities
        )
        partition = detect_communities(self.G, method="components")
        comm_df   = score_communities(partition, self.fm, self.G)
        self.assertTrue(
            (comm_df["suspicion_score"] >= 0).all() and
            (comm_df["suspicion_score"] <= 1).all(),
            "Suspicion scores out of [0, 1] range"
        )


class TestIntegration(unittest.TestCase):
    """End-to-end integration test: runs the full pipeline on tiny data."""

    def test_full_pipeline_smoke(self):
        """
        Smoke test: verify the pipeline runs without errors on minimal data.
        Does NOT check metric values — just that nothing crashes.
        """
        from src.ingestion.data_generator     import generate_synthetic_dataset
        from src.graph.graph_builder           import build_transaction_graph
        from src.features.feature_engineering import (
            build_feature_matrix, get_feature_columns
        )
        from src.community.community_detector import (
            detect_communities, score_communities
        )
        from src.models.ml_models             import (
            train_isolation_forest, train_random_forest
        )
        from src.lifecycle.lifecycle_detector import detect_lifecycle_stages

        # Tiny dataset
        df = generate_synthetic_dataset(
            n_accounts=100, n_transactions=400,
            mule_fraction=0.08, n_mule_networks=2,
            save=False,
        )
        G, _  = build_transaction_graph(df)
        fm    = build_feature_matrix(df, G)
        part  = detect_communities(G, method="components")
        cdf   = score_communities(part, fm, G)

        _, _, iso_scores, iso_preds = train_isolation_forest(fm)
        _, _, rf_res = train_random_forest(fm)

        lc = detect_lifecycle_stages(fm, use_ml=False)

        # Assertions
        self.assertGreater(len(iso_scores), 0)
        self.assertGreater(rf_res["pr_auc"], 0.0)
        self.assertIn("lifecycle_stage", lc.columns)
        self.assertIn("is_suspicious",   cdf.columns)

        print("\n  [Integration] Full smoke test PASSED")


# ── Standalone test runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  MULE DETECTION — UNIT & INTEGRATION TEST SUITE")
    print("=" * 65)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestDataGenerator,
        TestGraphBuilder,
        TestFeatureEngineering,
        TestMLModels,
        TestLifecycleDetector,
        TestCommunityDetector,
        TestIntegration,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  {len(result.failures)} FAILURES | {len(result.errors)} ERRORS "
              f"/ {result.testsRun} tests")
    print("=" * 65)

    sys.exit(0 if result.wasSuccessful() else 1)
