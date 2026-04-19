"""
src/api/scoring_api.py
=======================
Real-time mule detection scoring API built with FastAPI.

This simulates the production deployment of the trained models.
In a real FinTech system, this API would be called by:
  - The transaction processing engine (after each transaction batch)
  - The AML analyst dashboard (on-demand account lookup)
  - Scheduled batch jobs (nightly full portfolio re-scoring)

Endpoints
---------
  POST /score/account         — score a single account by ID
  POST /score/batch           — score a batch of accounts
  POST /score/transaction     — score a new transaction in real-time
  GET  /health                — health check
  GET  /model/info            — return loaded model metadata

Usage
-----
  pip install fastapi uvicorn

  # Start server:
  python src/api/scoring_api.py

  # Or with uvicorn directly:
  uvicorn src.api.scoring_api:app --host 0.0.0.0 --port 8000 --reload

  # Test with curl:
  curl -X POST http://localhost:8000/score/account \
       -H "Content-Type: application/json" \
       -d '{"account_id": "ACC000123"}'
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
import json
from glob import glob

# ── FastAPI import guard ──────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[Warning] FastAPI not installed. Run: pip install fastapi uvicorn")
    print("Defining mock classes for import compatibility.")

    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

# ── Pydantic request/response models ─────────────────────────────────────────

class AccountScoreRequest(BaseModel):
    account_id: str = Field(..., description="Account identifier")
    include_explanation: bool = Field(
        default=True,
        description="Include top feature contributions in response"
    )


class TransactionScoreRequest(BaseModel):
    sender_account:     str   = Field(..., description="Sending account ID")
    receiver_account:   str   = Field(..., description="Receiving account ID")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp:          Optional[str] = Field(
        default=None, description="ISO timestamp (defaults to now)"
    )


class BatchScoreRequest(BaseModel):
    account_ids: list[str] = Field(..., description="List of account IDs to score")
    threshold:   float     = Field(default=0.5, description="Suspicion threshold")


class AccountScoreResponse(BaseModel):
    account_id:       str
    suspicion_score:  float
    risk_tier:        str            # LOW / MEDIUM / HIGH / CRITICAL
    is_flagged:       bool
    lifecycle_stage:  Optional[str]
    top_risk_factors: list[dict]
    score_components: dict
    model_version:    str
    scored_at:        str


# ── Model registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """Loads and caches all trained models at startup."""

    def __init__(self):
        self.rf_bundle        = None
        self.gb_bundle        = None
        self.iso_bundle       = None
        self.gnn_scores       = None
        self.model_predictions = None
        self.feature_matrix   = None
        self.lifecycle_df     = None
        self.model_version    = "1.0.0"
        self._loaded          = False

    def load(self):
        """Load all models from disk."""
        print("Loading models from registry...")

        # Random Forest
        rf_path = OUTPUTS_MODELS / "random_forest.pkl"
        if rf_path.exists():
            with open(rf_path, "rb") as f:
                self.rf_bundle = pickle.load(f)
            print(f"  RF loaded: {rf_path}")

        # Gradient Boosting
        gb_path = OUTPUTS_MODELS / "gradient_boosting.pkl"
        if gb_path.exists():
            with open(gb_path, "rb") as f:
                self.gb_bundle = pickle.load(f)
            print(f"  GB loaded: {gb_path}")

        # Isolation Forest
        iso_path = OUTPUTS_MODELS / "isolation_forest.pkl"
        if iso_path.exists():
            with open(iso_path, "rb") as f:
                self.iso_bundle = pickle.load(f)
            print(f"  ISO loaded: {iso_path}")

        gnn_files = sorted(glob(str(OUTPUTS_RESULTS / "gnn_*_proba.csv")))
        if gnn_files:
            gnn_path = gnn_files[-1]
            self.gnn_scores = pd.read_csv(gnn_path).set_index("account")["gnn_proba"]
            print(f"  GNN scores loaded: {len(self.gnn_scores):,} accounts ({os.path.basename(gnn_path)})")

        # Feature matrix (for lookup)
        fm_path = DATA_PROCESSED / "feature_matrix.csv"
        if fm_path.exists():
            self.feature_matrix = pd.read_csv(fm_path).set_index("account")
            print(f"  Feature matrix loaded: {len(self.feature_matrix):,} accounts")

        pred_path = OUTPUTS_RESULTS / "model_predictions.csv"
        if pred_path.exists():
            self.model_predictions = pd.read_csv(pred_path).set_index("account")
            print(f"  Model predictions loaded: {len(self.model_predictions):,} accounts")

        # Lifecycle labels
        lc_path = DATA_PROCESSED / "lifecycle_results.csv"
        if lc_path.exists():
            self.lifecycle_df = pd.read_csv(lc_path).set_index("account")
            print(f"  Lifecycle labels loaded: {len(self.lifecycle_df):,} accounts")

        self._loaded = True
        print("Registry ready.\n")

    def is_ready(self):
        return self._loaded and (
            self.rf_bundle is not None or self.iso_bundle is not None
        )


# ── Scoring logic ─────────────────────────────────────────────────────────────

def _risk_tier(score: float) -> str:
    if score < 0.25:  return "LOW"
    if score < 0.50:  return "MEDIUM"
    if score < 0.75:  return "HIGH"
    return "CRITICAL"


def _ensemble_score(registry: ModelRegistry, x_scaled: np.ndarray) -> float:
    """Compute ensemble suspicion score from available models."""
    scores = []

    if registry.rf_bundle:
        rf    = registry.rf_bundle["model"]
        proba = rf.predict_proba(x_scaled)[0, 1]
        scores.append(("rf",  proba, 0.45))

    if registry.gb_bundle:
        gb    = registry.gb_bundle["model"]
        proba = gb.predict_proba(x_scaled)[0, 1]
        scores.append(("gb",  proba, 0.35))

    if registry.iso_bundle:
        iso   = registry.iso_bundle["model"]
        raw   = -iso.score_samples(x_scaled)[0]
        # Normalise iso score to [0,1] — use historical percentile as proxy
        normed = float(np.clip((raw + 0.5) / 1.0, 0, 1))
        scores.append(("iso", normed, 0.20))

    if not scores:
        return 0.5   # fallback if no models loaded

    total_w = sum(w for _, _, w in scores)
    return sum(s * w for _, s, w in scores) / total_w


def _lookup_precomputed_score(account_id: str, registry: ModelRegistry) -> dict | None:
    """Use stored predictions when available, including optional GNN scores."""
    if registry.model_predictions is None or account_id not in registry.model_predictions.index:
        return None

    row = registry.model_predictions.loc[account_id]
    score = row.get("ensemble_score")
    if pd.isna(score):
        return None

    details = {
        "ensemble_score": float(score),
        "iso_score": None if pd.isna(row.get("iso_score")) else float(row.get("iso_score")),
        "rf_proba": None if pd.isna(row.get("rf_proba")) else float(row.get("rf_proba")),
        "gb_proba": None if pd.isna(row.get("gb_proba")) else float(row.get("gb_proba")),
    }
    if "gnn_proba" in row.index and not pd.isna(row.get("gnn_proba")):
        details["gnn_proba"] = float(row.get("gnn_proba"))
    return details


def _top_features(registry: ModelRegistry, x_scaled: np.ndarray,
                  feat_cols: list, n: int = 5) -> list[dict]:
    """Return top N features driving the suspicion score."""
    if not registry.rf_bundle:
        return []

    rf      = registry.rf_bundle["model"]
    fi      = rf.feature_importances_
    contrib = fi * np.abs(x_scaled[0])   # importance × feature magnitude
    top_idx = np.argsort(contrib)[::-1][:n]

    return [
        {
            "feature":    feat_cols[i],
            "importance": round(float(fi[i]), 4),
            "value":      round(float(x_scaled[0][i]), 4),
        }
        for i in top_idx
    ]


def score_account(account_id: str, registry: ModelRegistry,
                  include_explanation: bool = True) -> dict:
    """Core scoring function — used by both API and batch jobs."""

    if registry.feature_matrix is None:
        raise ValueError("Feature matrix not loaded")

    if account_id not in registry.feature_matrix.index:
        raise KeyError(f"Account '{account_id}' not found in feature matrix")

    # ── Get features ─────────────────────────────────────────────────────────
    row       = registry.feature_matrix.loc[account_id]
    feat_cols = registry.rf_bundle["feat_cols"] if registry.rf_bundle \
                else [c for c in row.index if c not in {"is_fraud", "community_id"}]

    x_raw    = row[feat_cols].fillna(0).values.reshape(1, -1).astype(float)
    scaler   = (registry.rf_bundle or registry.iso_bundle)["scaler"]
    x_scaled = scaler.transform(x_raw)

    # ── Score ─────────────────────────────────────────────────────────────────
    precomputed = _lookup_precomputed_score(account_id, registry)
    score = precomputed["ensemble_score"] if precomputed is not None else _ensemble_score(registry, x_scaled)

    # ── Lifecycle stage ───────────────────────────────────────────────────────
    stage = None
    if registry.lifecycle_df is not None and account_id in registry.lifecycle_df.index:
        stage = registry.lifecycle_df.loc[account_id, "lifecycle_stage"]

    # ── Risk factors ─────────────────────────────────────────────────────────
    top_rf = _top_features(registry, x_scaled, feat_cols) if include_explanation else []

    return {
        "account_id":       account_id,
        "suspicion_score":  round(score, 4),
        "risk_tier":        _risk_tier(score),
        "is_flagged":       score >= 0.50,
        "lifecycle_stage":  stage,
        "top_risk_factors": top_rf,
        "score_components": precomputed or {},
        "model_version":    registry.model_version,
        "scored_at":        datetime.utcnow().isoformat() + "Z",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

if FASTAPI_AVAILABLE:
    app      = FastAPI(
        title       = "Mule Detection Scoring API",
        description = "Real-time AML mule account detection",
        version     = "1.0.0",
    )
    registry = ModelRegistry()

    @app.on_event("startup")
    async def startup():
        registry.load()

    @app.get("/health")
    async def health():
        return {
            "status":  "ok" if registry.is_ready() else "degraded",
            "models":  {
                "random_forest":    registry.rf_bundle  is not None,
                "gradient_boosting":registry.gb_bundle  is not None,
                "isolation_forest": registry.iso_bundle is not None,
                "gnn_scores":       registry.gnn_scores is not None,
            },
            "n_accounts_indexed": (
                len(registry.feature_matrix)
                if registry.feature_matrix is not None else 0
            ),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    @app.get("/model/info")
    async def model_info():
        feat_cols = []
        if registry.rf_bundle:
            feat_cols = registry.rf_bundle.get("feat_cols", [])
        return {
            "version":       registry.model_version,
            "n_features":    len(feat_cols),
            "feature_list":  feat_cols[:20],    # truncated for readability
            "models_loaded": [
                m for m, b in [
                    ("random_forest",     registry.rf_bundle),
                    ("gradient_boosting", registry.gb_bundle),
                    ("isolation_forest",  registry.iso_bundle),
                    ("gnn_scores",        registry.gnn_scores),
                ] if b is not None
            ],
        }

    @app.post("/score/account", response_model=AccountScoreResponse)
    async def score_account_endpoint(request: AccountScoreRequest):
        if not registry.is_ready():
            raise HTTPException(503, "Models not loaded yet — try again shortly")
        try:
            result = score_account(
                request.account_id, registry,
                include_explanation=request.include_explanation
            )
            return result
        except KeyError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, f"Scoring error: {e}")

    @app.post("/score/batch")
    async def score_batch_endpoint(request: BatchScoreRequest):
        if not registry.is_ready():
            raise HTTPException(503, "Models not loaded yet")

        results    = []
        errors     = []
        flagged    = []

        for acc in request.account_ids:
            try:
                r = score_account(acc, registry, include_explanation=False)
                results.append(r)
                if r["suspicion_score"] >= request.threshold:
                    flagged.append(acc)
            except KeyError:
                errors.append({"account_id": acc, "error": "not found"})
            except Exception as e:
                errors.append({"account_id": acc, "error": str(e)})

        return {
            "n_requested": len(request.account_ids),
            "n_scored":    len(results),
            "n_flagged":   len(flagged),
            "flagged_accounts": flagged,
            "scores":      results,
            "errors":      errors,
            "scored_at":   datetime.utcnow().isoformat() + "Z",
        }

    @app.post("/score/transaction")
    async def score_transaction_endpoint(request: TransactionScoreRequest):
        """
        Score a transaction by scoring BOTH the sender and receiver,
        then returning the higher of the two suspicion scores.
        """
        if not registry.is_ready():
            raise HTTPException(503, "Models not loaded yet")

        sender_result   = None
        receiver_result = None
        errors          = []

        for acc in [request.sender_account, request.receiver_account]:
            try:
                r = score_account(acc, registry, include_explanation=True)
                if acc == request.sender_account:
                    sender_result   = r
                else:
                    receiver_result = r
            except KeyError:
                errors.append(f"Account '{acc}' not in index")
            except Exception as e:
                errors.append(str(e))

        # High-value transaction rule: amount > 10k raises suspicion
        amount_flag = request.transaction_amount > 10_000

        max_score = max(
            sender_result["suspicion_score"]   if sender_result   else 0,
            receiver_result["suspicion_score"] if receiver_result else 0,
        )

        return {
            "transaction": {
                "sender":    request.sender_account,
                "receiver":  request.receiver_account,
                "amount":    request.transaction_amount,
                "timestamp": request.timestamp or datetime.utcnow().isoformat() + "Z",
            },
            "transaction_suspicion_score": round(max_score, 4),
            "high_value_flag":             amount_flag,
            "overall_risk_tier":           _risk_tier(max_score),
            "flag_transaction":            max_score >= 0.50 or amount_flag,
            "sender_score":                sender_result,
            "receiver_score":              receiver_result,
            "errors":                      errors,
        }


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install FastAPI first: pip install fastapi uvicorn")
        sys.exit(1)

    print("Starting Mule Detection Scoring API...")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
