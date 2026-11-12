"""
Fraud Detection API for UPI Fraud Detection System.

Provides REST endpoints for fraud risk scoring using trained models.

Run with:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health  - Health check
    POST /predict - Fraud risk scoring (accepts features, returns score + prediction)
    POST /predict_screenshot - Fraud risk from PhonePe/GPay screenshot
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import joblib
import numpy as np
import pandas as pd

from data_prep import prepare_features_target
from screenshot_parser import parse_screenshot, ScreenshotParserError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Global model cache: key -> (model, metadata)
_model_cache: Dict[str, Tuple[Any, Optional[Dict]]] = {}


def load_model_for_api(dataset: str, model_name: str) -> Tuple[Any, Optional[Dict]]:
    """Load a trained model from disk.

    Looks for models/{dataset}_{model_name}.joblib and optional .json metadata.
    On-disk model files use the short dataset name ("ieee"), while the API
    accepts the public name ("ieee-cis"); map between them here. Also accepts
    the short name for consistency with the dashboard's Pipeline Control page.

    Caches the loaded model in memory.
    """
    dataset = {"ieee-cis": "ieee"}.get(dataset, dataset)
    key = f"{dataset}_{model_name}"
    if key not in _model_cache:
        model_path = MODELS_DIR / f"{key}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)
        metadata = None
        meta_path = model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        _model_cache[key] = (model, metadata)
    return _model_cache[key]


async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "healthy", "service": "upi-fraud-detection"})


async def predict(request: Request) -> JSONResponse:
    """Fraud risk prediction endpoint.

    Expects JSON body:
        {
            "dataset": "ieee-cis",      # optional, default "ieee-cis"
            "model": "lightgbm",        # optional, default "lightgbm"
            "features": {               # required
                "TransactionAmt": 100.0,
                "card1": 12345,
                ...
            }
            # or a list of feature dicts for batch scoring
        }

    Returns:
        {
            "dataset": "ieee-cis",
            "model": "lightgbm",
            "predictions": [
                {"index": 0, "fraud_score": 0.85, "is_fraud": 1},
                ...
            ]
        }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    dataset = body.get("dataset", "ieee-cis")
    model_name = body.get("model", "lightgbm")
    features = body.get("features")

    if features is None:
        return JSONResponse(
            {"error": "Missing 'features' in request body"}, status_code=400
        )

    try:
        model, metadata = load_model_for_api(dataset, model_name)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    try:
        # Convert features to DataFrame
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            df = pd.DataFrame(features)
        else:
            return JSONResponse(
                {"error": "features must be a dict or list of dicts"}, status_code=400
            )

        # Prepare features without target column (API scoring mode)
        X, _ = prepare_features_target(df, require_target=False)

        # Reindex to match model's expected features, filling missing with -999
        expected_features = None
        if metadata and "feature_names" in metadata:
            expected_features = list(metadata["feature_names"])
        elif hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        if expected_features:
            X = X.reindex(columns=expected_features, fill_value=-999)

        # Get predictions
        scores = model.predict_proba(X)[:, 1]
        predictions = (scores >= 0.5).astype(int)

        results = []
        for i in range(len(scores)):
            results.append(
                {
                    "index": i,
                    "fraud_score": float(scores[i]),
                    "is_fraud": int(predictions[i]),
                }
            )

        return JSONResponse(
            {
                "dataset": dataset,
                "model": model_name,
                "predictions": results,
            }
        )

    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


async def predict_screenshot(request: Request) -> JSONResponse:
    """Fraud risk prediction from payment screenshot.

    Accepts multipart/form-data with:
        - image: screenshot file (required)
        - dataset: "ieee-cis" (optional, default "ieee-cis")
        - model: "lightgbm" (optional, default "lightgbm")

    Or JSON body with base64 encoded image:
        {
            "dataset": "ieee-cis",
            "model": "lightgbm",
            "image_base64": "base64_encoded_image_data"
        }

    Returns:
        {
            "dataset": "ieee-cis",
            "model": "lightgbm",
            "app_detected": "phonepe",
            "parsed_transaction": {...},
            "predictions": [{"index": 0, "fraud_score": 0.85, "is_fraud": 1}]
        }
    """
    try:
        model_name = "lightgbm"
        dataset = "ieee-cis"
        image_file = None
        image_path = None

        # Try multipart form data first
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            dataset = form.get("dataset", "ieee-cis")
            model_name = form.get("model", "lightgbm")
            image_file = form.get("image")

            if image_file:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(await image_file.read())
                    image_path = tmp.name
        else:
            # Try JSON with base64
            try:
                body = await request.json()
                dataset = body.get("dataset", "ieee-cis")
                model_name = body.get("model", "lightgbm")
                image_base64 = body.get("image_base64")

                if image_base64:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(base64.b64decode(image_base64))
                        image_path = tmp.name
            except Exception:
                pass

        if not image_path:
            return JSONResponse(
                {"error": "No image provided. Use multipart/form-data with 'image' field or JSON with 'image_base64'."},
                status_code=400
            )

        try:
            # Parse screenshot
            parsed = parse_screenshot(image_path)
        except ScreenshotParserError as exc:
            return JSONResponse({"error": f"Screenshot parsing failed: {exc}"}, status_code=400)
        finally:
            # Clean up temp file
            try:
                Path(image_path).unlink()
            except Exception:
                pass

        # Convert parsed transaction to features
        feature_df = parsed.to_features(require_target=False)

        # Load model and predict
        try:
            model, metadata = load_model_for_api(dataset, model_name)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

        try:
            # Prepare features
            X, _ = prepare_features_target(feature_df, require_target=False)

            # Reindex to match model's expected features
            expected_features = None
            if metadata and "feature_names" in metadata:
                expected_features = list(metadata["feature_names"])
            elif hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            if expected_features:
                X = X.reindex(columns=expected_features, fill_value=-999)

            # Get predictions
            scores = model.predict_proba(X)[:, 1]
            predictions = (scores >= 0.5).astype(int)

            results = []
            for i in range(len(scores)):
                results.append({
                    "index": i,
                    "fraud_score": float(scores[i]),
                    "is_fraud": int(predictions[i]),
                })

            # Build response with parsed transaction details
            parsed_data = {
                "app_type": parsed.app_type,
                "amount": parsed.amount,
                "sender_name": parsed.sender_name,
                "sender_upi": parsed.sender_upi,
                "receiver_name": parsed.receiver_name,
                "receiver_upi": parsed.receiver_upi,
                "transaction_id": parsed.transaction_id,
                "timestamp": parsed.timestamp,
                "status": parsed.status,
                "bank_name": parsed.bank_name,
            }

            return JSONResponse({
                "dataset": dataset,
                "model": model_name,
                "app_detected": parsed.app_type,
                "parsed_transaction": parsed_data,
                "predictions": results,
            })

        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"Prediction failed: {exc}"}, status_code=500)

    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/predict", predict, methods=["POST"]),
        Route("/predict_screenshot", predict_screenshot, methods=["POST"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
