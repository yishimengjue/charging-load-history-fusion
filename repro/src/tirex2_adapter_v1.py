"""Pinned CPU-only, per-series TiRex-2 inference. No local optimization."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_REVISION = "9d67931c2a5f0febb3b92599fbc846353b4e1a42"
MODEL_REVISION = "30788a311699acba7e6393ad9700ea7b7cb9bb2d"
MODEL = ROOT / "vendor" / f"tirex-2-model-{MODEL_REVISION}"
WEIGHT_SHA = "17760e4eee4c14e0d8bad8b9d9ce6f5076aaeeb9d956a88b8e95cf1493a1c176"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OPTIONS = dict(tta_diff=True, tta_sign_flip=False, pad_context=True)
BATCH_SIZE = 4
_configured = False


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def validate_histories(histories):
    x = np.asarray(histories, dtype=float)
    if x.ndim != 2 or x.shape[1] != 168:
        raise ValueError("Expected N by 168 history slots")
    valid = np.isfinite(x)
    if np.isinf(x).any() or (x[valid] < 0).any():
        raise ValueError("Expected nonnegative power or NaN")
    if ((valid.sum(axis=1) < 24) | (~valid[:, -24:].any(axis=1))).any():
        raise ValueError("Original history support requirement failed")
    with np.errstate(over="raise"):
        return x.astype(np.float32, copy=True)


def unpack(outputs, count):
    if len(outputs) != count:
        raise ValueError("Output count changed")
    result = []
    for value in outputs:
        q = np.asarray(value)
        if q.shape != (1, 9, 24) or not np.isfinite(q).all():
            raise ValueError("Nonfinite or incorrectly shaped forecast")
        result.append(q[0].copy())
    return np.stack(result) if result else np.empty((0, 9, 24), np.float32)


def load_pipeline():
    global _configured
    for name in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        if os.environ.get(name) != "1":
            raise ValueError("Explicit offline and telemetry flags are required")
    if sha(MODEL / "model.safetensors") != WEIGHT_SHA:
        raise ValueError("Weight hash mismatch")
    config = json.loads((MODEL / "config.json").read_text())
    if config["quantiles"] != QUANTILES or config.get("tta_diff") is not True:
        raise ValueError("Unexpected checkpoint configuration")
    import torch
    import tirex2

    installed = Path(tirex2.__file__).parent
    source = ROOT / "vendor" / f"tirex-2-{CODE_REVISION}" / "src/tirex2"
    for path in source.rglob("*.py"):
        if sha(path) != sha(installed / path.relative_to(source)):
            raise ValueError(f"Installed code differs from pinned source: {path}")
    if not _configured:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        _configured = True
    model = tirex2.load_model(MODEL, device="cpu", compile=False, use_flex_attention=False)
    model.model.requires_grad_(False)
    if model.model.training:
        raise ValueError("Model was not placed in evaluation mode")
    levels = [round(float(q), 6) for q in model.model.quantiles]
    if levels != QUANTILES:
        raise ValueError("Quantile mapping changed")
    return model


def predict(model, histories):
    import torch
    from tirex2 import TimeseriesType

    x = validate_histories(histories)
    if len(x) == 0:
        return np.empty((0, 9, 24), np.float32)
    before = x.copy()
    series = [TimeseriesType(target=torch.from_numpy(row[None, :]),
                             past_covariates=None, future_covariates=None) for row in x]
    with torch.inference_mode():
        result = model.forecast(series, prediction_length=24, output_type="numpy",
                                batch_size=BATCH_SIZE, **OPTIONS)
    if not np.array_equal(x, before, equal_nan=True):
        raise ValueError("Forecast mutated input")
    return unpack(result, len(x))
