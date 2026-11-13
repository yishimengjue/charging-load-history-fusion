"""Origin-frozen NaN, observation-mask and left-censored elapsed-slot features."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


HISTORY = 168
HORIZON = 24
MINIMUM_OBSERVATIONS = 24
RECENT_SLOTS = 24
RAW_NAMES = [f"history_lag_{lag}_hours" for lag in range(167, -1, -1)]
MASK_NAMES = [f"observed_lag_{lag}" for lag in range(167, -1, -1)]
AGE_NAMES = [f"elapsed_slots_lag_{lag}" for lag in range(167, -1, -1)]


def history_array(history) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.ndim != 2 or values.shape[1] != HISTORY:
        raise ValueError("History must have exactly 168 ordered snapshot columns")
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("History must contain only nonnegative finite values or NaN")
    return values


def history_eligible(history) -> np.ndarray:
    observed = np.isfinite(history_array(history))
    return (observed.sum(axis=1) >= MINIMUM_OBSERVATIONS) & observed[:, -RECENT_SLOTS:].any(axis=1)


def observation_features(history) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(history_array(history))
    positions = np.arange(HISTORY)[None, :]
    previous = np.maximum.accumulate(np.where(observed, positions, -1), axis=1)
    elapsed = np.where(previous < 0, HISTORY, positions - previous)
    return observed.astype(float), elapsed.astype(float)


def scalar_observation_features(history) -> tuple[np.ndarray, np.ndarray]:
    masks, ages = [], []
    for row in history:
        mask, elapsed, previous = [], [], None
        for index, value in enumerate(row):
            seen = bool(np.isfinite(value))
            if seen:
                previous = index
            mask.append(float(seen))
            elapsed.append(float(HISTORY if previous is None else index - previous))
        masks.append(mask)
        ages.append(elapsed)
    return np.asarray(masks).reshape(-1, HISTORY), np.asarray(ages).reshape(-1, HISTORY)


def feature_frame(history, augmented: bool = False) -> pd.DataFrame:
    values = history_array(history)
    if not isinstance(augmented, bool):
        raise TypeError("augmented must be boolean")
    base, names = values, list(RAW_NAMES)
    if augmented:
        mask, age = observation_features(values)
        base = np.column_stack((values, mask, age))
        names.extend(MASK_NAMES + AGE_NAMES)
    features = np.column_stack((np.repeat(base, HORIZON, axis=0),
                                np.tile(np.arange(HORIZON, dtype=float), len(values))))
    return pd.DataFrame(features, columns=names + ["forecast_step"])


def design_fingerprint(features, truth) -> str:
    digest = hashlib.sha256(json.dumps(list(features.columns)).encode("ascii"))
    for value in (features.to_numpy(dtype=float), np.asarray(truth, dtype=float)):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()
