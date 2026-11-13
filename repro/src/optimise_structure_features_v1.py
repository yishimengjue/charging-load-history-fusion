"""History-only horizon alignment, seasonal reference and two-part mean."""

import warnings

import numpy as np

from src.optimise_missing_features_v1 import feature_frame, history_array, history_eligible


def seasonal_reference(history):
    x = history_array(history)
    if not history_eligible(x).all():
        raise ValueError("Historical minimum-information requirement not met")
    week, day = x[:, :24].copy(), x[:, -24:].copy()
    if len(x):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            hourly = np.nanmedian(x.reshape(-1, 7, 24), axis=1)
            overall = np.nanmedian(x, axis=1)[:, None]
        anchor = np.where(np.isfinite(week), week, np.where(np.isfinite(day), day,
                          np.where(np.isfinite(hourly), hourly, overall)))
    else:
        hourly, anchor = np.empty((0,24)), np.empty((0,24))
    if not np.isfinite(anchor).all():
        raise ValueError("Eligible history did not produce a finite reference")
    return anchor, week, day, hourly


def aligned_features(history):
    anchor, week, day, hourly = seasonal_reference(history)
    frame = feature_frame(history, False)
    for name, values in (("target_week_lag_kw",week), ("target_day_lag_kw",day),
                         ("target_hour_median_kw",hourly), ("seasonal_reference_kw",anchor)):
        frame[name] = values.reshape(-1)
    return frame, anchor


def reconstruct(residual, anchor):
    residual, anchor = np.asarray(residual,float), np.asarray(anchor,float)
    if residual.shape != anchor.shape or not np.isfinite(residual).all() or not np.isfinite(anchor).all():
        raise ValueError("Invalid residual/reference shape or value")
    return np.maximum(0, anchor + residual)


def hurdle_mean(probability, positive_size):
    p, size = np.asarray(probability,float), np.asarray(positive_size,float)
    if p.shape != size.shape or not np.isfinite(p).all() or not np.isfinite(size).all() or (p<0).any() or (p>1).any():
        raise ValueError("Invalid occurrence probability or positive-size prediction")
    return p * np.maximum(size,0)
