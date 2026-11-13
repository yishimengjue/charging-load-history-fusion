"""Pinned offline Chronos-2-Small P50 adapter; no training or imputation."""

from pathlib import Path
import os
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
REVISION="ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a"
MODEL=ROOT/"vendor"/f"chronos-2-small-{REVISION}"
QUANTILES=[.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]
BATCH_SIZE=16
_configured=False


def validate_histories(histories):
    x=np.asarray(histories,dtype=float)
    if x.ndim!=2 or x.shape[1]!=168: raise ValueError("Expected N by 168 history slots")
    if np.isinf(x).any() or (x[np.isfinite(x)]<0).any(): raise ValueError("History must be nonnegative or NaN")
    finite=np.isfinite(x)
    if ((finite.sum(axis=1)<24)|(~finite[:,-24:].any(axis=1))).any():
        raise ValueError("Historical information/freshness requirement failed")
    with np.errstate(over="raise"):
        return x.astype(np.float32,copy=True)


def unpack_quantiles(outputs,expected_count):
    if len(outputs)!=expected_count: raise ValueError("Wrong number of model outputs")
    values=[]
    for output in outputs:
        if hasattr(output,"detach"): output=output.detach().cpu().numpy()
        output=np.asarray(output)
        if output.shape!=(1,len(QUANTILES),24) or not np.isfinite(output).all():
            raise ValueError("Nonfinite or incorrectly shaped model output")
        values.append(output[0].copy())
    array=np.stack(values) if values else np.empty((0,len(QUANTILES),24),np.float32)
    return np.maximum(array,0),int((array<0).sum())


def load_pipeline():
    global _configured
    for name in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_HUB_DISABLE_TELEMETRY","DO_NOT_TRACK"):
        if os.environ.get(name)!="1": raise ValueError("Offline/telemetry settings must be explicit")
    import torch
    from chronos import Chronos2Pipeline
    if not _configured:
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.manual_seed(42)
        _configured=True
    pipeline=Chronos2Pipeline.from_pretrained(str(MODEL),device_map="cpu",torch_dtype=torch.float32,
                 local_files_only=True,trust_remote_code=False,attn_implementation="eager")
    pipeline.model.eval()
    if pipeline.quantiles!=QUANTILES: raise ValueError("Unexpected quantile configuration")
    if pipeline.model_context_length<168 or pipeline.model_prediction_length<24:
        raise ValueError("Pinned model cannot support the declared task")
    return pipeline


def predict_histories(pipeline,histories):
    x=validate_histories(histories)
    if len(x)==0: return np.empty((0,len(QUANTILES),24),np.float32),0
    before=x.copy()
    outputs=pipeline.predict(x[:,None,:],prediction_length=24,context_length=168,batch_size=BATCH_SIZE,
                             cross_learning=False,limit_prediction_length=True)
    if not np.array_equal(x,before,equal_nan=True): raise ValueError("Inference mutated history")
    return unpack_quantiles(outputs,len(x))
