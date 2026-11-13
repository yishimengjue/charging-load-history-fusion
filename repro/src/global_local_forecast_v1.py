"""Unit-preserving preprocessing for pooled and local continued boosting."""

import numpy as np
import lightgbm as lgb


IDS=tuple(str(i) for i in range(1,10))
POWER_COLUMNS=np.array(list(range(168))+list(range(169,173))+
                       [j+k for j in (180,189,198) for k in range(4)])
CONFIG=dict(num_leaves=15,min_child_samples=40,learning_rate=.05,random_state=42,
    n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1,use_missing=True,
    zero_as_missing=False,objective='quantile',alpha=.5)


def target_scale(y):
    finite=np.asarray(y,float)
    finite=finite[np.isfinite(finite)]
    if not len(finite) or (finite<0).any(): raise ValueError('Invalid scale training targets')
    return max(float(np.quantile(finite,.9)),1.)


def transform(features,scale,depot,identity=True):
    if depot not in IDS or not np.isfinite(scale) or scale<1:
        raise ValueError('Unknown ID or invalid scale')
    x=np.asarray(features,float)
    if x.shape[-1]!=207: raise ValueError('Expected original S features')
    x=x.copy(); x[...,POWER_COLUMNS]/=scale
    columns=[x,np.full((*x.shape[:-1],1),np.log1p(scale))]
    if identity:
        onehot=np.zeros((*x.shape[:-1],9)); onehot[...,IDS.index(depot)]=1
        columns.append(onehot)
    return np.concatenate(columns,axis=-1)


def training_indices(data,month,cutoff_ns):
    index=data['train_'+month]
    if (data['origin_ns'][index]+24*3600*10**9>cutoff_ns).any():
        raise ValueError('Immature target admitted')
    observed=np.isfinite(data['truth'][index])
    if not observed.any(): raise ValueError('No labels')
    return index,observed


def fit_head(x,y,weight,trees,initial=None):
    model=lgb.LGBMRegressor(n_estimators=trees,**CONFIG)
    model.fit(x,y,sample_weight=weight,init_model=initial)
    return model.booster_


def prefix_unchanged(parent,child):
    p=parent.dump_model()['tree_info']
    c=child.dump_model()['tree_info']
    if c[:len(p)]!=p: raise ValueError('Continued fit altered parent trees')
    return len(p),len(c)
