from __future__ import annotations
import numpy as np

def expected_calibration_error(confidences, correct, n_bins=10):
    conf=np.asarray(confidences,dtype=float)
    corr=np.asarray(correct,dtype=float)
    if len(conf)==0: return 0.0
    ece=0.0
    edges=np.linspace(0.0,1.0,n_bins+1)
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        mask=(conf>=lo)&(conf<=hi) if i==n_bins-1 else (conf>=lo)&(conf<hi)
        if np.any(mask): ece += np.mean(mask)*abs(np.mean(conf[mask])-np.mean(corr[mask]))
    return float(ece)

def risk_coverage_curve(confidences, correct):
    conf=np.asarray(confidences,dtype=float)
    corr=np.asarray(correct,dtype=float)
    if len(conf)==0: return np.array([]),np.array([])
    order=np.argsort(-conf)
    errors=1.0-corr[order]
    coverage=np.arange(1,len(errors)+1,dtype=float)/len(errors)
    risk=np.cumsum(errors)/np.arange(1,len(errors)+1)
    return coverage,risk
