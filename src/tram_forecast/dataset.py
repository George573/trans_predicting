"""Compact sample identities; storage-backed loading will be added separately."""
from datetime import timedelta
import numpy as np
from .schema import SampleIdentity


def sample_index(routes,start,end):
    if end<=start: raise ValueError('end must follow start')
    return tuple(SampleIdentity(r,start+timedelta(days=c),start+timedelta(days=c+h-1))
                 for r in routes for c in range(21,(end-start).days)
                 for h in range(1,min(61,(end-start).days-c)+1))


def epoch_order(size,epoch,seed=67):
    if size<0 or epoch<0: raise ValueError('size and epoch must be nonnegative')
    return np.random.default_rng(seed+epoch).permutation(size)
