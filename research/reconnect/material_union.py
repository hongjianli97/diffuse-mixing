"""Raw material unions and direct block observations; no weight/state repair."""
import math
import numpy as np


def group_sum(values,side,ratio):
    a=np.asarray(values,dtype=float)
    if not isinstance(side,int) or not isinstance(ratio,int) or side<1 or ratio<1:
        raise ValueError('positive integer material-block dimensions required')
    if a.shape!=((side*ratio)**2,) or not np.isfinite(a).all():raise ValueError('complete finite material vector required')
    return a.reshape(side,ratio,side,ratio).sum(axis=(1,3)).ravel()


def block_index(side,ratio):
    if side<1 or ratio<1:raise ValueError('positive block dimensions required')
    row,col=np.indices((side*ratio,side*ratio))
    return ((row//ratio)*side+col//ratio).ravel()


def union_labels(labels,ratio=2):
    a=np.asarray(labels,dtype=float);n=math.isqrt(len(a))
    if a.shape!=(n*n,2) or n%ratio or ratio<1:raise ValueError('complete nested material labels required')
    axis=(np.arange(n)+.5)/n;expected=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    if not np.array_equal(a,expected):raise ValueError('saved material-square label order required; no reordering')
    side=n//ratio
    return a.reshape(side,ratio,side,ratio,2).mean(axis=(1,3)).reshape(-1,2)


def oracle_density(labels):
    a=np.asarray(labels,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():raise ValueError('finite original label coordinates required')
    return 1+.25*np.cos(2*np.pi*a[:,0])


def define_union(mass,c0,reference_states,side=64,ratio=2):
    m=np.asarray(mass,dtype=float);c=np.asarray(c0,dtype=float);ref=np.asarray(reference_states,dtype=float)
    if m.shape!=c.shape or m.shape!=((side*ratio)**2,) or np.any(m<=0) or not np.isfinite(m).all() or not np.isfinite(c).all():
        raise ValueError('original positive masses and complete finite initial means required')
    if ref.ndim!=2 or ref.shape[1]!=len(m) or not np.isfinite(ref).all():raise ValueError('original material-reference snapshots required')
    parent=group_sum(m,side,ratio)
    initial=group_sum(m*c,side,ratio)/parent
    refs=np.array([group_sum(m*u,side,ratio)/parent for u in ref])
    return parent,initial,refs


def observe_path(values,mass,target_mass,side=64,ratio=2):
    a=np.asarray(values,dtype=float);m=np.asarray(mass,dtype=float);mp=np.asarray(target_mass,dtype=float)
    if a.ndim!=2 or a.shape[1]!=len(m) or m.shape!=((side*ratio)**2,) or mp.shape!=(side*side,):raise ValueError('complete native states and original union capacities required')
    if not np.isfinite(a).all() or not np.isfinite(m).all() or not np.isfinite(mp).all() or np.any(m<=0) or np.any(mp<=0):raise ValueError('finite states and positive actual capacities required')
    if ratio==1:
        if not np.array_equal(m,mp):raise ValueError('identity observation cannot change mass')
        return a.copy()
    return np.array([group_sum(m*c,side,ratio)/mp for c in a])
