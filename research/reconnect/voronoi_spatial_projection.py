"""Identity-checked analysis adapter to unchanged raw material-parent observation."""
import numpy as np
from variable_parent_observation import parent_mass_audit, observe_parent


def validate_record(data):
    m,mp=np.asarray(data['mass']),np.asarray(data['parent_mass'])
    ids,pids=np.asarray(data['ids']),np.asarray(data['parent_ids'])
    index,per_child=np.asarray(data['parent_index']),np.asarray(data['parent_id_per_child'])
    N=int(np.sqrt(len(mp)))
    if N*N!=len(mp) or len(m)!=4*len(mp):raise ValueError('one complete2x2 material refinement required')
    for a,n in [(ids,len(m)),(pids,len(mp))]:
        if a.shape!=(n,) or a.dtype.kind not in 'iu' or len(np.unique(a))!=n:raise ValueError('complete unique integer identities required')
    if index.shape!=m.shape or index.dtype.kind not in 'iu' or np.any(index<0) or np.any(index>=len(mp)):
        raise ValueError('valid original integer parent map required')
    row,col=np.indices((2*N,2*N));expected=((row//2)*N+col//2).ravel()
    if not np.array_equal(index,expected) or not np.array_equal(per_child,pids[index]):
        raise ValueError('material nesting and identity map disagree; no reordering or repair')
    child_sum,rounding,audit=parent_mass_audit(m,mp)
    return child_sum,rounding,audit


def project_path(states,data):
    validate_record(data)
    a=np.asarray(states,dtype=float)
    if a.ndim!=2 or a.shape[1]!=len(data['mass']) or not np.isfinite(a).all():
        raise ValueError('finite complete fine-state path required')
    return np.array([observe_parent(c,data['mass'],data['parent_mass']) for c in a])
