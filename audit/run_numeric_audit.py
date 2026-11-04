"""Read-only checks of legacy operators; not a validation of the new method."""
from pathlib import Path
import sys,json,platform, warnings
import numpy as np, scipy
from scipy.spatial import Delaunay
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'reference_baseline/lhd'))
import netdiff as nd
rng=np.random.default_rng(2048);P=rng.random((45,2));tri=Delaunay(P).simplices
cond=np.exp(rng.uniform(-5,5,len(tri)));m=np.exp(rng.uniform(-1,1,len(P)));c=rng.random(len(P))
res={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
for mode,fn in [('element',nd.stiffness),('edge_harm',nd.stiffness_edge)]:
 K=fn(P,tri,cond);A=K.toarray();off=A.copy();np.fill_diagonal(off,0)
 dt=.02;y=spsolve(diags(m)+dt*K,m*c)
 E=lambda z:float(.5*np.sum(m*(z-np.dot(m,z)/m.sum())**2))
 res[mode]={'symmetry_max':float(np.max(abs(A-A.T))),'row_sum_max':float(np.max(abs(A.sum(1)))),'min_eigenvalue':float(np.linalg.eigvalsh(A).min()),'positive_offdiag_count':int(np.sum(off>1e-12)),'relative_mass_error':float(abs(m@(y-c))/(m@c)),'energy_before':E(c),'energy_after':E(y),'min_c':float(y.min()),'max_c':float(y.max())}
with warnings.catch_warnings(record=True) as w:
 warnings.simplefilter('always');K=nd.stiffness_edge(P,tri,np.zeros(len(tri)))
 res['zero_conductance']={'finite':bool(np.isfinite(K.data).all()),'max_abs':float(abs(K.data).max()) if K.nnz else 0,'warnings':[str(x.message) for x in w]}
# Variable smoothing radius neighbor enumeration in bcom.sph_diffusion misses pairs if only j sees i and i<j.
from scipy.spatial import cKDTree
p=np.array([[0.,0.],[1.,0.]]);h=np.array([.1,1.]);lst=cKDTree(p).query_ball_point(p,2*h)
I=np.repeat(np.arange(2),[len(x) for x in lst]);J=np.concatenate(lst);pairs=list(zip(I[I<J].tolist(),J[I<J].tolist()))
res['sph_pair_support_counterexample']={'legacy_pairs':pairs,'distance':1.,'mean_kernel_support':float(h.sum()),'pair_should_be_present':True}
(root/'audit/numeric_audit.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
