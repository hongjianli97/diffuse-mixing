"""A priori static material refinement for one known contact diffusion layer."""
import numpy as np
from subcell_control import split_faces_at_contact


def locally_refined_faces(parent_faces,D,t_min,resolution,growth=.2):
    """Preserve all parent faces; resolve contact widths by sqrt(D*t_min)/q.

    Use the monitor h(x)=(sqrt(D*t_min)+growth*|x|)/q. The logarithmic
    partition is made before solving, without inspecting any scalar result.
    """
    if not all(np.isfinite(v) and v>0 for v in (D,t_min,resolution,growth)):
        raise ValueError('positive finite D,time,resolution,growth required')
    base=split_faces_at_contact(parent_faces);ell=np.sqrt(D*t_min);pieces=[]
    for a,b in zip(base[:-1],base[1:]):
        negative=b<=0
        lo,hi=(-b,-a) if negative else (a,b)
        span=np.log1p(growth*(hi-lo)/(ell+growth*lo))
        n=max(1,int(np.ceil(span/np.log1p(growth/resolution))))
        pos=lo+(ell+growth*lo)/growth*np.expm1(np.linspace(0,span,n+1))
        points=-pos[::-1] if negative else pos
        # Define physical interval endpoints exactly as supplied, not a
        # post-solution geometry or mass correction.
        points[0]=a;points[-1]=b
        if not pieces:pieces.extend(points.tolist())
        else:pieces.extend(points[1:].tolist())
    faces=np.asarray(pieces)
    if np.any(np.diff(faces)<=0) or not np.all(np.isin(parent_faces,faces)):
        raise RuntimeError('invalid static local partition')
    return faces
