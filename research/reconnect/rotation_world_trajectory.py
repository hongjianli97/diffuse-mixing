"""Pure prescribed rational rotation/world periodic basis; no material scalar data."""
from fractions import Fraction as F
import numpy as np


def prescribed_world_trajectory(labels,ids,rho0):
    a=np.asarray(labels,dtype=float);ids=np.asarray(ids);rho=np.asarray(rho0,dtype=float)
    if a.shape!=(len(ids),2) or ids.dtype.kind not in 'iu' or len(set(map(int,ids)))!=len(ids):raise ValueError('complete pure labels andIDs required')
    if rho.shape!=(len(ids),) or not np.isfinite(a).all() or not np.isfinite(rho).all() or np.any(rho<=0):raise ValueError('finite positions andpositive givenrho required')
    points=tuple(tuple(F(float(v)) for v in p) for p in a);frames=[]
    for j in range(33):
        t=F(j,640);s=t/F(1,10);den=1+s*s;c=(1-s*s)/den;v=2*s/den
        E=((c,-v),(v,c));world=[[(E[0][0]*x+E[0][1]*y),(E[1][0]*x+E[1][1]*y)] for x,y in points]
        boundary={'origin_exact':['0','0'],'periods_exact':['1','1'],
          'orientation_rows_exact':[[str(q) for q in row] for row in E],
          'basis_columns_exact':[[str(c),str(v)],[str(-v),str(c)]]}
        frames.append({'frame':j,'time_exact':str(t),'boundary':boundary,
          'world_positions_exact':[[str(q) for q in p] for p in world],'rho':rho.tolist()})
    return {'ids':list(map(int,ids)),'frames':frames,'T_exact':'1/20','K':16,
      'law':'s=t/(2T),proper rationalrotation;theta=2atan(s),not constantangularspeed',
      'given_global_boundary_not_fitted_from_trajectory':True,'mass_scalar_reference_or_internal_faces_supplied':False}
