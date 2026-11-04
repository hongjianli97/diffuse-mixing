"""An analytic support obstruction, not a fitted or repaired diffusion method.

On a uniform sheared grid, nonnegative neighbour rates have a cross/transverse
second-moment ratio in [k-R hx/hy, k+R hx/hy]. Zero outside this interval rules
out isotropic quadratic consistency for that stencil. Passing is only necessary.
"""
from pathlib import Path
import argparse,json
import numpy as np
from material_polygon_history import MaterialPolygonHistory,POLYGON_UNITS,signed_area
from becker_polygon_benchmark import quad_connectivity
from nonorthogonal_flux import internal_face_geometry,closed_operator_audit
from vertex_mpfa import build_vertex_mpfa


def sheared_stencil_certificate(hx,hy,shear,radius=1):
    if not np.isfinite([hx,hy,shear]).all() or hx<=0 or hy<=0 or radius<1 or int(radius)!=radius:
        raise ValueError('finite positive spacings and integer column reach required')
    lower=shear-radius*hx/hy;upper=shear+radius*hx/hy
    return {'hx':hx,'hy':hy,'shear':shear,'maximum_column_reach':radius,
        'cross_to_transverse_second_moment_interval':[lower,upper],
        'zero_cross_moment_is_possible_by_this_necessary_condition':bool(lower<=0<=upper),
        'necessary_column_reach_from_cross_moment':int(np.ceil(abs(shear)*hy/hx)),
        'sufficiency_or_positive_operator_construction_claimed':False,
        'assumptions':'constant density, homogeneous affine grid; interior support p=-R..R, q=-1,0,1; nonnegative rates and nonzero transverse variance'}


def control():
    # One homogeneous algebraic control, independent of the Becker results.
    nx=ny=5;hx=1/16;hy=.5;shear=.6;D=.02
    positions=np.array([[i*hx+shear*j*hy,j*hy] for j in range(ny+1) for i in range(nx+1)])
    cells=quad_connectivity(nx,ny);ids=np.array([f'v{i}' for i in range(len(positions))]);mass=signed_area(positions[cells])
    history=MaterialPolygonHistory([0.,1.],np.stack((positions,positions)),ids,np.array([f'c{i}' for i in range(len(cells))]),
        ids[cells],mass,np.full(len(cells),.5),POLYGON_UNITS,{'geometry_kind':'material_quadrilateral_cells'})
    centers=positions[cells].mean(axis=1);a,b=history.edge_vertex_indices.T
    coefficient=np.repeat((D*.5*np.linalg.norm(positions[b]-positions[a],axis=1))[:,None],2,axis=1)
    operator=build_vertex_mpfa(history,positions,centers,coefficient);geometry=internal_face_geometry(history,positions,centers)
    audit,arrays=closed_operator_audit(operator['F'],geometry,mass);cell=12;Q=arrays['Q'];d=centers-centers[cell]
    first=Q[cell]@d;second=.5*np.einsum('j,jd,je->de',Q[cell],d,d)
    neighbours=np.flatnonzero(Q[cell]);neighbours=neighbours[neighbours!=cell]
    certificate=sheared_stencil_certificate(hx,hy,shear)
    result={'scope':'one 25-cell homogeneous algebraic control, not a second Becker run or convergence scan',
      'physical_acceptance':False,'D':D,'interior_observed_cell':cell,'certificate':certificate,
      'maximum_first_moment_residual':float(np.max(abs(first))),
      'second_moment_tensor':second.tolist(),'target_isotropic_tensor':(D*np.eye(2)).tolist(),
      'maximum_second_moment_residual':float(np.max(abs(second-D*np.eye(2)))),
      'interior_row_offdiagonal_rates':{str(j):float(Q[cell,j]) for j in neighbours},
      'minimum_interior_offdiagonal_rate':float(Q[cell,neighbours].min()),
      'all_cross_row_support_products_dx_dy_strictly_positive':bool(np.all(np.prod(d[neighbours][d[neighbours,1]!=0],axis=1)>0)),
      'weight_fitting_or_clipping_performed':False,'closed_operator':audit}
    return result,{'positions':positions,'mass':mass,'centroids':centers,'observed_cell':cell,
        'displacements':d,'first_moment':first,'second_moment':second,'F':operator['F'],**arrays}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if (out/'stencil-moments.json').exists() or (out/'stencil-moments.npz').exists():raise RuntimeError('refusing to overwrite moment control')
    result,arrays=control();np.savez_compressed(out/'stencil-moments.npz',**arrays)
    (out/'stencil-moments.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
