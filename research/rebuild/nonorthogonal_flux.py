"""Frozen-face scalar flux diagnostics using true mass-centroid states.

Two-point scalar conductances and one linear scalar reconstruction reference
are kept separate. No coefficient is fitted to a manufactured flux outcome;
no negative generator entries are clipped or used in a production backend.
"""
import numpy as np
from scipy.linalg import expm


def internal_face_geometry(history,positions,mass_centroids):
    x=np.asarray(positions,dtype=float);center=np.asarray(mass_centroids,dtype=float)
    if x.shape!=(len(history.vertex_ids),2) or center.shape!=(len(history.cell_ids),2):
        raise ValueError('matching frozen vertex and mass-centroid arrays required')
    if not np.isfinite(x).all() or not np.isfinite(center).all():raise ValueError('finite geometry required')
    edges=np.flatnonzero(~history.boundary_edge_mask);incidence=history.incidence[:,edges]
    owner=np.argmax(incidence==1,axis=0);other=np.argmax(incidence==-1,axis=0)
    a,b=history.edge_vertex_indices[edges].T;tangent=x[b]-x[a];length=np.linalg.norm(tangent,axis=1)
    if np.any(length<=0):raise ValueError('nonpositive edge length')
    normal=np.column_stack((tangent[:,1],-tangent[:,0]))/length[:,None]
    displacement=center[other]-center[owner];distance=np.linalg.norm(displacement,axis=1)
    normal_distance=np.sum(displacement*normal,axis=1)
    if np.any(normal_distance<=0) or np.any(distance<=0):raise ValueError('mass centroids do not lie on the expected face sides')
    tangential=displacement-normal_distance[:,None]*normal
    tangent_distance=np.linalg.norm(tangential,axis=1)
    return {'edge_indices':edges,'owner':owner,'other':other,'incidence':incidence,
            'length':length,'normal':normal,'displacement':displacement,'distance':distance,
            'normal_distance':normal_distance,'tangential_displacement':tangential,
            'angle_radians':np.arctan2(tangent_distance,normal_distance),
            'any_nonnegative_scalar_relative_error_lower_bound':tangent_distance/distance}


def two_point_flux_matrix(geometry,face_coefficient,cells,mode):
    coefficient=np.asarray(face_coefficient,dtype=float)
    if coefficient.shape!=geometry['length'].shape or not np.isfinite(coefficient).all() or np.any(coefficient<0):
        raise ValueError('finite nonnegative integrated face coefficient required')
    if mode=='normal_distance':distance=geometry['normal_distance']
    elif mode=='center_distance':distance=geometry['distance']
    else:raise ValueError('unknown two-point distance convention')
    weight=coefficient/distance;F=np.zeros((len(weight),cells));edge=np.arange(len(weight))
    F[edge,geometry['owner']]=weight;F[edge,geometry['other']]=-weight
    return F,weight


def scalar_reconstruction_reference(geometry,face_coefficient,centroids):
    """Inverse-distance weighted linear scalar reconstruction, not F/C fitting."""
    coefficient=np.asarray(face_coefficient,dtype=float)
    if coefficient.shape!=geometry['length'].shape or not np.isfinite(coefficient).all() or np.any(coefficient<0):
        raise ValueError('finite nonnegative integrated face coefficient required')
    center=np.asarray(centroids);n=len(center);operator=np.zeros((n,2,n));conditions=[]
    neighbours=[set() for _ in range(n)]
    for i,j in zip(geometry['owner'],geometry['other']):neighbours[i].add(int(j));neighbours[j].add(int(i))
    for i,nb in enumerate(neighbours):
        js=np.array(sorted(nb));d=center[js]-center[i]
        if len(js)<2 or np.linalg.matrix_rank(d)<2:raise ValueError('rank-deficient scalar reconstruction; no fallback')
        w=1/np.sum(d*d,axis=1);H=d.T@(d*w[:,None])
        B=np.linalg.solve(H,d.T*w);conditions.append(float(np.linalg.cond(H)))
        for k,j in enumerate(js):operator[i,:,j]=B[:,k]
        operator[i,:,i]=-B.sum(axis=1)
    pair_gradient=(operator[geometry['owner']]+operator[geometry['other']])/2
    F=-coefficient[:,None]*np.einsum('ed,edc->ec',geometry['normal'],pair_gradient)
    return F,operator,{'maximum_geometric_reconstruction_condition':max(conditions)}


def closed_operator_audit(flux_matrix,geometry,mass,probe_time=1e-4):
    m=np.asarray(mass,dtype=float)
    if m.ndim!=1 or np.any(m<=0) or not np.isfinite(m).all():raise ValueError('positive fixed mass vector required')
    L=geometry['incidence']@flux_matrix;Q=-L/m[:,None]
    off=Q.copy();np.fill_diagonal(off,np.inf);i,j=np.unravel_index(np.argmin(off),off.shape)
    P=expm(probe_time*Q)
    report={'generator_constant_residual':float(np.max(abs(Q.sum(axis=1)))),
            'generator_mass_residual':float(np.max(abs(m@Q))),
            'mass_weighted_symmetry_defect':float(np.max(abs(L-L.T))),
            'largest_mass_weighted_energy_eigenvalue':float(np.linalg.eigvalsh(-(L+L.T)/2).max()),
            'minimum_offdiagonal_generator':float(off[i,j]),
            'strict_negative_offdiagonal_count':int(np.count_nonzero(off<0)),
            'probe_time':probe_time,'minimum_raw_propagator_entry':float(P.min()),
            'propagator_constant_residual':float(np.max(abs(P.sum(axis=1)-1))),
            'propagator_mass_residual':float(np.max(abs(m@P-m)))}
    initial=np.zeros(len(m));initial[j]=1.;response=P@initial
    report['basis_counterexample']={'initial_cell':int(j),'observed_cell':int(i),
             'initial_observed_value':0.,'initial_derivative_at_observed_cell':float(Q[i,j]),
             'value_at_probe_time':float(response[i]),'minimum_response':float(response.min())}
    return report,{'L':L,'Q':Q,'P':P,'basis_initial':initial,'basis_response':response}
