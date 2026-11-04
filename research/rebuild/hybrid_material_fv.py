"""Mass-centroid hybrid finite-volume energy baseline on true material faces.

One scalar per material cell (mass average), one massless trace per real face.
Coefficient kappa_K=D*m_K/V_K and stabilization A_e/d_Ke are fixed physical /
geometric choices. No probe fitting, coefficient clipping or post-symmetrizing.
This HMM-type baseline is not assumed monotone or accepted for fluid ORC.
"""
import numpy as np
from material_polygon_history import signed_area


def build_hybrid_fv(topology,mass,centroids,D):
    m=np.asarray(mass,dtype=float);b=np.asarray(centroids,dtype=float);C=np.asarray(topology['incidence'])
    p=np.asarray(topology['positions']);cv=np.asarray(topology['cell_vertices']);ev=np.asarray(topology['edge_vertices'])
    n,E=C.shape
    if m.shape!=(n,) or b.shape!=(n,2) or not np.isfinite(m).all() or np.any(m<=0) or not np.isfinite(b).all():
        raise ValueError('matching positive physical masses and finite true mass centroids required')
    if not np.isfinite(D) or D<0:raise ValueError('finite nonnegative scalar D required')
    volume=signed_area(p[cv]);a,z=ev.T;tangent=p[z]-p[a];length=np.linalg.norm(tangent,axis=1)
    if np.any(volume<=0) or np.any(length<=0):raise ValueError('positive true volume and face area required')
    normal=np.column_stack((tangent[:,1],-tangent[:,0]))/length[:,None];face_center=(p[a]+p[z])/2
    boundary=np.flatnonzero(topology['boundary_mask']);boundary_sign=C[:,boundary].sum(axis=0)
    Hcc=np.zeros((n,n));Hcf=np.zeros((n,E));Hfc=np.zeros((E,n));Hff=np.zeros((E,E))
    local_H=np.zeros((n,5,5));local_G=np.zeros((n,2,5));local_R=np.zeros((n,4,5));cell_faces=np.empty((n,4),dtype=int)
    distances=np.empty((n,4));kappa=D*m/volume;closure=[];identity=[]
    for cell in range(n):
        faces=np.flatnonzero(C[cell]!=0)
        if len(faces)!=4:raise ValueError('four real faces per material cell required')
        outward=C[cell,faces,None]*normal[faces];d=face_center[faces]-b[cell];dn=np.sum(d*outward,axis=1)
        if np.any(dn<=0):raise ValueError('mass centroid not strictly inside real face half-planes; no repair')
        area_normal=outward*length[faces,None]
        G=np.zeros((2,5));G[:,1:]=area_normal.T/volume[cell];G[:,0]=-G[:,1:].sum(axis=1)
        R=np.column_stack((-np.ones(4),np.eye(4)))-d@G
        weight=length[faces]/dn
        H=kappa[cell]*(volume[cell]*(G.T@G)+R.T@(weight[:,None]*R))
        cell_faces[cell]=faces;distances[cell]=dn;local_H[cell]=H;local_G[cell]=G;local_R[cell]=R
        closure.append(float(np.max(abs(area_normal.sum(axis=0)))))
        identity.append(float(np.max(abs(area_normal.T@d-volume[cell]*np.eye(2)))))
        Hcc[cell,cell]+=H[0,0];Hcf[cell,faces]+=H[0,1:];Hfc[faces,cell]+=H[1:,0]
        Hff[np.ix_(faces,faces)]+=H[1:,1:]
    diagnostics={'method':'mass-centroid hybrid material FV, fixed unit stabilization',
        'density_closure':'true cell mass / true cell volume, piecewise constant kappa=rho_bar*D',
        'stabilization_factor':1.,'maximum_area_normal_closure':max(closure),
        'maximum_divergence_gradient_identity_residual':max(identity),'zero_diffusion':bool(D==0),
        'posthoc_symmetrization_or_clipping':False,'accepted_physical_backend':False}
    if D==0:
        T=np.zeros((E,n));W=np.zeros((E,len(boundary)));L=np.zeros((n,n));trace_determined=False
        diagnostics['face_block_condition_number']=None
    else:
        T=-np.linalg.solve(Hff,Hfc);J=np.eye(E)[:,boundary];W=-np.linalg.solve(Hff,J)
        L=Hcc+Hcf@T;trace_determined=True
        diagnostics['face_block_condition_number']=float(np.linalg.cond(Hff))
        diagnostics['maximum_raw_trace_elimination_residual']=float(np.max(abs(Hff@T+Hfc)))
        diagnostics['maximum_boundary_response_elimination_residual']=float(np.max(abs(Hff@W+J)))
    # Compute real outward face flux from the local energy derivative. For an
    # internal face choose its +incidence owner; do not average two residuals.
    all_outward=np.zeros((n,4,n));all_boundary_response=np.zeros((n,4,len(boundary)))
    for cell,faces in enumerate(cell_faces):
        all_outward[cell]=-local_H[cell,1:,1:]@T[faces]
        all_outward[cell,:,cell]-=local_H[cell,1:,0]
        all_boundary_response[cell]=-local_H[cell,1:,1:]@W[faces]
    F=np.zeros((E,n));B=np.zeros((E,len(boundary)));shared_residual=[]
    for e in range(E):
        positive=np.flatnonzero(C[:,e]==1);owners=np.flatnonzero(C[:,e]!=0)
        owner=int(positive[0] if len(positive) else owners[0]);slot=int(np.flatnonzero(cell_faces[owner]==e).item())
        F[e]=C[owner,e]*all_outward[owner,slot];B[e]=C[owner,e]*all_boundary_response[owner,slot]
        if len(owners)==2:
            other=int(owners[owners!=owner][0]);other_slot=int(np.flatnonzero(cell_faces[other]==e).item())
            shared_residual.append(float(np.max(abs(all_outward[owner,slot]+all_outward[other,other_slot]))))
    Q=-L/m[:,None];boundary_rate=-Hcf@W/m[:,None]
    diagnostics['maximum_shared_face_operator_residual']=max(shared_residual,default=0.)
    diagnostics['maximum_cell_flux_vs_schur_operator_residual']=float(np.max(abs(C@F-L)))
    return {'trace_from_cells':T,'trace_from_boundary':W,'L':L,'Q':Q,'boundary_rate':boundary_rate,
        'face_flux_from_cells':F,'face_flux_from_boundary':B,'cell_outward_flux_from_cells':all_outward,
        'cell_outward_flux_from_boundary':all_boundary_response,'cell_faces':cell_faces,'boundary_faces':boundary,
        'boundary_sign':boundary_sign,'volume':volume,'kappa':kappa,'face_area':length,'face_center':face_center,
        'normal':normal,'distance_to_faces':distances,'local_energy':local_H,'local_gradient':local_G,'local_residual':local_R,
        'face_block':Hff,'cell_face_block':Hcf,'face_cell_block':Hfc,'trace_determined':trace_determined,'diagnostics':diagnostics}


def hybrid_action(operator,state,boundary_outward_flux=None):
    c=np.asarray(state,dtype=float);h=np.zeros(len(operator['boundary_faces'])) if boundary_outward_flux is None else np.asarray(boundary_outward_flux,dtype=float)
    if c.shape!=(operator['Q'].shape[1],) or h.shape!=(len(operator['boundary_faces']),) or not np.isfinite(c).all() or not np.isfinite(h).all():
        raise ValueError('finite cell mass averages and matching outward boundary flux required')
    if operator['diagnostics']['zero_diffusion'] and np.any(h!=0):raise ValueError('nonzero diffusive boundary flux incompatible with D=0')
    trace=operator['trace_from_cells']@c+operator['trace_from_boundary']@h if operator['trace_determined'] else None
    return {'rate':operator['Q']@c+operator['boundary_rate']@h,
        'face_flux':operator['face_flux_from_cells']@c+operator['face_flux_from_boundary']@h,
        'trace':trace}
