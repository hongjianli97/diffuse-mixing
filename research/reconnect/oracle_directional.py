"""Normal-first identity analysis of a saved oracle; no model correction."""
import numpy as np
from oracle_path_reference import path_gamma
from contact_continuous import gauss,normal_intervals,transverse_moments,check_h
from contact_frozen_probe import point_fields,AMPLITUDE,validate


def directional_response(x,h,D,s_order,z_order):
    validate(4,D);check_h(h);nodes,weights=gauss(s_order);gauss(z_order)
    f0=float(np.cos(point_fields(float(x),0.,D)['phase']));Ja=0.;Jb=0.
    for left,right in normal_intervals(float(x),h):
        s=(left+right)/2+(right-left)/2*nodes
        f=np.cos(point_fields((float(x)+s)%1.,0.,D)['phase'])
        K=path_gamma(float(x),s,D);b0,bk=transverse_moments(s,h,z_order)
        Ja+=float((right-left)/2*np.dot(weights,K*b0*(f-f0)))
        Jb+=float((right-left)/2*np.dot(weights,K*f*(bk-b0)))
    return Ja,Jb


def observe_integrals(mass,y_integrals,cell_integrals):
    m=np.asarray(mass,dtype=float);y=np.asarray(y_integrals,dtype=float);I=np.asarray(cell_integrals,dtype=float);N=len(y)
    if m.shape!=(N*N,) or I.shape!=(2,N) or not np.isfinite(m).all() or np.any(m<=0) or not np.isfinite(y).all() or not np.isfinite(I).all():
        raise ValueError('positive original masses, finite saved transverse integrals and two column integrals required')
    numerator=(2*AMPLITUDE*y[None,:,None]*I[:,None,:]).reshape(2,N*N)
    return numerator/m[None,:],numerator


def material_components(mass,saved,h,D,quadrature):
    x=np.asarray(saved['outer_x']);w=np.asarray(saved['outer_weight']);column=np.asarray(saved['outer_column'])
    y=np.asarray(saved['y_cosine_integrals']);N=len(y);order=quadrature['cell_x_order'];gauss(order)
    if x.ndim!=1 or x.shape!=w.shape or x.shape!=column.shape or len(x)%order or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<=0) or column.dtype.kind not in 'iu' or np.any(column<0) or np.any(column>=N):
        raise ValueError('unmodified positive saved quadrature records required')
    responses=np.array([directional_response(float(value),h,D,quadrature['normal_displacement_order'],quadrature['transverse_order']) for value in x])
    I=np.zeros((2,N))
    for start in range(0,len(x),order):
        stop=start+order;j=int(column[start])
        if not np.all(column[start:stop]==j):raise ValueError('saved material column changed within a quadrature panel')
        for axis in [0,1]:I[axis,j]+=float(w[start:stop]@responses[start:stop,axis])
    rates,numerator=observe_integrals(mass,y,I)
    return {'rate_a':rates[0],'rate_b':rates[1],'normal_column_integrals':I,
        'component_mass_rate_numerators':numerator,'outer_directional_responses':responses}


def gram_budget(mass,errors,total_error,pde_total,near):
    m=np.asarray(mass,dtype=float);e=np.asarray(errors,dtype=float);total=np.asarray(total_error,dtype=float);ref=np.asarray(pde_total,dtype=float);near=np.asarray(near,dtype=bool)
    if m.ndim!=1 or e.shape!=(3,len(m)) or any(a.shape!=m.shape for a in [total,ref,near]) or not np.isfinite(m).all() or np.any(m<=0) or any(not np.isfinite(a).all() for a in [e,total,ref]):
        raise ValueError('three finite error vectors and matching positive mass/reference/regions required')
    norm2=float(m@(ref*ref))
    if norm2<=0:raise ValueError('nonzero original global PDE norm required')
    result={}
    for key,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]:
        G=np.array([[float(m[mask]@(e[i,mask]*e[j,mask]))/norm2 for j in range(3)] for i in range(3)])
        norms=np.sqrt(np.diag(G));actual=float(m[mask]@(total[mask]**2))/norm2
        closure=total-e.sum(axis=0)
        result[key]={'cells':int(mask.sum()),'gram_over_global_reference_norm_squared':G.tolist(),
            'normal_error_norm':float(norms[0]),'transverse_error_norm':float(norms[1]),'evaluation_residual_norm':float(norms[2]),
            'saved_total_error_norm':float(np.sqrt(actual)),
            'signed_twice_cross_terms':{'normal_transverse':float(2*G[0,1]),'normal_evaluation':float(2*G[0,2]),'transverse_evaluation':float(2*G[1,2])},
            'squared_norm_identity_residual':actual-float(G.sum()),
            'vector_identity_residual_norm':float(np.sqrt(float(m[mask]@(closure[mask]**2))/norm2)),
            'normal_transverse_correlation':float(G[0,1]/(norms[0]*norms[1])) if norms[0]*norms[1]>0 else None}
    return result
