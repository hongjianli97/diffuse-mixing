"""Oracle-only path-resistance continuous reference; never a node backend.

The complete prescribed periodic rho(a) and its contact positions are extra
reference information. Existing graphs, masses and scalar states are untouched.
"""
import numpy as np
from contact_continuous import gauss,normal_intervals,cell_intervals,transverse_moments,check_h
from contact_frozen_probe import point_fields,AMPLITUDE,WAVE,validate


def path_gamma(x,s,D=.02):
    """Reciprocal path average of 1/(rho*D), with at most one contact.

    Displacement/contact comparisons avoid subtracting nearly equal primitive
    values. No fractions, states or weights are clipped or renormalized.
    """
    validate(4,D);x,s=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(s,dtype=float))
    if not np.isfinite(x).all() or not np.isfinite(s).all() or np.any(abs(s)>=.5):
        raise ValueError('finite periodic reference points and |normal displacement|<.5 required')
    a=x%1.;rho=np.where(a<.5,1.,4.);gamma=np.asarray(rho*D)
    next_face=np.floor(2*a)/2+.5;previous_face=np.floor(2*a)/2
    distance=np.where(s>=0,next_face-a,previous_face-a)
    crosses=((s>0)&(s>distance))|((s<0)&(s<distance))
    result=gamma.copy()
    if np.any(crosses):
        fraction=distance[crosses]/s[crosses]
        if np.any(fraction<0) or np.any(fraction>1):raise RuntimeError('invalid analytic path partition; no repair')
        opposite=np.where(rho[crosses]==1.,4.,1.)*D
        result[crosses]=1/(fraction/gamma[crosses]+(1-fraction)/opposite)
    return result


def normal_response(x,h,D,s_order,z_order):
    validate(4,D);check_h(h);nodes,weights=gauss(s_order);gauss(z_order)
    phi=point_fields(float(x),0.,D)['phase'];f0=float(np.cos(phi));total=0.
    for left,right in normal_intervals(float(x),h):
        s=(left+right)/2+(right-left)/2*nodes
        f=np.cos(point_fields((float(x)+s)%1.,0.,D)['phase'])
        K=path_gamma(float(x),s,D);b0,bk=transverse_moments(s,h,z_order)
        total+=float((right-left)/2*np.dot(weights,K*(f*bk-f0*b0)))
    return total


def continuous_point_rate(x,y,h,D,s_order,z_order):
    rho=float(point_fields(x,y,D)['rho'])
    return 2*AMPLITUDE*np.cos(WAVE*y)/rho*normal_response(x,h,D,s_order,z_order)


def material_means(N,mass,labels,h,D,quadrature):
    validate(N,D);check_h(h);m=np.asarray(mass,dtype=float);a=np.asarray(labels,dtype=float)
    if m.shape!=(N*N,) or a.shape!=(N*N,2) or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('positive original masses and original observation labels required')
    axis=(np.arange(N)+.5)/N;expected=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    if not np.array_equal(a,expected):raise ValueError('same regular aligned original material cells required')
    x_order=quadrature['cell_x_order'];s_order=quadrature['normal_displacement_order'];z_order=quadrature['transverse_order']
    nodes,weights=gauss(x_order);gauss(s_order);gauss(z_order)
    integrals=np.zeros(N);raw_x=[];raw_weight=[];raw_column=[];raw_response=[]
    for column in range(N):
        for left,right in cell_intervals(column,N,h):
            x=(left+right)/2+(right-left)/2*nodes;w=(right-left)/2*weights
            J=np.array([normal_response(float(value),h,D,s_order,z_order) for value in x])
            integrals[column]+=float(w@J)
            raw_x.extend(x);raw_weight.extend(w);raw_column.extend([column]*len(x));raw_response.extend(J)
    y=np.sinc(1/N)*np.cos(WAVE*axis)/N;numerator=2*AMPLITUDE*y[:,None]*integrals[None,:]
    return {'rate':numerator.ravel()/m,'normal_cell_integrals':integrals,'component_mass_rate_numerator':numerator.ravel(),
        'outer_x':np.array(raw_x),'outer_weight':np.array(raw_weight),'outer_column':np.array(raw_column,dtype=int),
        'outer_normal_response':np.array(raw_response),'y_cosine_integrals':y}
