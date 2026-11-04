"""Continuous integral of the unchanged kernel, reference/analysis only.

This does not build or correct any candidate graph. The physical continuous
scalar is integrated first, then rho-weighted onto original material cells.
"""
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre
from positive_kernel_2d import slope_over_radius
from contact_frozen_probe import point_fields,AMPLITUDE,WAVE,validate


@lru_cache(maxsize=None)
def gauss(order):
    if not isinstance(order,(int,np.integer)) or order<2:raise ValueError('integer Gauss order>=2 required')
    return roots_legendre(order)


def check_h(h):
    if not np.isscalar(h) or not np.isfinite(h) or not 0<h<.125:
        raise ValueError('positive h with separated periodic contact supports required')


def transverse_moments(s,h,order):
    check_h(h);s=np.asarray(s,dtype=float)
    if not np.isfinite(s).all() or np.any(abs(s)>2*h):raise ValueError('normal displacement inside raw support required')
    z,w=gauss(order);width=np.sqrt((2*h-abs(s))*(2*h+abs(s)))
    # Both moments are even in z. Integrate [0,width] and multiply by two;
    # this resolves the radial centre without spending half the nodes twice.
    transverse=width[...,None]*(z+1)/2
    B=slope_over_radius(np.sqrt(s[...,None]**2+transverse**2),h)
    b0=width*np.sum(B*w,axis=-1)
    bk=width*np.sum(B*np.cos(WAVE*transverse)*w,axis=-1)
    return b0,bk


def normal_intervals(x,h):
    check_h(h)
    if not np.isscalar(x) or not np.isfinite(x) or not 0<=x<=1:raise ValueError('reference target in [0,1] required')
    cuts=[-2*h,0.,2*h]
    for contact in [-.5,0.,.5,1.,1.5]:
        s=contact-x
        if -2*h<s<2*h:cuts.append(float(s))
    edges=sorted(set(cuts))
    return list(zip(edges[:-1],edges[1:]))


def normal_response(x,h,D,s_order,z_order):
    validate(4,D);gauss(s_order);gauss(z_order)
    v=point_fields(float(x),0.,D);gamma=float(v['rho'])*D;f0=float(np.cos(v['phase']))
    nodes,weights=gauss(s_order);total=0.
    for left,right in normal_intervals(float(x),h):
        s=(left+right)/2+(right-left)/2*nodes
        # Wrapping belongs only to this periodic continuous reference.
        neighbor=point_fields((float(x)+s)%1.,0.,D);other=neighbor['rho']*D
        H=2*gamma*other/(gamma+other);f=np.cos(neighbor['phase'])
        b0,bk=transverse_moments(s,h,z_order)
        total+=float((right-left)/2*np.dot(weights,H*(f*bk-f0*b0)))
    return total


def continuous_point_rate(x,y,h,D,s_order,z_order):
    v=point_fields(x,y,D)
    return 2*AMPLITUDE*np.cos(WAVE*y)/float(v['rho'])*normal_response(x,h,D,s_order,z_order)


def cell_intervals(column,N,h):
    validate(N,.02);check_h(h)
    if not isinstance(column,(int,np.integer)) or not 0<=column<N:raise ValueError('valid original material column required')
    left=column/N;right=(column+1)/N;cuts=[left,right]
    for contact in [0.,.5,1.]:
        for value in [contact,(contact-2*h)%1.,(contact+2*h)%1.]:
            if left<value<right:cuts.append(float(value))
    edges=sorted(set(cuts))
    return list(zip(edges[:-1],edges[1:]))


def material_means(N,mass,labels,h,D,quadrature):
    validate(N,D);check_h(h)
    m=np.asarray(mass,dtype=float);a=np.asarray(labels,dtype=float)
    if m.shape!=(N*N,) or a.shape!=(N*N,2) or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('positive original saved masses and labels required')
    axis=(np.arange(N)+.5)/N;expected=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    if not np.array_equal(a,expected):raise ValueError('same original regular aligned material observation cells required')
    x_order=quadrature['cell_x_order'];s_order=quadrature['normal_displacement_order'];z_order=quadrature['transverse_order']
    nodes,weights=gauss(x_order);gauss(s_order);gauss(z_order)
    integrals=np.zeros(N);raw_x=[];raw_weight=[];raw_column=[];raw_response=[]
    for column in range(N):
        for left,right in cell_intervals(column,N,h):
            x=(left+right)/2+(right-left)/2*nodes;w=(right-left)/2*weights
            J=np.array([normal_response(float(value),h,D,s_order,z_order) for value in x])
            integrals[column]+=float(w@J)
            raw_x.extend(x);raw_weight.extend(w);raw_column.extend([column]*len(x));raw_response.extend(J)
    y_integral=np.sinc(1/N)*np.cos(WAVE*axis)/N
    numerator=2*AMPLITUDE*y_integral[:,None]*integrals[None,:]
    rate=numerator.ravel()/m
    return {'rate':rate,'normal_cell_integrals':integrals,'component_mass_rate_numerator':numerator.ravel(),
        'outer_x':np.array(raw_x),'outer_weight':np.array(raw_weight),'outer_column':np.array(raw_column,dtype=int),
        'outer_normal_response':np.array(raw_response),'y_cosine_integrals':y_integral}
