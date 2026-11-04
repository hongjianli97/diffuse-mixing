"""A prescribed 2D continuity benchmark built from Becker material motion.

Given rotation and determinant-one shear stress the geometric ledger. They
are benchmark definitions, not fitted deformation gradients or ledger inputs.
This is not an unforced 2D Navier--Stokes/DNS validation flow.
"""
import numpy as np
from scipy.special import roots_legendre
from density_geometry_study import mass_coordinate_at_x
from material_polygon_history import signed_area


def quad_connectivity(nx=32,ny=2):
    return np.array([[j*(nx+1)+i,j*(nx+1)+i+1,(j+1)*(nx+1)+i+1,(j+1)*(nx+1)+i]
                     for j in range(ny) for i in range(nx)],dtype=int)


def _parameters(times,ndim,T):
    t=np.asarray(times);shape=(len(t),)+(1,)*(ndim-2)
    theta=np.pi/3*t/T;shear=.6*np.sin(np.pi*t/T)
    return (np.cos(theta).reshape(shape),np.sin(theta).reshape(shape),shear.reshape(shape),
            (.6*np.pi/T*np.cos(np.pi*t/T)).reshape(shape),t.reshape(shape))


def to_world(points,times,T=.1):
    c,s,k,_,t=_parameters(times,points.ndim,T)
    x=points[...,0]+k*points[...,1];y=points[...,1]
    return np.stack((c*x-s*y+.1*t/T,s*x+c*y-.05*t/T),axis=-1)


def from_world(points,times,T=.1):
    c,s,k,_,t=_parameters(times,points.ndim,T)
    px=points[...,0]-.1*t/T;py=points[...,1]+.05*t/T
    x=c*px+s*py;y=-s*px+c*py
    return np.stack((x-k*y,y),axis=-1)


def true_vertices(flow,mass_edges,y_edges,times,T=.1):
    x=flow.x_mu(mass_edges[None,:],times[:,None]);ny=len(y_edges)
    base=np.stack((np.tile(x,(1,ny)),np.broadcast_to(np.repeat(y_edges,len(mass_edges)),(len(times),ny*len(mass_edges)))),axis=-1)
    return to_world(base,times,T)


def density_velocity(flow,points,times,T=.1):
    canonical=from_world(points,times,T);xi=canonical[...,0];eta=canonical[...,1]
    c,s,_,kd,t=_parameters(times,points.ndim,T)
    z=flow.z_at_xi(flow.u1*t-xi);shock_u=flow.u(z)
    rho=flow.j/shock_u;stream_velocity=flow.u1-shock_u
    px=points[...,0]-.1*t/T;py=points[...,1]+.05*t/T
    vx=c*(stream_velocity+kd*eta)-(np.pi/(3*T))*py+.1/T
    vy=s*(stream_velocity+kd*eta)+(np.pi/(3*T))*px-.05/T
    return rho,np.stack((vx,vy),axis=-1)


def actual_polygon_mass(flow,positions,cell_vertices,times,T=.1,order=32):
    """Independent primitive/Gauss integral for benchmark parallelograms."""
    p=positions[:,cell_vertices];canonical=from_world(positions,times,T)[:,cell_vertices]
    closure=canonical[:,:,2]-canonical[:,:,1]-canonical[:,:,3]+canonical[:,:,0]
    if np.max(abs(closure))>1e-12:raise ValueError('reference integral requires the benchmark parallelogram structure')
    area=signed_area(p);origin=canonical[:,:,0,0]
    ex=canonical[:,:,1,0]-origin;ey=canonical[:,:,3,0]-origin
    if np.any(ex<=0):raise ValueError('reference x-direction must be nondegenerate and increasing')
    z,w=roots_legendre(order);v=(z+1)/2;weights=w/2
    left=origin[:,:,None]+ey[:,:,None]*v;right=left+ex[:,:,None]
    mu0=mass_coordinate_at_x(flow,left,times[:,None,None])
    mu1=mass_coordinate_at_x(flow,right,times[:,None,None])
    return area*np.sum((mu1-mu0)*weights,axis=-1)/ex


def direct_density_mass(flow,quad,t,T=.1,order=32):
    z,w=roots_legendre(order);u=(z+1)/2;weights=w/2
    p=quad[0]+u[:,None,None]*(quad[1]-quad[0])+u[None,:,None]*(quad[3]-quad[0])
    rho,_=density_velocity(flow,p[None,...],np.array([t]),T)
    return float(signed_area(quad)*np.sum(rho[0]*weights[:,None]*weights[None,:]))


def physical_edge_mass_flux(history,flow,times,T=.1,order=32):
    g=history.evaluate(times);x=g['positions'];velocity=g['vertex_velocity'];a,b=history.edge_vertex_indices.T
    edge=x[:,b]-x[:,a];z,w=roots_legendre(order);u=(z+1)/2;weights=w/2
    p=x[:,a,None,:]+u[None,None,:,None]*edge[:,:,None,:]
    mesh_velocity=velocity[:,a,None,:]+u[None,None,:,None]*(velocity[:,b,None,:]-velocity[:,a,None,:])
    rho,fluid=density_velocity(flow,p,times,T);relative=fluid-mesh_velocity
    normal_dot=relative[...,0]*edge[:,:,None,1]-relative[...,1]*edge[:,:,None,0]
    return np.sum(rho*normal_dot*weights,axis=-1)
