"""Independent periodic density-contact material means and frozen references.

All analytic geometry/derivatives stay in the input/reference producer.
q=+rho*D*grad(c) below, so rho*c_t=div(q); physical Fick flux is -q.
"""
import numpy as np
from scipy.special import roots_legendre
from trajectory_chain import TrajectorySamples,SI_UNITS

AMPLITUDE=.2
PHASE=np.pi/5
WAVE=2*np.pi


def validate(N,D):
    if not isinstance(N,(int,np.integer)) or N<4 or N%2:
        raise ValueError('even N>=4 with both density contacts on material faces required')
    if not np.isscalar(D) or not np.isfinite(D) or D<=0:
        raise ValueError('positive finite D for the resistance-coordinate reference required')


def point_fields(a,b,D=.02):
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<0) or np.any(a>1):
        raise ValueError('finite coordinates, a in [0,1] required; no coordinate repair')
    validate(4,D)
    rho=np.where(a<.5,1.,4.);resistance=.5/D+.5/(4*D)
    xi=np.where(a<.5,a/D,(.5/D)+(a-.5)/(4*D))/resistance
    slope=1/(rho*D*resistance);phi=WAVE*xi+PHASE
    cos_a=np.cos(phi);sin_a=np.sin(phi);cos_b=np.cos(WAVE*b);sin_b=np.sin(WAVE*b)
    c=.5+AMPLITUDE*cos_a*cos_b
    ca=-AMPLITUDE*WAVE*slope*sin_a*cos_b;cb=-AMPLITUDE*WAVE*cos_a*sin_b
    rate_a=-AMPLITUDE*D*WAVE**2*slope**2*cos_a*cos_b
    rate_b=-AMPLITUDE*D*WAVE**2*cos_a*cos_b
    return {'rho':rho,'xi':xi,'resistance':resistance,'xi_a':slope,'phase':phi,'c':c,'c_a':ca,'c_b':cb,
        'q_a':rho*D*ca,'q_b':rho*D*cb,'rate_a':rate_a,'rate_b':rate_b,'rate':rate_a+rate_b}


def contact_inputs(N=128,D=.02):
    validate(N,D);axis=(np.arange(N)+.5)/N
    labels=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    v=point_fields(labels[:,0],labels[:,1],D);mass=v['rho']/N**2
    c0=.5+AMPLITUDE*np.sinc(v['xi_a']/N)*np.sinc(1/N)*np.cos(v['phase'])*np.cos(WAVE*labels[:,1])
    arrays={'mass':mass,'c0':c0,'ids':np.arange(N*N),'material_label_centers':labels,'rho0_point':v['rho'],
        'point_scalar_reference_only':v['c'],'true_cell_area_reference_only':np.full(N*N,1/N**2)}
    # Two identical records satisfy the shared trajectory schema. No evolution.
    data=TrajectorySamples([0.,.05],np.broadcast_to(labels,(2,N*N,2)),arrays['ids'],mass,c0,SI_UNITS,
        rho=np.broadcast_to(v['rho'],(2,N*N)))
    return data,arrays


def analytic_face_reference(N,D,labels,mass):
    validate(N,D);a=np.asarray(labels);m=np.asarray(mass);half=1/(2*N)
    if a.shape!=(N*N,2) or m.shape!=(N*N,) or not np.isfinite(m).all() or np.any(m<=0):
        raise ValueError('matching positive saved masses and contact-aligned cell labels required')
    x=a[:,0];y=a[:,1];v=point_fields(x,y,D);R=v['resistance']
    left=point_fields(x-half,y,D)['phase'];right=point_fields(x+half,y,D)['phase']
    y_cos_integral=np.sinc(1/N)*np.cos(WAVE*y)/N
    x_cos_integral=np.sinc(v['xi_a']/N)*np.cos(v['phase'])/N
    q_right=-AMPLITUDE*WAVE/R*np.sin(right)*y_cos_integral
    q_left=-AMPLITUDE*WAVE/R*np.sin(left)*y_cos_integral
    q_top=-AMPLITUDE*WAVE*v['rho']*D*np.sin(WAVE*(y+half))*x_cos_integral
    q_bottom=-AMPLITUDE*WAVE*v['rho']*D*np.sin(WAVE*(y-half))*x_cos_integral
    rate_a=(q_right-q_left)/m;rate_b=(q_top-q_bottom)/m
    return {'right_q_integral':q_right,'left_q_integral':q_left,'top_q_integral':q_top,'bottom_q_integral':q_bottom,
        'rate_a':rate_a,'rate_b':rate_b,'rate':rate_a+rate_b}


def volume_reference(N,D,labels,mass,order):
    validate(N,D)
    if not isinstance(order,(int,np.integer)) or order<2:raise ValueError('Gauss order>=2 required')
    a=np.asarray(labels);m=np.asarray(mass)
    if a.shape!=(N*N,2) or m.shape!=(N*N,) or not np.isfinite(m).all() or np.any(m<=0):
        raise ValueError('matching finite contact-aligned cells and positive saved masses required')
    z,w=roots_legendre(order);offset=z/(2*N);weights=w[:,None]*w[None,:]/(4*N*N)
    values={key:[] for key in ['mass','state_numerator','rate_a_numerator','rate_b_numerator','c_rate_integral','physical_dissipation']}
    for start in range(0,len(a),128):
        centers=a[start:start+128];x=centers[:,0,None,None]+offset[None,:,None];y=centers[:,1,None,None]+offset[None,None,:]
        v=point_fields(x,y,D);measure=v['rho']*weights[None,:,:]
        integrands={'mass':np.ones_like(v['c']),'state_numerator':v['c'],
            'rate_a_numerator':v['rate_a'],'rate_b_numerator':v['rate_b'],
            'c_rate_integral':v['c']*v['rate'],'physical_dissipation':D*(v['c_a']**2+v['c_b']**2)}
        for key,integrand in integrands.items():values[key].append(np.sum(measure*integrand,axis=(1,2)))
    result={key:np.concatenate(parts) for key,parts in values.items()}
    result['state']=result['state_numerator']/m
    result['rate_a']=result['rate_a_numerator']/m;result['rate_b']=result['rate_b_numerator']/m
    result['rate']=result['rate_a']+result['rate_b']
    return result


def contact_regions(labels,h=.04):
    a=np.asarray(labels)[:,0]
    if not np.isfinite(h) or h<=0 or h>=.125:raise ValueError('positive h with separate interface strips required')
    distance=np.minimum(np.minimum(a,1-a),abs(a-.5))
    near=distance<=2*h
    return {'distance_to_contact':distance,'near_contact':near,'far_from_contact':~near}


def regional_metrics(mass,candidate,reference,regions,rate_a,rate_b):
    m=np.asarray(mass);error=np.asarray(candidate)-reference;norm2=float(m@(reference*reference));error2=float(m@(error*error))
    result={}
    for key,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',regions['near_contact']),('far_from_contact',regions['far_from_contact'])]:
        squared=float(m[mask]@(error[mask]**2))
        result[key]={'cells':int(mask.sum()),'mass_kg':float(m[mask].sum()),
            'error_over_global_reference_norm':float(np.sqrt(squared/norm2)),
            'reference_over_global_reference_norm':float(np.sqrt(float(m[mask]@(reference[mask]**2))/norm2)),
            'candidate_over_global_reference_norm':float(np.sqrt(float(m[mask]@(candidate[mask]**2))/norm2)),
            'error_squared_fraction_of_global':squared/error2 if error2 else None,
            'maximum_absolute_rate_error_per_s':float(np.max(abs(error[mask]))),
            'reference_a_rate_over_global_reference_norm':float(np.sqrt(float(m[mask]@(rate_a[mask]**2))/norm2)),
            'reference_b_rate_over_global_reference_norm':float(np.sqrt(float(m[mask]@(rate_b[mask]**2))/norm2))}
    return result
