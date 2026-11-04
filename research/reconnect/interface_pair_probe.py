"""Independent material-cell reference and fixed-mass regional observations."""
from fractions import Fraction as F
import numpy as np


def quadratic_cell_reference(center,ell,rho,mass,D,origin):
    """Exact constant-phase square integrals, without changing given mass.

    Stored dyadic geometry/masses are exact; D is the declared decimal .02.
    This analytic reference is not an edge coefficient or new initial field.
    """
    values=[center,ell,rho,mass,D,origin]
    if any(not np.isscalar(v) or not np.isfinite(v) for v in values):raise ValueError('finite reference arguments required')
    a,w,p,m,d,o=[F(str(v)) for v in values]
    if min(w,p,m)<=0 or d<0:raise ValueError('positive cell width/rho/original mass and D>=0 required')
    left=a-w/2;right=a+w/2;gamma=p*d
    numerator=p*w*((right-o)**3-(left-o)**3)/6
    qleft=-gamma*(left-o);qright=-gamma*(right-o)
    mass_rate=w*(qleft-qright)
    exact={'left_face_m':left,'right_face_m':right,'implied_cell_mass_kg':p*w*w,
        'state_mass_numerator_kg':numerator,'mass_average':numerator/m,
        'left_q_normal':qleft,'right_q_normal':qright,
        'PDE_mass_rate_kg_per_s':mass_rate,'PDE_mass_average_rate_per_s':mass_rate/m}
    return {'exact':{key:str(v) for key,v in exact.items()},'values':{key:float(v) for key,v in exact.items()},
        'original_mass_matches_constant_phase_square':p*w*w==m}


def region_summary(masses,rates,references):
    m=np.asarray(masses,dtype=float);r=np.asarray(rates,dtype=float);ref=np.asarray(references,dtype=float)
    if m.ndim!=1 or not len(m) or r.shape!=m.shape or ref.shape!=m.shape or not all(np.isfinite(v).all() for v in [m,r,ref]) or np.any(m<=0):
        raise ValueError('matching finite rates/references and positive original masses required')
    total=float(m.sum());inventory=float(m@r);reference_inventory=float(m@ref)
    observed=inventory/total;reference=reference_inventory/total
    norm2=float(m@(ref*ref))
    if reference==0 or norm2<=0:raise ValueError('nonzero regional reference required')
    error=r-ref;direct_inventory_error=float(m@error)
    return {'total_original_mass_kg':total,'raw_region_mass_rate_kg_per_s':inventory,'PDE_region_mass_rate_kg_per_s':reference_inventory,
        'mass_average_rate_per_s':observed,'mass_average_reference_per_s':reference,
        'mass_average_error_per_s':observed-reference,'mass_average_relative_error':(observed-reference)/reference,
        'mass_weighted_local_relative_L2_error':float(np.sqrt(float(m@(error*error))/norm2)),
        'arithmetic_mean_diagnostic_only_per_s':float(r.mean()),
        'arithmetic_mean_relative_error_diagnostic_only':float((r.mean()-ref.mean())/ref.mean()),
        'direct_region_inventory_error_kg_per_s':direct_inventory_error,
        'inventory_error_identity_residual_kg_per_s':(inventory-reference_inventory)-direct_inventory_error,
        'original_masses_unchanged':True,'arithmetic_mean_is_material_observation':False}
