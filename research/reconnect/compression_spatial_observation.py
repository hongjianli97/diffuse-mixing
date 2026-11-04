"""Nested material observations and reference reuse; no candidate geometry."""
import numpy as np
from smooth_density_probe import analytic_reference
from variable_parent_observation import parent_mass_audit,observe_parent
from compression_reference import project_to_saved_cells


def fine_inputs(N,parent_ids,parent_mass):
    if N!=2*int(np.sqrt(len(parent_ids))) or len(parent_ids)!=len(parent_mass):
        raise ValueError('one exact 2x2 material refinement required')
    ref=analytic_reference(N,.25,.02);mass=ref['mass'];state=ref['state'];labels=ref['labels']
    rows,columns=np.indices((N,N));parent_index=((rows//2)*(N//2)+columns//2).ravel()
    child_sum,rounding,audit=parent_mass_audit(mass,parent_mass)
    return {'mass':mass,'c0':state,'ids':np.arange(N*N),'material_label_centers':labels,
        'rho0_point':ref['rho_point'],'parent_index':parent_index,'parent_id_per_child':np.asarray(parent_ids)[parent_index],
        'child_mass_sum':child_sum,'parent_mass_rounding_relative':rounding},audit


def reuse_reference(saved,N,mass):
    """Project saved column averages; never advance or modify a reference."""
    times=saved['snapshot_time'];result={}
    for M,steps in [(512,256),(1024,256),(1024,512),(1024,1024)]:
        prefix=f'n{M}-k{steps}';mf=saved[prefix+'_mass_1d'];B=saved[prefix+'_B_snapshots']
        result[prefix]=np.array([project_to_saved_cells(b,mf,float(t),.02,N,mass)[0] for b,t in zip(B,times)])
    return result


def restrict_path(states,mass,parent_mass):
    return np.array([observe_parent(c,mass,parent_mass) for c in states])
