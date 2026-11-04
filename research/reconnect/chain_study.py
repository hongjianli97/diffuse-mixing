"""Reconnect trajectory metadata, supplied exchange, ordered propagation, mixing.

Only a three-node algebra fixture is propagated. Saved fluid candidates are
read/audited, never advanced or admitted to fluid ORC.
"""
from pathlib import Path
import argparse,hashlib,json,platform
import numpy as np
import scipy
from scipy.linalg import expm
from trajectory_chain import TrajectorySamples,SI_UNITS,laplacian,reference_distances,propagate_supplied,operator_audit


def fixture():
    data=TrajectorySamples(time=[0.,.04,.1],positions=[[[0.],[.7],[1.5]],[[0.],[1.3],[2.]],[[0.],[1.4],[2.2]]],
        ids=['material-A','material-B','material-C'],mass=[1.,2.,3.],c0=[0.,.2,1.],units=SI_UNITS)
    pairs=np.array([[0,1],[1,2]]);g=np.array([[.8,.4],[0.,.9]])
    return data,pairs,g


def verify(root,source):
    entries=json.loads((source/'sha256.json').read_text())
    for name,h in entries.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise RuntimeError('prior evidence changed: '+name)


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/rebuild-stage3-hybrid-face';verify(root,source)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    config={'study':'LTN mainline reconnection, supplied-graph algebra fixture and read-only fluid admission audit',
        'physical_parameters_changed':False,'three_node_fixture_is_a_fluid_benchmark':False,
        'supplied_conductances':'prescribed toy graph values in kg/s; not inferred physical diffusion from positions',
        'position_role':'fixed ids, reference lengths, current neighbor mask demonstration; no material surface input',
        'current_neighbor_mask':'sampled endpoint separation < 1 m; g supplied independently on the allowed edges',
        'time_rule':'one piecewise-constant supplied graph per interval, backward Euler with chronological left multiplication',
        'reference_graph':'two reference edges, lengths from initial positions, distances never updated',
        'state':'fixed mass-average material states; toy graph has no continuum cell geometry',
        'rho_used_to_infer_mass_or_volume':False,'material_faces_required':False,'F_or_C_required':False,
        'fluid_orc_enabled':False,'new_fluid_transient_run':False,'new_fluid_parameter_or_mesh_search':False,'third_stage_physical_acceptance':False}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    data,pairs,g=fixture();answer=propagate_supplied(data,pairs,g);zero=propagate_supplied(data,pairs,np.zeros_like(g))
    d0,lengths=reference_distances(data,pairs);current_length=np.linalg.norm(data.positions[:,pairs[:,1]]-data.positions[:,pairs[:,0]],axis=2)
    allowed=current_length[:-1]<1.
    if np.any(g[~allowed]!=0):raise RuntimeError('toy supplied exchange violates declared current-neighbor mask')
    dt=np.diff(data.time);Qs=-answer['laplacians']/data.mass[None,:,None]
    exact0=expm(dt[0]*Qs[0]);exact1=expm(dt[1]*Qs[1]);ordered=exact1@exact0
    averaged=expm(np.einsum('s,sij->ij',dt,Qs));reverse_BE=answer['steps'][0]@answer['steps'][1]
    order={'Q_commutator_max_abs_per_s2':float(np.max(abs(Qs[0]@Qs[1]-Qs[1]@Qs[0]))),
        'chronological_BE_vs_reversed_BE_max_abs':float(np.max(abs(answer['P']-reverse_BE))),
        'piecewise_exact_ordered_vs_exp_integrated_Q_max_abs':float(np.max(abs(ordered-averaged))),
        'BE_vs_piecewise_exact_max_abs':float(np.max(abs(answer['P']-ordered))),
        'exact_comparison_is_graph_algebra_not_continuum_reference':True}
    with np.load(root/'runs/rebuild-stage3-noflux-probe/noflux-probe.npz') as z:
        graph=operator_audit(z['mass'],z['unchanged_Q'])
    with np.load(source/'hybrid-face-baseline.npz') as z:hybrid=operator_audit(z['mass'],z['operator_Q'])
    old=json.loads((root/'runs/rebuild-stage3-noflux-probe/results.json').read_text());h=json.loads((source/'results.json').read_text())
    backend={'saved_optimized_graph':{'algebra_audit':graph,'saved_frozen_rate_error':old['all_cells']['relative_mass_weighted_rate_error'],
            'physical_acceptance':False,'fluid_admitted':False,'reason':'saved independent physical probe fails; nonnegative graph algebra is insufficient'},
        'hybrid_material_face':{'algebra_audit':hybrid,'saved_frozen_rate_error':h['hybrid_all_cells']['relative_mass_weighted_rate_error'],
            'physical_acceptance':False,'fluid_admitted':False,'reason':'raw generator has substantial negative offdiagonal; mass-average constitutive reference only'},
        'one_dimensional_ordinary_FV':{'role':'validated benchmark/control branch for recorded one-dimensional cases',
            'general_point_trajectory_backend':False,'fluid_orc_admitted':False,'extra_inputs':'true moving material endpoints and density/resistance closure'},
        'new_fluid_transient_runs':0,'new_fluid_propagators_constructed':0,'new_fluid_orc_evaluations':0}
    result={'graph_fixture_kernel_audit':answer['kernel_audit'],'graph_fixture_metrics':answer['metrics'],
        'graph_fixture_energy_balances':answer['energy_balances'],'graph_fixture_time_order':order,
        'D_zero_control':{'interpretation':'supplied exchange is zero while prescribed positions change; no physical g(D) law asserted',
            'maximum_state_change':float(np.max(abs(zero['states']-data.c0))),
            'final_I':zero['metrics'][-1]['I'],'final_H':zero['metrics'][-1]['H']},
        'fixed_reference_distance_0_2_m':float(d0[0,2]),'final_current_separation_0_2_m':float(abs(data.positions[-1,2,0]-data.positions[-1,0,0])),
        'reference_edge_0_1_retained_after_current_exchange_zero':bool(g[1,0]==0 and d0[0,1]==lengths[0]),
        'backend_admission':backend,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    np.savez_compressed(output/'supplied-graph-chain.npz',time=data.time,positions=data.positions,ids=data.ids,mass=data.mass,c0=data.c0,
        exchange_union_pairs=pairs,conductances=g,current_neighbor_mask=allowed,reference_pairs=pairs,reference_edge_lengths=lengths,reference_distances=d0,
        states=answer['states'],P=answer['P'],step_operators=answer['steps'],laplacians=answer['laplacians'],
        forward_row_mass_kernel=answer['forward_row_mass_kernel'],forward_column_component_mass_map=answer['forward_column_component_mass_map'],
        zero_exchange_states=zero['states'],zero_exchange_P=zero['P'],piecewise_exact_ordered_P=ordered,exp_integrated_Q=averaged,reversed_BE_P=reverse_BE)
    for name,value in [('results.json',result),('backend-admission.json',backend),('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':4})]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
