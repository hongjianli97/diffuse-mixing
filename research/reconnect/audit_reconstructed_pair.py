"""Audit the fixed pair receipts, immutable inputs and historical provenance."""
from fractions import Fraction as F
from inspect import signature
from pathlib import Path
import argparse
import json
import math
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from reconstructed_path_rows import build_reconstructed_path_rows
from reconstructed_pair_study import SOURCE,array_record


def analytic_inverse_density(start,delta):
    # Extra-information control only; never called by coefficient construction.
    cuts={F(0),F(1)}
    if delta:
        for k in range(-2,5):
            t=(F(k,2)-start)/delta
            if 0<t<1:cuts.add(t)
    cuts=sorted(cuts)
    return sum(((b-a)/(1 if (start+(a+b)*delta/2)%1<F(1,2) else 4)
                for a,b in zip(cuts,cuts[1:])),F(0))


def main(output):
    out=Path(output).resolve();assert not (out/'sha256.json').exists()
    cfg=read(out/'config.json');protocol=read(SOURCE/'gate-spec.json')
    assert cfg['protocol']==protocol and cfg['source_protocol_sha256']==digest(SOURCE/'gate-spec.json')
    assert (out/'source-gate-spec.json').read_bytes()==(SOURCE/'gate-spec.json').read_bytes()
    assert (out/'source-stage3-status.json').read_bytes()==(SOURCE/'stage3-status.json').read_bytes()
    status=read(out/'stage3-status.json');result=read(out/'main-c0-comparison.json');edges=read(out/'edges.json')
    checks=read(out/'tests.json')
    assert checks=={'successful':True,'checks_run':7,'failures':0,'errors':0,'skipped':0}
    assert status['recorded_checks_total']==403+7==410 and status['previous_checks_rerun']==0
    assert status['new_candidate_graphs']==status['new_global_matrices']==status['new_candidate_transient_runs']==0
    assert status['new_reference_time_solutions']==status['new_flow_solves']==status['LP_solves']==0
    assert not status['finite_node_candidate_admitted'] and not status['fluid_orc_enabled']
    assert status['stage3_physical_acceptance']=='incomplete'
    params=set(signature(build_reconstructed_path_rows).parameters)
    assert not params.intersection({'c0','cbar','reference_rate','PDE_rate','true_faces','true_volumes','F_C','velocity_gradients'})
    provenance=read(out/'input-provenance.json')
    assert digest(ROOT/protocol['source_snapshot'])==protocol['source_snapshot_sha256']==provenance['source_snapshot_sha256']
    assert digest(out/'original-inputs.npz')==provenance['input_artifact_sha256']
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(ROOT/protocol['source_snapshot'],allow_pickle=False) as z:
        for key,record in provenance['original_members'].items():
            original=z[key][0] if key in ['positions','rho'] else z[key]
            assert array_record(original)==record==array_record(data[key]),key
    assert F(float(data['D']))==F(protocol['D_exact']) and F(float(data['h']))==F(protocol['h_exact'])
    assert data['periods'].tolist()==protocol['periods']
    lookup={int(v):i for i,v in enumerate(data['ids'])};edge_pairs=[tuple(e['ids']) for e in edges]
    assert edge_pairs==sorted(set(edge_pairs)) and len(edges)==663
    assert result['row_neighbor_counts']==[332,332]
    means={int(v):F(float(c)) for v,c in zip(data['ids'],data['c0'])}
    precise_rates={60:[],100:[]};recomputed=[];all_paths_analytic=True;max_path_float_error=0.
    for e in edges:
        left,right=e['ids'];i,j=lookup[left],lookup[right]
        path=e['path'];parts=path['intervals'];retained=set(e['retained_ids'])
        assert {left,right}<=retained
        assert parts[0]['start']=='0' and parts[-1]['stop']=='1'
        for a,b in zip(parts,parts[1:]):assert F(a['stop'])==F(b['start'])
        for v in parts:
            assert F(v['stop'])>F(v['start']) and v['source_id'] in retained
            assert F(v['density'])==F(float(data['rho'][lookup[v['source_id']]]))
        mean=sum(((F(v['stop'])-F(v['start']))/F(v['density']) for v in parts),F(0))
        assert mean==F(path['mean_inverse_density_exact'])>0
        dx=tuple(F(v) for v in e['displacement_exact'])
        assert sum((v*v for v in dx),F(0))==F(e['distance_squared_exact'])<=(2*F(protocol['h_exact']))**2
        all_paths_analytic &= mean==analytic_inverse_density(F(float(data['positions'][i,0])),dx[0])
        qi=F(float(data['mass'][i]))/F(float(data['rho'][i]))
        qj=F(float(data['mass'][j]))/F(float(data['rho'][j]))
        assert F(e['conductance_exact_given_binary_B_h'])==2*qi*qj*F(protocol['D_exact'])/mean*F(e['B_h_original_float'])>=0
        max_path_float_error=max(max_path_float_error,abs(e['mean_inverse_density_float_minus_exact_display']))
    assert all_paths_analytic
    norm=result['fixed_global_PDE_norm'];pair_mass=[]
    for row,side in zip(result['rows'],['left','right']):
        target=row['target_id'];i=lookup[target];pair_mass.append(data['mass'][i])
        neighbor_ids={v['id'] for v in row['neighbors']}
        assert neighbor_ids==set(map(int,data['ids'][data[side+'_columns']]))-{target}
        # Independent exact support enumeration for the original binary snapshot.
        support=set()
        origin=[F(float(v)) for v in data['positions'][i]]
        for identity,point in zip(data['ids'],data['positions']):
            d=[min(abs(F(float(v))-a),1-abs(F(float(v))-a)) for a,v in zip(origin,point)]
            if int(identity)!=target and sum(v*v for v in d)<=(2*F(protocol['h_exact']))**2:support.add(int(identity))
        assert neighbor_ids==support
        for neighbor in row['neighbors']:
            assert {target,neighbor['id']}==set(edges[neighbor['edge_index']]['ids'])
        mass=F(float(data['mass'][i]))
        action=sum((F(edges[v['edge_index']]['conductance_exact_given_binary_B_h'])*(means[v['id']]-means[target])
                    for v in row['neighbors']),F(0))/mass
        assert action==F(row['reconstructed_rate_exact_given_binary_B_h'])
        assert float(action)==row['reconstructed_rate']
        assert row['reference_rate']==data['reference_rate'][i] and row['saved_kernel_rate']==data['kernel_rate'][i]
        for p in (60,100):
            precise_rates[p].append(sum((F(edges[v['edge_index']][f'high_precision_{p}_conductance_exact_decimal'])*
                (means[v['id']]-means[target]) for v in row['neighbors']),F(0))/mass)
        recomputed.append(float(action)-row['reference_rate'])
    exact_precision_delta=np.array([float(a-b) for a,b in zip(precise_rates[60],precise_rates[100])])
    sub_float_precision_indicator=float(np.sqrt(np.array(pair_mass)@(exact_precision_delta**2)))/norm
    error=float(np.sqrt(np.array(pair_mass)@(np.array(recomputed)**2)))/norm
    assert error==result['reconstructed_pair_error']
    indicator=sum(result['empirical_difference_indicators'].values())
    assert indicator==result['sum_of_empirical_difference_indicators']
    assert result['benefit_old_minus_reconstructed']==result['saved_kernel_pair_error']-error
    if result['decision']=='resolvable_pair_improvement':assert result['benefit_old_minus_reconstructed']>indicator+sub_float_precision_indicator
    elif result['decision']=='resolvable_pair_deterioration':assert -result['benefit_old_minus_reconstructed']>indicator+sub_float_precision_indicator
    else:assert abs(result['benefit_old_minus_reconstructed'])<=indicator+sub_float_precision_indicator
    assert result['open_pair_mass_rate_display']!=0 # open pair, no conservation repair
    shared=result['shared_target_edge_index'];assert edges[shared]['ids']==[63,64]
    assert all(sum(v['edge_index']==shared for v in row['neighbors'])==1 for row in result['rows'])
    saved=read(ROOT/'runs/reconnect-FV-face-budget/source-main-c0-comparison.json')
    assert norm==saved['original_all_cell_mass_weighted_PDE_rate_norm']
    assert result['saved_kernel_pair_error']==saved['saved_kernel_two_row_error_contribution_over_global_norm']
    assert result['saved_FV_pair_error_context_only']==saved['FV_two_row_error_contribution_over_global_norm']
    preserved=preservation() # original files, attachments and four known historical differences
    known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    parent=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']
    parent=parent+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in parent:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,expected in entries.items():assert digest(ROOT/rel)==expected,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==67
    save(out/'preservation.json',preserved)
    audit={'source_protocol_and_original_member_hashes_verified':True,
        'full_support_matches_exact_enumeration_and_saved_kernel_columns':True,
        'all_663_path_integrals_match_extra_information_analytic_control_exactly':all_paths_analytic,
        'analytic_density_not_used_to_build_coefficients':True,
        'all_edge_constitutive_factors_and_exact_mass_mean_actions_verified':True,
        'shared_target_edge_and_open_pair_handling_verified':True,
        'max_inverse_density_float_reduction_absolute_error':max_path_float_error,
        '60_vs_100_digit_action_indicator_before_float_subtraction':sub_float_precision_indicator,
        'note':'main 60/100 display difference can round to zero; audit retains exact decimal-action difference',
        'new_software_checks':7,'recorded_checks_total':410,'previous_checks_rerun':0,
        'new_snapshot_diagnostics':1,'new_full_graphs_or_transients':0,
        'prior_formal_manifests_verified':67,'original_files_verified':782,'historical_attachments_verified':15,
        'finite_node_candidate_admitted':False,'fluid_orc_enabled':False}
    save(out/'receipt-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
