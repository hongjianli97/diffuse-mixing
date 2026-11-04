"""Audit the two fixed tensor rules using independent cell-major assembly."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from ideal_original_c0_pair_study import SOURCE,record,comparison,NORM
from ideal_material_pair_reference import cell_support_offsets
from oracle_path_reference import path_gamma
from positive_kernel_2d import slope_over_radius


def cell_major_control(data,rule_arrays,row,order):
    """Same Gauss points, one neighbor cell at a time, no target-block loop."""
    N=128;h=.04;D=.02;local=rule_arrays['local_node_offsets'];w=rule_arrays['weights_2D']
    prefix=f'row{row["target_id"]}_'
    points=rule_arrays[prefix+'target_nodes'];centers=rule_arrays[prefix+'neighbor_lifted_centers']
    neighbors=rule_arrays[prefix+'neighbor_indices'];target=row['target_index']
    cell_mass=[];absolute_sum=0.
    for center,j in zip(centers,neighbors):
        neighbor=center+local
        dx=neighbor[None,:,0]-points[:,None,0];dy=neighbor[None,:,1]-points[:,None,1]
        kernel=path_gamma(points[:,None,0],dx,D)*slope_over_radius(np.hypot(dx,dy),h)
        weighted=kernel*w[None,:]*(data['c0'][j]-data['c0'][target])
        point_values=np.sum(weighted,axis=1)
        cell_mass.append(2*float(np.dot(w,point_values)))
        absolute_sum+=2*float(np.sum(w[:,None]*abs(weighted)))
    total=math.fsum(cell_mass);rate=total/data['mass'][target]
    # A standard operation-count floating bound for reordering the very same
    # sampled integrand, not a new physical acceptance tolerance.
    eps=np.finfo(float).eps;operations=order**4+64
    gamma=operations*eps/(1-operations*eps)
    arithmetic_bound=gamma*absolute_sum/data['mass'][target]
    difference=rate-row['rate']
    assert abs(difference)<=arithmetic_bound
    stored=rule_arrays[prefix+'neighbor_signed_mass_rate_contributions']
    discrepancy=np.array(cell_mass)-stored
    assert abs(math.fsum(discrepancy))<=arithmetic_bound*data['mass'][target]
    return {'target_id':row['target_id'],'order':order,'cell_major_rate':rate,
        'cell_major_minus_primary_rate':difference,'same_integrand_rounding_bound':arithmetic_bound,
        'bound_not_a_physical_error_threshold':True,'maximum_per_cell_mass_rate_difference':float(np.max(abs(discrepancy)))}


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    cfg=read(out/'config.json');spec=read(SOURCE/'next-gate-spec.json')
    assert cfg['protocol']==spec and cfg['parent_gate_sha256']==digest(SOURCE/'next-gate-spec.json')
    assert (out/'source-gate-spec.json').read_bytes()==(SOURCE/'next-gate-spec.json').read_bytes()
    assert not spec['already_executed'] and spec['two_predeclared_tensor_Gauss_orders']==[8,16]
    assert read(out/'tests.json')=={'successful':True,'checks_run':6,'failures':0,'errors':0,'skipped':0}
    status=read(out/'stage3-status.json');execution=read(out/'execution.json');p=read(out/'input-provenance.json')
    assert status['recorded_checks_total']==417+6==423 and status['previous_checks_rerun']==0
    assert execution['study_exit_code']==execution['tests_exit_code']==0
    assert execution['candidate_oracle_reference_elapsed_seconds']<600 and execution['peak_RSS_MiB']<512
    assert digest(ROOT/spec['source_snapshot'])==spec['source_snapshot_sha256']==p['source_snapshot_sha256']
    assert digest(out/'original-inputs.npz')==p['input_artifact_sha256']
    assert digest(SOURCE/'saved-vector-inputs.npz')==p['source_saved_vector_inputs_sha256']
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(SOURCE/'saved-vector-inputs.npz',allow_pickle=False) as z:
        assert set(z.files)==set(data)
        for k in data:assert record(z[k])==record(data[k])==p['copied_original_array_records'][k],k
    assert (out/'source-point-pair.json').read_bytes()==(ROOT/'runs/reconnect-reconstructed-pair/main-c0-comparison.json').read_bytes()
    levels={};controls=[];all_evaluations=0
    for order in [8,16]:
        records=read(out/f'order{order}-records.json')
        assert records['order']==order and records['raw_weights_not_renormalized']
        with np.load(out/f'order{order}-integrals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        z=arrays['nodes_1D'];w=arrays['weights_1D'];local=arrays['local_node_offsets'];weights=arrays['weights_2D']
        assert len(z)==len(w)==order and len(weights)==order**2
        assert np.all(w>0) and np.all(weights>0)
        assert np.array_equal(weights,(w[:,None]*w[None,:]).ravel()/(4*128**2))
        X,Y=np.meshgrid(z,z,indexing='xy')
        assert np.array_equal(local,np.column_stack([X.ravel(),Y.ravel()])/(2*128))
        complete=[]
        for metadata in records['rows']:
            row=dict(metadata);identity=row['target_id'];prefix=f'row{identity}_';i=row['target_index'];r,c=divmod(i,128)
            assert int(data['ids'][i])==identity and row['original_mass']==data['mass'][i] and row['original_cbar']==data['c0'][i]
            for key,value in arrays.items():
                if key.startswith(prefix):row[key[len(prefix):]]=value
            offsets=row['neighbor_offsets']
            assert [list(v) for v in cell_support_offsets(128,.04)]==spec['neighbor_cell_offsets']==offsets.tolist()
            assert len(offsets)==420 and len(set(map(int,row['neighbor_indices'])))==420
            assert np.array_equal(row['neighbor_indices'],[((r+int(dy))%128)*128+(c+int(dx))%128 for dx,dy in offsets])
            assert np.array_equal(row['neighbor_ids'],data['ids'][row['neighbor_indices']])
            assert np.array_equal(row['neighbor_lifted_centers'],data['positions'][i]+offsets/128)
            assert np.array_equal(row['target_nodes'],data['positions'][i]+local)
            assert np.array_equal(row['target_weights'],weights)
            numerator=2*float(weights@row['target_node_inner_integrals'])
            assert numerator==row['raw_mass_rate_numerator'] and numerator/data['mass'][i]==row['rate']
            assert math.fsum(map(float,row['neighbor_signed_mass_rate_contributions']))==row['mass_rate_from_cell_sum_control']
            assert row['kernel_node_pairs_evaluated']==420*order**4 and row['negative_raw_B_count']==0
            assert row['minimum_raw_path_gamma']>0
            assert row['quadrature_area']==math.fsum(map(float,weights))
            all_evaluations+=row['kernel_node_pairs_evaluated']
            controls.append(cell_major_control(data,arrays,row,order));complete.append(row)
        assert [v['target_id'] for v in complete]==[63,64]
        levels[order]=complete
    assert all_evaluations==58490880
    result,budget_arrays=comparison(data,levels,read(out/'source-point-pair.json'))
    assert result==read(out/'results.json')
    with np.load(out/'signed-budget.npz',allow_pickle=False) as z:
        for k,v in budget_arrays.items():assert np.array_equal(v,z[k]),k
    assert result['fixed_global_PDE_norm']==NORM and result['primary_rule_predeclared']==16
    assert result['point_minus_ideal_pair_resolvable'] and result['ideal_minus_continuous_pair_resolvable']
    assert all(row['point_minus_ideal_sign_resolved'] and row['ideal_minus_continuous_sign_resolved'] for row in result['rows'])
    assert result['point_quadrature_projection_correlation']<0
    assert result['open_pair_ideal_mass_rate']!=0
    assert result['candidate_weights_or_saved_results_changed'] is False
    assert status['orders_executed']==[8,16] and status['new_ideal_original_c0_rows']==2
    assert status['new_reference_only_fixed_configurations']==1
    for key in ['new_candidate_weights','new_candidate_graphs','new_global_matrices','new_candidate_transient_runs','new_PDE_reference_solves','new_exact_algebra_certificate_groups']:
        assert status[key]==0,key
    assert status['global_point_pair_vs_projection_split']=='still_not_identified_by_two_rows'
    assert not status['finite_node_candidate_admitted'] and not status['fluid_orc_enabled']
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']
    prior=prior+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==71
    save(out/'preservation.json',preserved)
    audit={'source_protocol_and_all_original_array_hashes_verified':True,
        'known_material_geometry_and_density_oracle_reference_only':True,
        'raw_positive_Gauss_weights_and_complete_420_cell_support_verified':True,
        'all_58490880_four_dimensional_node_pairs_accounted_for':True,
        'independent_cell_major_reassembly_same_rules':controls,
        'primary_Gauss16_and_signed_three_term_budget_verified':True,
        'all_per_row_sign_separations_exceed_fixed_empirical_difference_sums':True,
        'opposing_terms_not_interpreted_as_global_causal_shares':True,
        'open_pair_and_shared_rounding_residuals_preserved':True,
        'new_software_checks':6,'recorded_checks_total':423,'previous_checks_rerun':0,
        'new_candidate_weights':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':71,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'reference-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    main(parser.parse_args().directory)
