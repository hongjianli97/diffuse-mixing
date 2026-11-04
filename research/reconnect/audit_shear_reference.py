"""Saved full 2D reference states, projections, physics and inherited bytes."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_kernel2d import digest
from shear_reference import ShearReferenceOperator,project_mass_average
from shear_probe import shear_point_fields
from compression_reference_study import metrics
from compression_midpoint import weighted_norm
from variable_parent_observation import observe_parent


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-shear-spatial'
    old=json.loads((source/'preservation.json').read_text());originals=json.loads(Path('/workspace/original-handoff-sha256.json').read_text())
    for rel,h in originals.items():assert digest(root/rel)==h,rel
    attachments=json.loads(Path('/workspace/setup-ltn/attachment-manifest.json').read_text())
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],str(p)
    formal=old['formal_scientific_manifests_verified']+[{'manifest':str((source/'sha256.json').relative_to(root)),
        'manifest_sha256':digest(source/'sha256.json'),'entries_verified':len(json.loads((source/'sha256.json').read_text()))}]
    for item in formal:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=json.loads(p.read_text());assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(root/rel)==h,(item['manifest'],rel)
    historical=old['historical_diagnostic_manifests_not_claimed_as_current_hash_matches']
    for item in historical:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=json.loads(p.read_text());assert len(entries)==item['entries']
        differences=[{'path':rel,'saved':h,'current':digest(root/rel)} for rel,h in entries.items() if digest(root/rel)!=h]
        assert sorted(differences,key=lambda v:v['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda v:v['path'])
    cfg=json.loads((out/'config.json').read_text());r=json.loads((out/'results.json').read_text())
    assert cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_shear_reference_predeclared']
    for rel,h in cfg['input_sources_sha256'].items():assert digest(root/rel)==h
    with np.load(out/'shear-reference.npz') as z:
        for N,path in zip([128,256],cfg['protocol']['saved_mass_c0_ids_sources']):
            with np.load(root/path) as old_input:
                for key in ['mass','c0','ids','material_label_centers']:np.testing.assert_array_equal(z[f'N{N}_{key}'],old_input[key])
        times=z['snapshot_time']
        for name,M in [('M63-k128',63),('M127-k128',127),('M127-k256',127)]:
            prefix=name+'_';op=ShearReferenceOperator(M);rho=z[prefix+'rho0_grid'];accepted=z[prefix+'accepted_point_states']
            np.testing.assert_array_equal(rho,op.rho);np.testing.assert_array_equal(accepted[0],op.initial_scalar())
            inventory=np.mean(rho[None,:,:]*accepted,axis=(1,2));mean=inventory/np.mean(rho)
            energy=.5*np.mean(rho[None,:,:]*(accepted-mean[:,None,None])**2,axis=(1,2))
            np.testing.assert_array_equal(z[prefix+'weighted_inventory'],inventory)
            np.testing.assert_array_equal(z[prefix+'point_half_variance_energy'],energy)
            np.testing.assert_array_equal(z[prefix+'accepted_extrema'],np.stack([accepted.min(axis=(1,2)),accepted.max(axis=(1,2))],axis=1))
            snapshots=z[prefix+'point_snapshots'];controls=[]
            for index,(t,c) in enumerate(zip(times,snapshots)):
                rate=op.rhs(float(t),c.ravel()).reshape(M,M);chi=op.dissipation(float(t),c)
                np.testing.assert_array_equal(z[prefix+'snapshot_rates'][index],rate)
                assert float(np.mean(rho*rate))==z[prefix+'snapshot_mass_rate'][index]
                assert float(np.mean(rho*c*rate)+chi)==z[prefix+'snapshot_energy_identity_residual'][index]
                frozen=op.rhs(float(t),op.initial_scalar().ravel()).reshape(M,M)
                analytic=shear_point_fields(op.a,op.b,.25,.25*t/.05,.02)[2]['rate']
                controls.append(float(np.max(abs(frozen-analytic))))
            assert controls==r['analytic_initial_profile_PDE_controls'][name]['analytic_initial_profile_full_PDE_max_difference_per_s_at_times']
            for N in [128,256]:
                m=z[f'N{N}_mass'];c0=z[f'N{N}_c0'];states=z[f'{name}_N{N}_states'];imag=z[f'{name}_N{N}_projection_imaginary']
                for index,c in enumerate(snapshots):
                    projected,imaginary=project_mass_average(c,rho,N,m)
                    np.testing.assert_array_equal(projected,states[index]);np.testing.assert_array_equal(imaginary,imag[index])
                    assert metrics(m,projected,c0)==r['reference_cases'][name]['observations'][str(N)]['metrics'][index]
        for N in [128,256]:
            m=z[f'N{N}_mass'];c0=z[f'N{N}_c0'];mu=m@c0/m.sum();sigma=weighted_norm(m,c0-mu)
            relative=lambda v:weighted_norm(m,v)/sigma
            space=z[f'M63-k128_N{N}_states']-z[f'M127-k128_N{N}_states'];temporal=z[f'M127-k128_N{N}_states']-z[f'M127-k256_N{N}_states']
            np.testing.assert_array_equal(z[f'N{N}_space_difference'],space);np.testing.assert_array_equal(z[f'N{N}_time_difference'],temporal)
            np.testing.assert_array_equal(z[f'N{N}_best_reference_states'],z[f'M127-k256_N{N}_states'])
            control=r['reference_uncertainty_controls'][str(N)]
            np.testing.assert_array_equal([relative(v) for v in space],control['space_63_vs127_relative_to_original_fluctuation'])
            np.testing.assert_array_equal([relative(v) for v in temporal],control['max_step_halving_127_relative_to_original_fluctuation'])
        nesting=np.array([observe_parent(c,z['N256_mass'],z['N128_mass']) for c in z['N256_best_reference_states']])-z['N128_best_reference_states']
        np.testing.assert_array_equal(z['reference_parent_nesting_difference'],nesting)
        with np.load(out/'shear-reference-observations.npz') as small:
            for key in small.files:np.testing.assert_array_equal(small[key],z[key])
    assert not r['fluid_orc_enabled'] and not r['new_candidate_graphs'] and not r['new_candidate_transient_runs']
    checks={'predeclared_three_reference_cases_and_original_inputs_verified':True,
        'accepted_point_inventory_energy_and_extrema_recomputed':True,
        'snapshot_PDE_rates_energy_identity_and_analytic_profile_controls_recomputed':True,
        'all_original_cell_Fourier_projections_and_imaginary_residues_recomputed':True,
        'space_time_difference_vectors_metrics_and_material_nesting_verified':True,'small_observations_equal_raw_arrays':True,
        'initial_diagnostic_list_error_preserved':(out/'tests-initial-diagnostic.log').is_file(),
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'fluid_orc_enabled':False}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'old_candidate_code_and_results_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
