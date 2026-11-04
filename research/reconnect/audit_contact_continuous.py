"""Inherited byte preservation and raw full-PDE frozen shear evidence audit."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_contact_frozen import digest
from contact_continuous import gauss,cell_intervals,normal_response
from contact_continuous_study import budget
from contact_frozen_probe import point_fields,AMPLITUDE
from compression_midpoint import weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-contact-frozen'
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
    protocol=json.loads((source/'stage3-status.json').read_text())['next_contact_continuous_control_predeclared']
    assert cfg['protocol_source_sha256']==digest(source/'stage3-status.json') and cfg['protocol']==protocol
    with np.load(out/'contact-continuous.npz') as z,np.load(source/'contact-observations.npz') as old_data:
        for key in ['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate','rate_error','near_contact','far_from_contact','distance_to_contact']:
            np.testing.assert_array_equal(z[key],old_data[key])
        m=z['mass'];pde=z['reference_rate'];discrete=z['kernel_rate'];near=z['near_contact'];norm=weighted_norm(m,pde)
        for name,quad in zip(['coarse','fine'],protocol['fixed_quadrature_levels']):
            row=r['continuous_runs'][name];assert row['quadrature']==quad
            order=quad['cell_x_order'];nodes,weights=gauss(order);cursor=0;integrals=np.zeros(128)
            x=z[name+'_outer_x'];w=z[name+'_outer_weight'];J=z[name+'_outer_normal_response'];columns=z[name+'_outer_column']
            for column in range(128):
                for left,right in cell_intervals(column,128,.04):
                    stop=cursor+order
                    np.testing.assert_array_equal(x[cursor:stop],(left+right)/2+(right-left)/2*nodes)
                    np.testing.assert_array_equal(w[cursor:stop],(right-left)/2*weights)
                    np.testing.assert_array_equal(columns[cursor:stop],column)
                    integrals[column]+=float(w[cursor:stop]@J[cursor:stop]);cursor=stop
            assert cursor==len(x)==row['outer_quadrature_points']
            # Re-evaluate a fixed small subset of raw response samples, not a new rule/physical case.
            samples=sorted(set([0,len(x)-1,int(np.argmin(abs(x-.003))),int(np.argmin(abs(x-.497))),int(np.argmin(abs(x-.503))),int(np.argmin(abs(x-.75)))]))
            for index in samples:assert normal_response(float(x[index]),.04,.02,quad['normal_displacement_order'],quad['transverse_order'])==J[index]
            np.testing.assert_array_equal(integrals,z[name+'_normal_cell_integrals'])
            axis=(np.arange(128)+.5)/128;y=np.sinc(1/128)*np.cos(2*np.pi*axis)/128
            np.testing.assert_array_equal(z[name+'_y_cosine_integrals'],y)
            numerator=(2*AMPLITUDE*y[:,None]*integrals[None,:]).ravel()
            np.testing.assert_array_equal(z[name+'_component_mass_rate_numerator'],numerator)
            rate=numerator/m;np.testing.assert_array_equal(z[name+'_rate'],rate)
            assert budget(m,discrete,rate,pde,near)==row['budget']
            assert float(m@rate)==row['raw_mass_rate_kg_per_s']
            assert -float(m@(z['c0']*rate))==row['projected_chi_kg_per_s']
            phi=point_fields(x,0.,.02)['phase'];chi=-AMPLITUDE**2*float(w@(np.cos(phi)*J))
            assert chi==row['physical_continuous_nonlocal_chi_kg_per_s'] and chi>=0.
        np.testing.assert_array_equal(z['best_continuous_rate'],z['fine_rate'])
        np.testing.assert_array_equal(z['quadrature_difference'],z['coarse_rate']-z['fine_rate'])
        for key,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]:
            assert weighted_norm(m[mask],z['quadrature_difference'][mask])/norm==r['quadrature_difference_controls'][key]
        np.testing.assert_array_equal(z['fixed_width_deviation'],z['fine_rate']-pde)
        np.testing.assert_array_equal(z['discrete_mean_closure_remainder'],discrete-z['fine_rate'])
        np.testing.assert_array_equal(z['vector_identity_residual'],discrete-pde-z['fixed_width_deviation']-z['discrete_mean_closure_remainder'])
    initial=(out/'initial-tests-source.py.txt').read_text();current=(root/'research/reconnect/test_contact_continuous.py').read_text()
    tolerances=lambda text:[line.strip() for line in text.splitlines() if 'self.assertLess(' in line]
    assert tolerances(initial)==tolerances(current)
    assert r['new_candidate_graphs']==r['new_candidate_transient_runs']==0 and not r['fluid_orc_enabled'] and not r['contact_transient_admitted']
    checks={'predeclared_fixed_h_and_two_rules_verified':True,'original_mass_c0_ids_candidate_and_PDE_rates_unchanged':True,
        'raw_quadrature_nodes_weights_and_material_mean_closure_verified':True,
        'fixed_response_samples_independently_recomputed':True,'all_regional_vectors_norms_signed_cross_terms_and_identities_recomputed':True,
        'raw_quadrature_differences_and_dissipation_recomputed':True,
        'initial_failed_selfcheck_and_pre_fix_sources_preserved':True,'all_test_assertion_tolerances_unchanged_after_parity_fix':True,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'full_P_constructed':False,'fluid_orc_enabled':False}
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
