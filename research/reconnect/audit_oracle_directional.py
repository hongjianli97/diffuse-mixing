"""Saved-operator directional budget: original bytes, signed Gram and raw closure audit."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_contact_constitutive import digest
from oracle_directional import directional_response,observe_integrals,gram_budget
from compression_midpoint import weighted_norm


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-oracle-path'
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
    protocol=json.loads((source/'stage3-status.json').read_text())['next_oracle_directional_budget_predeclared']
    assert cfg['protocol']==protocol and cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    data_path=root/protocol['source'];ref_path=root/protocol['PDE_component_source']
    assert digest(data_path)==cfg['input_source_sha256'] and digest(ref_path)==cfg['PDE_component_source_sha256']
    with np.load(out/'oracle-directional.npz') as z,np.load(data_path) as old_data,np.load(ref_path) as ref:
        for key in ['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate','harmonic_fine_rate','near_contact','far_from_contact','distance_to_contact']:
            np.testing.assert_array_equal(z[key],old_data[key])
        for axis in ['a','b']:
            np.testing.assert_array_equal(z['PDE_rate_'+axis],ref['face_rate_'+axis])
            np.testing.assert_array_equal(z['PDE_'+axis+'_face_volume16_difference'],ref['face_rate_'+axis]-ref['volume16_rate_'+axis])
            np.testing.assert_array_equal(z['PDE_'+axis+'_volume8_16_difference'],ref['volume8_rate_'+axis]-ref['volume16_rate_'+axis])
        np.testing.assert_array_equal(z['reference_rate'],z['PDE_rate_a']+z['PDE_rate_b'])
        m=z['mass'];pde=z['reference_rate'];near=z['near_contact'];norm=weighted_norm(m,pde)
        for name,quad in zip(['coarse','fine'],protocol['fixed_quadrature_levels']):
            row=r['directional_runs'][name];assert row['quadrature']==quad
            total=z[name+'_saved_total_rate'];np.testing.assert_array_equal(total,old_data[name+'_rate'])
            for key in ['outer_x','outer_weight','outer_column','y_cosine_integrals']:
                np.testing.assert_array_equal(z[name+'_'+key],old_data[name+'_'+key])
            x=z[name+'_outer_x'];w=z[name+'_outer_weight'];col=z[name+'_outer_column'];J=z[name+'_outer_directional_responses']
            assert len(x)==row['outer_quadrature_points'];order=quad['cell_x_order'];I=np.zeros((2,128))
            for start in range(0,len(x),order):
                stop=start+order;j=int(col[start]);np.testing.assert_array_equal(col[start:stop],j)
                for axis in [0,1]:I[axis,j]+=float(w[start:stop]@J[start:stop,axis])
            np.testing.assert_array_equal(z[name+'_normal_column_integrals'],I)
            rates,numerator=observe_integrals(m,z[name+'_y_cosine_integrals'],I)
            np.testing.assert_array_equal(z[name+'_component_mass_rate_numerators'],numerator)
            np.testing.assert_array_equal(rates[0],z[name+'_rate_a']);np.testing.assert_array_equal(rates[1],z[name+'_rate_b'])
            samples=sorted(set([0,len(x)-1,int(np.argmin(abs(x-.003))),int(np.argmin(abs(x-.497))),int(np.argmin(abs(x-.503))),int(np.argmin(abs(x-.75)))]))
            for i in samples:np.testing.assert_array_equal(J[i],directional_response(float(x[i]),.04,.02,quad['normal_displacement_order'],quad['transverse_order']))
            errors=np.array([rates[0]-z['PDE_rate_a'],rates[1]-z['PDE_rate_b'],total-rates[0]-rates[1]])
            np.testing.assert_array_equal(z[name+'_error_vectors'],errors)
            np.testing.assert_array_equal(z[name+'_total_error'],total-pde)
            np.testing.assert_array_equal(z[name+'_error_identity_residual'],total-pde-errors.sum(axis=0))
            assert gram_budget(m,errors,total-pde,pde,near)==row['budget']
            assert [float(m@v) for v in [rates[0],rates[1],total]]==row['raw_component_mass_rates_kg_per_s']
            assert [-float(m@(z['c0']*v)) for v in [rates[0],rates[1],errors[2]]]==row['projected_chi_components_kg_per_s']
            assert -float(m@(z['c0']*total))==row['saved_total_projected_chi_kg_per_s']
        for key,values in r['component_quadrature_differences'].items():
            diff=z['coarse_'+key]-z['fine_'+key];np.testing.assert_array_equal(z[key+'_rule_difference'],diff)
            for region,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]:
                assert weighted_norm(m[mask],diff[mask])/norm==values[region]
        for key,values in r['saved_PDE_component_controls'].items():
            for region,mask in [('all',np.ones(len(m),dtype=bool)),('near_contact',near),('far_from_contact',~near)]:
                assert weighted_norm(m[mask],z[key][mask])/norm==values[region]
    assert r['candidate_and_total_operator_unchanged'] and r['reference_only']
    assert r['new_candidate_graphs']==r['new_candidate_transient_runs']==r['new_reference_transient_runs']==0 and not r['finite_node_candidate_admitted'] and not r['fluid_orc_enabled']
    checks={'predeclared_normal_first_same_operator_scope_verified':True,'original_mass_c0_ids_PDE_and_saved_total_rates_unchanged':True,
        'original_saved_quadrature_records_and_material_mean_closure_verified':True,
        'raw_directional_response_samples_recomputed':True,'all_raw_components_error_vectors_and_evaluation_residuals_recomputed':True,
        'all_region_signed_Gram_cross_terms_and_vector_identities_recomputed':True,'component_quadrature_and_PDE_controls_recomputed':True,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False}
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
