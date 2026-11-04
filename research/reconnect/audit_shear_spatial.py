"""Inherited bytes and one fine frozen shear common-rate evidence audit."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_kernel2d import digest
from shear_probe import make_shear_samples,shear_gauss_reference,shear_face_reference,shear_geometry
from compression_midpoint import load_laplacian,weighted_norm
from shear_spatial_observation import compare_parent_rates
from variable_parent_observation import observe_parent,parent_mass_audit
from sparse_exchange_audit import audit_laplacian


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-shear-frozen'
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
    cfg=json.loads((out/'config.json').read_text());r=json.loads((out/'results.json').read_text());c=r['comparison']
    assert cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_shear_spatial_control_predeclared']
    for rel,h in cfg['input_sources_sha256'].items():assert digest(root/rel)==h
    fine_path=root/cfg['protocol']['fine_mass_c0_ids_source'];coarse_path=root/cfg['protocol']['coarse_mass_c0_ids_and_reference_source']
    with np.load(out/'shear-spatial.npz') as z,np.load(fine_path) as fine,np.load(coarse_path) as coarse:
        keys=['mass','c0','ids','material_label_centers','rho0_point','parent_index','parent_id_per_child']
        for key in keys:np.testing.assert_array_equal(z[key],fine[key])
        for key in ['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate','face8_rate','face16_rate','gauss8_rate','gauss16_rate']:
            np.testing.assert_array_equal(z['parent_'+key],coarse[key])
        m=z['mass'];mp=z['parent_mass'];refp=z['parent_reference_rate'];labels=z['material_label_centers'];c0=z['c0']
        child,rounding,mass_audit=parent_mass_audit(m,mp);assert mass_audit==r['parent_mass_roundoff_audit']
        np.testing.assert_array_equal(z['child_mass_sum'],child);np.testing.assert_array_equal(z['parent_mass_rounding_relative'],rounding)
        np.testing.assert_array_equal(z['parent_id_per_child'],z['parent_ids'][z['parent_index']])
        data,arrays,audit=make_shear_samples(labels,z['ids'],m,c0,z['rho0_point'],z['time'])
        np.testing.assert_array_equal(z['positions'],data.positions);np.testing.assert_array_equal(z['rho'],data.rho)
        for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
        for key,value in shear_geometry(256,labels,.25,.05).items():np.testing.assert_array_equal(z[key],value)
        for order in [8,16]:
            gauss=shear_gauss_reference(256,.25,.25,.02,labels,m,order);face=shear_face_reference(256,.25,.25,.02,labels,m,order)
            for prefix,control in [('gauss',gauss),('face',face)]:
                for key,value in control.items():np.testing.assert_array_equal(z[f'{prefix}{order}_{key}'],value)
        L=load_laplacian(z);assert L.shape==(65536,65536) and audit_laplacian(m,L)==r['sparse_graph_audit']
        np.testing.assert_array_equal(-(L@c0)/m,z['kernel_rate']);np.testing.assert_array_equal(z['reference_rate'],z['face16_rate'])
        np.testing.assert_array_equal(z['quadrature_area'],m/data.rho[1])
        np.testing.assert_array_equal(z['rate_error'],z['kernel_rate']-z['reference_rate'])
        observed,comparison=compare_parent_rates(mp,refp,z['parent_kernel_rate'],m,z['kernel_rate'])
        np.testing.assert_array_equal(observed,z['common_kernel_rate'])
        np.testing.assert_array_equal(observe_parent(z['reference_rate'],m,mp),z['common_reference_rate'])
        np.testing.assert_array_equal(z['common_rate_error'],observed-refp)
        for key,value in comparison.items():assert value==c[key]
        normp=weighted_norm(mp,refp);relp=lambda v:weighted_norm(mp,v)/normp;R=lambda v:observe_parent(v,m,mp)
        vectors={'coarse_face_vs_volume':z['parent_face16_rate']-z['parent_gauss16_rate'],
            'coarse_face8_vs16':z['parent_face8_rate']-z['parent_face16_rate'],
            'coarse_volume8_vs16':z['parent_gauss8_rate']-z['parent_gauss16_rate'],
            'fine_face_vs_volume_observed':R(z['face16_rate']-z['gauss16_rate']),
            'fine_face8_vs16_observed':R(z['face8_rate']-z['face16_rate']),
            'fine_volume8_vs16_observed':R(z['gauss8_rate']-z['gauss16_rate']),
            'reference_nesting':z['common_reference_rate']-refp,
            'coarse_pair_vs_CSR':z['coarse_pair_scatter_rate']-z['parent_kernel_rate'],
            'fine_pair_vs_CSR_observed':R(z['pair_scatter_rate']-z['kernel_rate'])}
        for key,vector in vectors.items():assert relp(vector)==c['empirical_indicator_components'][key]
        assert sum(c['empirical_indicator_components'].values())==c['combined_empirical_indicator']
        assert np.all(z['conductances']>0) and not r['fluid_orc_enabled'] and r['new_frozen_graphs']==1 and r['new_transient_runs']==0
        with np.load(out/'shear-spatial-observations.npz') as small:
            for key in small.files:np.testing.assert_array_equal(small[key],z[key])
    checks={'predeclared_single_shear_joint_level_verified':True,'saved_fine_mass_c0_ids_parent_map_and_coarse_data_unchanged':True,
        'full_Eulerian_and_material_face_Gauss8_16_arrays_recomputed':True,
        'raw_CSR_action_and_area_weights_verified':True,'signed_common_rates_paired_errors_and_all_indicator_components_recomputed':True,
        'small_observations_equal_raw_case':True,'coarse_graph_rebuilt':False,'new_transient_runs':0,'fluid_orc_enabled':False,'full_P_constructed':False}
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
