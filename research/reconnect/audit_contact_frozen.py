"""Inherited byte preservation and raw full-PDE frozen shear evidence audit."""
from pathlib import Path
import argparse,json
import numpy as np
from package_compression_midpoint import digest as streamed_digest
from contact_frozen_probe import contact_inputs,analytic_face_reference,volume_reference,contact_regions,regional_metrics
from compression_midpoint import pair_action
from compression_midpoint import load_laplacian,weighted_norm
from sparse_exchange_audit import audit_laplacian


_hash_cache={}
def digest(path):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if key not in _hash_cache:_hash_cache[key]=streamed_digest(path)
    return _hash_cache[key]


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-shear-transient-spatial'
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
    assert cfg['protocol']==json.loads((source/'stage3-status.json').read_text())['next_contact_frozen_predeclared']
    data,inputs=contact_inputs(128,.02);m=data.mass;c=data.c0;labels=inputs['material_label_centers']
    with np.load(out/'contact-frozen.npz') as z:
        for key,value in inputs.items():np.testing.assert_array_equal(z[key],value)
        for key,value in [('time',data.time),('positions',data.positions),('rho',data.rho)]:np.testing.assert_array_equal(z[key],value)
        face=analytic_face_reference(128,.02,labels,m)
        for key,value in face.items():np.testing.assert_array_equal(z['face_'+key],value)
        volumes={order:volume_reference(128,.02,labels,m,order) for order in [8,16]}
        for order,values in volumes.items():
            for key,value in values.items():np.testing.assert_array_equal(z[f'volume{order}_'+key],value)
        regions=contact_regions(labels,.04)
        for key,value in regions.items():np.testing.assert_array_equal(z[key],value)
        np.testing.assert_array_equal(z['reference_rate'],face['rate'])
        L=load_laplacian(z);assert audit_laplacian(m,L)==r['graph_algebra']
        rate=-(L@c)/m;np.testing.assert_array_equal(rate,z['kernel_rate'])
        np.testing.assert_array_equal(z['rate_error'],rate-face['rate'])
        scatter=pair_action(m,z['pairs'],z['conductances'],c);np.testing.assert_array_equal(scatter,z['pair_scatter_rate'])
        norm=weighted_norm(m,face['rate']);relative=lambda v:weighted_norm(m,v)/norm
        assert regional_metrics(m,rate,face['rate'],regions,face['rate_a'],face['rate_b'])==r['regional_comparison']
        controls=r['reference_controls']
        assert relative(face['rate']-volumes[16]['rate'])==controls['analytic_faces_vs_volume16_relative']
        assert relative(volumes[8]['rate']-volumes[16]['rate'])==controls['volume8_vs_volume16_relative']
        assert float(np.max(abs(m-volumes[16]['mass'])/m))==controls['exact_mass_vs_Gauss16_max_relative']
        assert float(np.max(abs(c-volumes[16]['state'])))==controls['exact_c0_vs_Gauss16_max_absolute']
        assert float(m@face['rate'])==controls['physical_inventory_rate_faces_kg_per_s']
        assert float(m@volumes[16]['rate'])==controls['physical_inventory_rate_volume16_kg_per_s']
        physical_chi=float(volumes[16]['physical_dissipation'].sum())
        assert physical_chi==r['physical_continuous_chi_kg_per_s']
        assert float(volumes[16]['c_rate_integral'].sum()+physical_chi)==controls['physical_energy_identity_volume16_residual_kg_per_s']
        assert -float(m@(c*face['rate']))==r['projected_reference_chi_kg_per_s']
        raw=r['raw_graph_controls'];i,j=z['pairs'].T;g=z['conductances']
        assert float(c@(L@c))==raw['chi_CSR_kg_per_s']
        assert float(g@((c[j]-c[i])**2))==raw['chi_pair_kg_per_s']
        assert relative(scatter-rate)==raw['pair_scatter_vs_CSR_relative']
        assert float(np.max(abs(scatter-rate)))==raw['pair_scatter_vs_CSR_max_absolute_per_s']
        assert float(m@rate)==raw['raw_component_mass_rate_kg_per_s']
        np.testing.assert_array_equal(z['quadrature_area'],inputs['true_cell_area_reference_only'])
        assert np.all(g>=0) and raw['chi_CSR_kg_per_s']>=0
        assert abs(r['regional_comparison']['all']['error_over_global_reference_norm']**2-
            sum(r['regional_comparison'][key]['error_over_global_reference_norm']**2 for key in ['near_contact','far_from_contact']))<1e-14
        with np.load(out/'contact-observations.npz') as small:
            for key in small.files:np.testing.assert_array_equal(small[key],z[key])
    assert r['new_frozen_graphs']==1 and r['new_candidate_transient_runs']==0 and not r['fluid_orc_enabled'] and not r['full_P_constructed']
    checks={'predeclared_one_stationary_contact_frozen_graph_verified':True,'exact_material_mass_average_inputs_verified':True,
        'analytic_faces_and_full_piecewise_PDE_Gauss8_16_recomputed':True,
        'original_global_norm_and_predeclared_two_h_masks_verified':True,'all_region_errors_and_contributions_recomputed':True,
        'raw_shared_graph_action_and_positive_energy_verified':True,'m_over_rho_equals_true_area_exactly':True,
        'all_small_observations_equal_raw_NPZ':True,'new_candidate_transient_runs':0,'full_P_constructed':False,'fluid_orc_enabled':False}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'old_candidate_code_and_results_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md',
        'content_hash_cache':'streamed hash once per unchanged path/size/mtime tuple; all historical manifest entries compared'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
