"""Inherited evidence preservation and saved-vector audit for one 2D control."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import roots_legendre
from positive_kernel_2d import kernel_value


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-mainline'
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
    cfg=json.loads((out/'config.json').read_text());result=json.loads((out/'results.json').read_text());h=cfg['h_m']
    with np.load(out/'periodic-kernel2d.npz') as z:
        m=z['mass'];c=z['c0'];L=csr_matrix((z['L_data'],z['L_indices'],z['L_indptr']),shape=tuple(z['L_shape']))
        np.testing.assert_allclose(-(L@c)/m,z['kernel_rate'],atol=0.,rtol=0.)
        np.testing.assert_allclose(z['reference_rate'],-8*np.pi**2*cfg['D_m2_per_s']*(c-.5),atol=2e-15,rtol=2e-15)
        np.testing.assert_array_equal(z['finite_width_error'],z['continuum_finite_width_rate']-z['reference_rate'])
        np.testing.assert_array_equal(z['node_quadrature_error'],z['kernel_rate']-z['continuum_finite_width_rate'])
        norm=np.sqrt(m@(z['reference_rate']**2))
        for key,vector in [('relative_mass_weighted_instantaneous_rate_error',z['kernel_rate']-z['reference_rate']),
            ('finite_width_rate_bias_relative_to_local_PDE',z['finite_width_error']),('node_quadrature_rate_difference_relative_to_local_PDE',z['node_quadrature_error'])]:
            np.testing.assert_allclose(np.sqrt(m@(vector*vector))/norm,result[key],atol=1e-15)
        np.testing.assert_array_equal(z['quadrature_area'],m/z['rho'][1])
        assert np.all(z['conductances']>=0) and cfg['new_fluid_transient_runs']==0 and not cfg['fluid_orc_enabled']
        factor=np.sinc(1/cfg['N_per_direction'])**2;a=z['material_label_centers']
        np.testing.assert_allclose(c,.5+.2*factor*np.cos(2*np.pi*a[:,0])*np.cos(2*np.pi*a[:,1]),atol=0.,rtol=0.)
    nodes,weights=roots_legendre(64);r=h*(nodes+1);weights=h*weights
    normalization=float(2*np.pi*np.sum(weights*kernel_value(r,h)*r));moment=float(2*np.pi*np.sum(weights*kernel_value(r,h)*r**3)/h**2)
    np.testing.assert_allclose(normalization,1.,atol=2e-14);np.testing.assert_allclose(moment,14/33,atol=2e-14)
    checks={'radial_W_normalization_order64':normalization,'mu2_over_h2_order64':moment,'exact_mu2_over_h2':'14/33',
        'saved_sparse_action_mass_average_state_and_error_vectors_verified':True,'saved_error_metrics_recomputed':True,
        'new_fluid_transient_runs':0,'new_propagators_constructed':0}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'prior_user_material_snapshots_covered_by_source_scientific_manifest':True,
        'new_probe_is_explicitly_separate_from_old_becker_and_frozen_nonorthogonal_cases':True,
        'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
