"""Read-only inherited-evidence audit; write only this new run's preservation log."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from noflux_probe_study import weighted_error


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(source,output):
    root=Path(__file__).resolve().parents[2];source=Path(source).resolve();output=Path(output).resolve()
    previous=json.loads((source/'preservation.json').read_text())
    originals=json.loads(Path('/workspace/original-handoff-sha256.json').read_text())
    for rel,h in originals.items():
        assert digest(root/rel)==h,rel
    attachments=json.loads(Path('/workspace/setup-ltn/attachment-manifest.json').read_text())
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],str(p)
    manifests=previous['formal_scientific_manifests_verified']
    manifests=manifests+[{'manifest':str((source/'sha256.json').relative_to(root)),
        'manifest_sha256':digest(source/'sha256.json'),'entries_verified':len(json.loads((source/'sha256.json').read_text()))}]
    for item in manifests:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'],str(p)
        entries=json.loads(p.read_text());assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(root/rel)==h,rel
    historical=previous['historical_diagnostic_manifests_not_claimed_as_current_hash_matches']
    for item in historical:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'],str(p)
        entries=json.loads(p.read_text());assert len(entries)==item['entries']
        mismatches=[{'path':rel,'saved':h,'current':digest(root/rel)} for rel,h in entries.items() if digest(root/rel)!=h]
        assert sorted(mismatches,key=lambda v:v['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda v:v['path'])
    names={k:k for k in ['mass','original_c0','quads','mass_centroids','columns','rows','boundary_cell_mask']}
    names.update({'probe_average':'probe_average_order64','physical_reference_rate':'face_rate_order64',
        'physical_reference_flux':'face_flux_order64','saved_graph_rate':'graph_rate'})
    names.update({'topology_'+k:'topology_'+k for k in ['positions','cell_vertices','edge_vertices','incidence','boundary_mask']})
    r=json.loads((output/'results.json').read_text())
    with np.load(source/'noflux-probe.npz') as a,np.load(output/'hybrid-face-baseline.npz') as z:
        for current,old in names.items():np.testing.assert_array_equal(z[current],a[old],err_msg=current)
        m=z['mass'];c=z['probe_average'];Q=z['operator_Q'];L=z['operator_L'];F=z['operator_face_flux_from_cells'];C=z['topology_incidence']
        np.testing.assert_allclose(z['hybrid_rate'],Q@c,atol=0.,rtol=0.)
        np.testing.assert_allclose(z['hybrid_face_flux'],F@c,atol=0.,rtol=0.)
        np.testing.assert_allclose(Q,-L/m[:,None],atol=0.,rtol=0.)
        np.testing.assert_allclose(z['basis_rate'],Q@z['basis_initial'],atol=0.,rtol=0.)
        np.testing.assert_allclose(z['basis_rate'],-C@(F@z['basis_initial'])/m,atol=2e-12,rtol=2e-12)
        for key,mask in [('all_cells',np.ones(len(m),bool)),('interior_cells',~z['boundary_cell_mask']),('boundary_cells',z['boundary_cell_mask'])]:
            fresh=weighted_error(m,z['hybrid_rate'],z['physical_reference_rate'],mask)
            for metric,value in fresh.items():np.testing.assert_allclose(value,r['hybrid_'+key][metric],atol=1e-15,rtol=1e-14)
        inside=~z['topology_boundary_mask'];area=z['operator_face_area'];delta=z['hybrid_face_flux']-z['physical_reference_flux']
        error=np.sqrt(np.sum(delta[inside]**2/area[inside])/np.sum(z['physical_reference_flux'][inside]**2/area[inside]))
        np.testing.assert_allclose(error,r['internal_face_flux_density_relative_L2_error'],atol=1e-15)
        off=Q.copy();np.fill_diagonal(off,np.inf);i,j=np.unravel_index(off.argmin(),Q.shape)
        assert i==r['basis_initial_derivative_counterexample']['observed_cell'] and j==r['basis_initial_derivative_counterexample']['initial_cell']
        assert Q[i,j]<0 and z['basis_initial'][i]==0 and z['basis_initial'][j]==1 and np.count_nonzero(z['basis_initial'])==1
        np.testing.assert_allclose(Q[i,j],r['minimum_raw_offdiagonal_rate_per_s'],atol=0.,rtol=0.)
    audit={'checked_original_files':len(originals),'checked_attachments':len(attachments),
        'formal_scientific_manifests_verified':manifests,'all_formal_scientific_entries_unchanged':True,
        'inherited_arrays_equal_to_source':names,'all_inherited_arrays_equal_to_source':True,
        'rate_native_face_reference_error_and_basis_failure_recomputed_from_saved_arrays':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,
        'historical_note':previous['historical_note'],'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (output/'preservation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'originals_verified':len(originals),'attachments_verified':len(attachments),'formal_manifests_verified':len(manifests),
        'historical_manifests_with_unchanged_preexisting_differences':len(historical),'inherited_arrays_verified':len(names),'saved_error_and_negative_basis_derivative_verified':True},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
