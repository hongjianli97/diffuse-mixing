"""Read only sealed old inputs and define one delta case; never instantiate a flow."""
from pathlib import Path
from fractions import Fraction
import argparse, ast, json, resource, shutil, signal, time
import numpy as np
from audit_route_decision import ROOT, read, save, digest

OUT = ROOT/'runs/reconnect-becker-delta-definition'
PARENT = ROOT/'runs/reconnect-becker-common-k32'
PROTOCOL = ROOT/'research/reconnect/becker_delta_definition_protocol.json'


def preservation():
    previous = read(PARENT/'preservation.json')
    formal = previous['formal_manifests_verified'] + [{
        'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(PARENT/'sha256.json'),
        'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal) == 112
    for item in formal:
        p = ROOT/item['manifest']
        assert digest(p) == item['manifest_sha256']
        entries = read(p)
        assert len(entries) == item['entries_verified']
        for name,h in entries.items():
            assert digest(ROOT/name) == h,name
    originals = read('/workspace/original-handoff-sha256.json')
    attachments = read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals) == 782 and len(attachments) == 15
    for name,h in originals.items():
        assert digest(ROOT/name) == h,name
    for item in attachments:
        p = Path('/workspace')/item['path']
        assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256']
    historical = previous['historical_diagnostic_differences_retained']
    assert len(historical) == 4
    for item in historical:
        p = ROOT/item['manifest']
        assert digest(p) == item['manifest_sha256']
        entries = read(p)
        assert len(entries) == item['entries']
        differences = [{'path':k,'saved':v,'current':digest(ROOT/k)} for k,v in entries.items() if digest(ROOT/k) != v]
        assert sorted(differences,key=lambda x:x['path']) == sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda x:x['path'])
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':historical,'all_prior_scientific_files_unchanged':True}


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def inventory():
    old = ROOT/'runs/reconnect-becker-spatial'
    col = arrays(old/'n32/column.npz')
    parent = arrays(old/'n128/column.npz')['mass_edges']
    material = arrays(PARENT/'n32/material.npz')
    edges,m,c0 = col['mass_edges'],col['mass'],col['c0']
    assert len(m) == 32 and edges[0] == 0 and len(parent) == 129
    np.testing.assert_array_equal(np.diff(edges),m)
    np.testing.assert_array_equal(parent[::4],edges)
    np.testing.assert_array_equal(material['mass'],np.tile(m/8,8))
    np.testing.assert_array_equal(material['c0'],np.tile(c0,8))
    mesh = read(old/'reference/mesh-contract.json')
    assert mesh['base_cells'] == 16486 and mesh['fine_cells'] == 32972
    inputs = [arrays(old/f'reference/input-{name}.npz') for name in ('l0-k8192','l1-k8192','l1-k16384')]
    for k,z in enumerate(inputs):
        counts = np.array(mesh['counts'])*(1 if k == 0 else 2)
        starts = np.r_[0,np.cumsum(counts)]
        assert z['mass_edges'][0] == 0 and z['mass_edges'][-1] == edges[-1]
        assert len(z['mass']) == sum(counts) and np.isfinite(z['c0']).all()
        np.testing.assert_array_equal(z['mass_edges'][starts],parent)
        np.testing.assert_array_equal(np.diff(z['mass_edges']),z['mass'])
    np.testing.assert_array_equal(inputs[1]['mass_edges'][::2],inputs[0]['mass_edges'])
    for key in ('mass_edges','mass','c0'):
        np.testing.assert_array_equal(inputs[1][key],inputs[2][key])
    return {'original_columns':32,'rows':8,'nodes':256,'fixed_M_star':float(edges[-1]),
        'fixed_M_star_hex':float(edges[-1]).hex(),'fixed_M_star_exact':str(Fraction(float(edges[-1]))),
        'candidate_mass_extrema':[float(m.min()),float(m.max())],
        'candidate_c0_extrema':[float(c0.min()),float(c0.max())],
        'mass_edges_and_saved_mass_identical':True,'material_extrusion_bitwise_verified':True,
        'reference_parent_nested_with32_columns':True,'reference_base_cells':16486,'reference_fine_cells':32972,
        'reference_fine_inputs_at_two_times_bitwise_equal':True,'reference_initial_arrays_reusable':True,
        'new_delta_total_evaluated':False,'new_delta_positions_density_states_or_reference_evaluated':False}


def source_excerpts():
    found = []
    for path,names in {
        'research/rebuild/becker.py':['B','eta','u','xi','mu','z_mu','x_mu','rho_mu','initial'],
        'research/reconnect/t1_uniform_core.py':['h_indicator'],
        'research/reconnect/aligned_mass_reference.py':['conductances']}.items():
        source = (ROOT/path).read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.FunctionDef) and node.name in names:
                found.append({'path':path,'name':node.name,'line':node.lineno,'sha256':digest(ROOT/path),
                              'text':ast.get_source_segment(source,node)})
    for path,needle in [
        ('research/reconnect/aligned_becker_check.py','p[-1]==flow.total'),
        ('research/reconnect/becker_spatial_core.py','H[K]=old'),
        ('docs/CODEX_TASK.md','分别细化 h、Δt、物理层厚度δ'),
        ('docs/METHOD.md','每个空间实验至少三档 h')]:
        lines = (ROOT/path).read_text().splitlines()
        hits = [(i+1,s) for i,s in enumerate(lines) if needle in s]
        assert len(hits) == 1
        line,text = hits[0]
        found.append({'path':path,'name':needle,'line':line,'sha256':digest(ROOT/path),'text':text})
    return found


def main():
    started = time.monotonic()
    def alarm(*_):
        raise TimeoutError('registered120s read-only definition limit')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(120)
    assert not OUT.exists()
    OUT.mkdir()
    p = read(PROTOCOL)
    assert digest(PARENT/'next-gate-spec.json') == p['source_gate_sha256']
    assert digest(PARENT/'sha256.json') == p['parent_seal_sha256']
    gate = read(PARENT/'next-gate-spec.json')
    assert gate['type'] == 'read_only_minimal_independent_delta_control_definition' and not gate['already_executed']
    seal = read(PARENT/'sha256.json')
    for name,h in p['sources'].items():
        assert seal[name] == h and digest(ROOT/name) == h,name
        target = OUT/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save(OUT/'implementation-lock.json',{name:digest(ROOT/name) for name in p['implementation_files']})
    save(OUT/'source-excerpts.json',source_excerpts())
    save(OUT/'input-inventory.json',inventory())
    old = read(PARENT/'results.json')
    assert old['common_K32_spatial_diagnostic_resolved'] and old['by_grid']['32']['shifted_H_time_control']['resolved']
    assert not old['by_grid']['64']['shifted_H_time_control']['resolved'] and not old['by_grid']['128']['shifted_H_time_control']['resolved']
    save(OUT/'definition.json',{
        'problem_family':'fixed_saved_material_mass_coordinates_and_initial_means',
        'fixed':p['fixed'],'recompute':p['recompute'],'forbidden_reuse':p['forbidden_reuse'],
        'coordinate_identities':['xi_delta(z)=delta*q(z)','xi_z=B*u/j','X_mu=u/j=1/rho','X_t=u1-u',
            'integral_rho_dx_between_material_faces=mu_right-mu_left',
            'dc_dt_at_fixed_mu=d_mu(rho_delta^2*D*d_mu(c))',
            'xi(log9)-xi(-log9)=B*(u1+u2)*log9/j=delta'],
        'new_to_old_delta_ratio_exact':str(Fraction(p['proposed_delta'])/Fraction(p['old_delta'])),
        'M_star_is_old_saved_endpoint_not_new_flow_total':True,
        'both_initial_physical_endpoints_and_Eulerian_c0_fixed':False,
        'reference_mesh_and_c0_reused_reference_terminal_recomputed':True,
        'proposed_case':p['proposed_case'],'fixed_budget_contract':p['contract'],
        'contract_sha256':digest(ROOT/p['contract']),
        'old_common_K32_spatial_diagnostic_resolved':True,
        'old_shifted_H_outcomes':old['shifted_H_time_controls_resolved'],
        'old_case_budgets_preserved':old['by_grid']['32'],
        'new_candidate_h_control_completed':False,'delta_zero_limit_claimed':False,
        'new_delta_data_generated':False,'new_graphs':0,'new_candidate_paths':0,'new_reference_PDE_solves':0,
        'new_density_sources':0,'new_software_checks':0,'recorded_checks_total':502,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(OUT/'preservation.json',preservation())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    assert peak < 1024
    save(OUT/'execution.json',{'completed':True,'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':peak,
        'old_arrays_only':True,'flow_geometry_or_PDE_functions_called':False,'source_files':len(p['sources'])})
    signal.alarm(0)
    print('Fixed-material delta definition completed; no new delta flow/input/state/solve generated.')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
