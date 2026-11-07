"""Independent saved-input and symbolic-contract audit; never evaluate new density."""
from pathlib import Path
from fractions import Fraction as F
import argparse,ast,json,math,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main(directory,write=False):
    start=time.monotonic();out=Path(directory);p=read(out/'protocol.json');definition=read(out/'definition.json')
    inv=read(out/'input-inventory.json');symbol=read(out/'symbolic-checks.json');ex=read(out/'execution.json')
    assert ex['completed'] and ex['elapsed_seconds']<120 and ex['peak_RSS_MiB']<1024
    assert digest(out/'source-gate-spec.json')==p['source_gate_sha256']
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']==p['sources'][item['source']]
    assert digest(ROOT/definition['contract'])==definition['contract_sha256']==p['sources'][definition['contract']]
    src=out/'source/runs/reconnect-t1-uniform';b=arrays(src/'builder-inputs.npz');d=arrays(src/'n32/material-reference.npz');g=read(src/'n32/geometry.json')
    assert read(src/'results.json')['T1_registered_case_passed'] and read(src/'numerical-audit.json')['T1_registered_case_passed']
    assert inv['N_per_axis']==32 and inv['nodes']==1024 and inv['saved_shared_faces']==2048
    for i in range(1024):
        assert int(b['n32_ids'][i])==i and b['n32_rho'][i]==1 and F(float(d['mass'][i]))==F(1,1024)
        assert F(float(b['n32_positions'][i,0]))==F(2*(i%32)+1,64)
        assert F(float(b['n32_positions'][i,1]))==F(2*(i//32)+1,64)
    np.testing.assert_array_equal(d['reference'][0],d['c0'])
    neighbors=[set() for _ in range(1024)]
    for (i,j),shift,ratio,conductance in zip(g['edges'],g['image_shifts'],g['face_over_distance_exact'],g['conductances_exact']):
        assert F(ratio)==1 and F(conductance)==F(g['D_exact'])==F(.02)
        neighbors[i].add(j);neighbors[j].add(i)
        delta=[F(float(b['n32_positions'][j,k]))+shift[k]-F(float(b['n32_positions'][i,k])) for k in (0,1)]
        assert sum(x*x for x in delta)==F(1,1024) and delta[0]*delta[1]==0
    for i,adj in enumerate(neighbors):
        row,col=divmod(i,32)
        assert adj=={row*32+(col-1)%32,row*32+(col+1)%32,((row-1)%32)*32+col,((row+1)%32)*32+col}
    for path in inv['oracle_paths']:
        K=path['K'];statepath=src/f'n32/k{K}/trajectory-states.npz';runpath=src/f'n32/k{K}/native-results.json'
        assert digest(statepath)==path['states_sha256'] and digest(runpath)==path['run_sha256'] and not path['path_rerun']
        z=arrays(statepath);assert z['states'].shape==z['independent_states'].shape==(K+1,1024)
        np.testing.assert_array_equal(z['states'][0],d['c0']);np.testing.assert_array_equal(z['independent_states'][0],d['c0'])
        assert all(F(s['dt_exact'])==F(1,20*K) for s in read(runpath)['steps'])
    # Independent beta-integral evaluation, not polynomial integration from producer.
    integral=sum((coefficient*2**(k+2)*F(math.factorial(k+1)*math.factorial(6),math.factorial(k+8))
        for k,coefficient in enumerate((F(1),F(3),F(35,12)))),F(0))
    assert integral==F(symbol['radial_integral_exact'])==F(2,9) and F(9,2)*integral==1
    coeff=list(map(F,symbol['kernel_polynomial_coefficients']))
    for x in (F(0),F(1,3),F(1),F(3,2),F(2)):
        assert sum((v*x**k for k,v in enumerate(coeff)),F(0))==(1-x/2)**6*(1+3*x+F(35,12)*x*x)
    support_count=1+4*3+4*sum(1 for a in (1,2,3) for b in (1,2,3) if a*a+b*b<16)
    assert support_count==symbol['lattice_support_terms_including_self']==45
    assert F(symbol['h_exact'])==F(1,16) and 2*F(symbol['h_exact'])==F(symbol['support_radius_exact'])<F(1,2)
    assert F(symbol['static_cardinal_fan_area_symbolic_exact'])==F(1,2)*F(2,32)*F(2,32)==F(1,512)
    assert F(symbol['proxy_anchor_mass_over_material_mass_symbolic_exact'])==F(1,512)/F(1,1024)==2
    assert not symbol['lattice_density_finite_sum_numerically_evaluated'] and not symbol['new_numerical_density_value_produced']
    excerpts=read(out/'source-excerpts.json');assert len(excerpts)==8
    for e in excerpts:
        source=(ROOT/e['path']).read_text();assert digest(ROOT/e['path'])==e['sha256']
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==e['name'])
        assert node.lineno==e['line'] and ast.get_source_segment(source,node)==e['text']
    get=lambda name:next(e['text'] for e in excerpts if e['name']==name)
    assert 'np.abs' in get('_star') and 'np.where(np.isfinite' in get('chained_lnJ_v2')
    assert 'np.linalg.solve' in get('_affine_F') and 'tensor=True' in get('ledger_v3') and 'lstsq' in get('_arc_resid')
    assert 'C2/h**2' in get('kernel_value')
    case=definition['proposed_case'];assert case==p['proposed_case']
    assert case['N']==32 and case['K']==[4,8,16] and case['new_density_fields']==case['new_coefficient_packets']==2
    assert case['max_new_actual_paths']==case['max_new_independent_paths']==3 and case['new_action_intervals']==28
    assert case['new_geometry_graphs']==case['new_reference_PDE_solves']==case['old_paths_rerun']==0
    assert case['ledger_paths_reused_after_exact_coefficient_identity']==3 and case['new_software_checks_before_science']==2
    assert definition['static_time_rule_inherited_from_T1_not_Becker_H_waiver'] and not definition['density_bias_physical_tolerance_introduced']
    assert definition['recorded_checks_total']==502 and definition['future_software_checks_if_both_pass']==504
    assert not definition['new_density_arrays_or_coefficients_generated'] and definition['new_actual_or_independent_paths']==0
    for filename in ('t1_density_definition.py','audit_t1_density_definition.py'):
        tree=ast.parse((ROOT/'research/reconnect'/filename).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert set(imports)<={'pathlib','fractions','audit_route_decision'}
        assert not any(isinstance(n,ast.Import) and any(a.name.startswith(('scipy','ledger','positive_kernel','periodic_voronoi','voronoi_time','t1_uniform')) for a in n.names) for n in ast.walk(tree))
    assert not definition['F_correction_enabled'] and not definition['legacy_drivers_imported_or_called']
    assert definition['stage3_physical_acceptance']=='incomplete' and not definition['fluid_orc_enabled'] and definition['all_Becker_unresolved_gates_preserved']
    findings={'passed':True,'saved_T1_inputs_geometry_and_path_headers_verified':True,'given_mass_c0_reference_unchanged':True,
        'kernel_unit_integral_independent_beta_proof':'1','finite_lattice_density_not_evaluated':True,
        'ledger_proxy_anchor_mass_factor_two_explicit':True,'legacy_F_fit_abs_area_and_fallback_excerpts_verified':True,
        'static_time_rule_distinguished_from_Becker_decline':True,'proposed_counts_and_ledger_reuse_guard_verified':True,
        'new_density_arrays_or_coefficients_or_paths':0,'new_software_checks':0,'recorded_checks_total':502,
        'scope':'saved-input and contract feasibility,not execution or general density/physical validation'}
    if write:save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-start,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__))})
    else:assert read(out/'audit.json')['findings']==findings
    print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
