"""Read-only T1 density-source contract and saved-input inventory; no solver imports."""
from pathlib import Path
from fractions import Fraction as F
import argparse,ast,math,resource,shutil,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest

PARENT=ROOT/'runs/reconnect-becker-delta004-budget'
T1=ROOT/'runs/reconnect-t1-uniform'
PROTOCOL=ROOT/'research/reconnect/t1_density_definition_protocol.json'


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def inventory():
    b=arrays(T1/'builder-inputs.npz');d=arrays(T1/'n32/material-reference.npz');g=read(T1/'n32/geometry.json')
    receipt=read(T1/'n32/geometry-only-audit.json');r=read(T1/'results.json');a=read(T1/'numerical-audit.json')
    assert r['T1_registered_case_passed'] and a['T1_registered_case_passed']
    assert digest(T1/'n32/geometry.json')==receipt['cache_sha256'] and digest(T1/'builder-inputs.npz')==receipt['builder_packet_sha256']
    x,ids,rho=b['n32_positions'],b['n32_ids'],b['n32_rho'];m,c0=d['mass'],d['c0']
    assert x.shape==(1024,2) and ids.shape==m.shape==rho.shape==c0.shape==(1024,)
    assert d['reference'].shape==d['symbol'].shape==(3,1024)
    assert np.isfinite(c0).all() and c0.min()>=0 and c0.max()<=1 and all(F(float(v))==F(1,1024) for v in m)
    assert all(v==1 for v in rho) and all(int(v)==i for i,v in enumerate(ids)) and g['ids']==ids.tolist()
    for i,(xx,yy) in enumerate(x):
        assert F(float(xx))==F(2*(i%32)+1,64) and F(float(yy))==F(2*(i//32)+1,64)
    assert g['periods_exact']==['1','1'] and g['thickness_exact']=='1' and F(g['D_exact'])==F(.02)
    assert len(g['edges'])==2048 and len(g['cell_areas_exact'])==1024
    counts=[0]*1024;directions=set()
    for (i,j),(sx,sy),ratio,conductance in zip(g['edges'],g['image_shifts'],g['face_over_distance_exact'],g['conductances_exact']):
        assert i<j and F(ratio)==1 and F(conductance)==F(.02)
        dx=F(float(x[j,0]))+sx-F(float(x[i,0]));dy=F(float(x[j,1]))+sy-F(float(x[i,1]))
        assert (abs(dx)==F(1,32) and dy==0) or (dx==0 and abs(dy)==F(1,32))
        directions.add((str(dx),str(dy)));counts[i]+=1;counts[j]+=1
    assert all(v==4 for v in counts) and all(F(v)==F(1,1024) for v in g['cell_areas_exact'])
    np.testing.assert_array_equal(d['reference'][0],c0)
    paths=[]
    for K in (4,8,16):
        z=arrays(T1/f'n32/k{K}/trajectory-states.npz');run=read(T1/f'n32/k{K}/native-results.json')
        assert z['states'].shape==z['independent_states'].shape==(K+1,1024) and len(run['steps'])==K
        np.testing.assert_array_equal(z['states'][0],c0);np.testing.assert_array_equal(z['independent_states'][0],c0)
        assert all(F(s['dt_exact'])==F(1,20*K) for s in run['steps'])
        paths.append({'K':K,'saved_steps':K,'states_sha256':digest(T1/f'n32/k{K}/trajectory-states.npz'),
            'run_sha256':digest(T1/f'n32/k{K}/native-results.json'),'path_rerun':False})
    return {'N_per_axis':32,'nodes':1024,'saved_shared_faces':2048,'all_saved_nodes_have_four_cardinal_neighbors':True,
        'saved_uniform_material_mass_exact':'1/1024','given_initial_oracle_density':1,'saved_reference_initial_bitwise_equal_c0':True,
        'saved_original_common_sigma16':r['common_initial_fluctuation_norm'],'oracle_paths':paths,
        'given_positions_or_density_recomputed':False,'new_density_or_area_or_coefficient_arrays_generated':False}


def algebra():
    # Formal polynomial coefficients, not evaluations of a new density field.
    a=[F(math.comb(6,k))*(-F(1,2))**k for k in range(7)];b=[F(1),F(3),F(35,12)]
    c=[F(0)]*9
    for i,x in enumerate(a):
        for j,y in enumerate(b):c[i+j]+=x*y
    integral=sum((v*F(2)**(k+2)/F(k+2) for k,v in enumerate(c)),F(0))
    assert integral==F(2,9) and F(9,2)*integral==1
    support_count=sum(1 for a in range(-3,4) for b in range(-3,4) if a*a+b*b<16)
    assert support_count==45 and F(2,32)*2<F(1,2)
    dx=F(1,32);Astar=F(1,2)*4*dx*dx;mass=dx*dx
    assert Astar/mass==2
    return {'kernel_polynomial_coefficients':[str(x) for x in c],'radial_integral_exact':str(integral),
        'angular_normalization_factor_without_pi_exact':'9/2','continuous_kernel_unit_integral_verified':True,
        'h_exact':'1/16','support_radius_exact':'1/8','periodic_half_length_exact':'1/2','lattice_support_terms_including_self':support_count,
        'lattice_density_finite_sum_numerically_evaluated':False,'new_numerical_density_value_produced':False,
        'static_cardinal_fan_area_symbolic_exact':str(Astar),'proxy_anchor_mass_over_material_mass_symbolic_exact':'2',
        'ledger_static_density_identity_is_symbolic_only':True,'future_ledger_coefficients_must_be_checked_not_assigned':True}


def excerpts():
    selected={
        'reference_baseline/lhd/lhdx.py':['_star','chained_lnJ_v2'],
        'reference_baseline/lhd/ledger_ub.py':['_affine_F','ledger_v3'],
        'reference_baseline/lhd/ledger_v4.py':['fan_area','_arc_resid'],
        'research/reconnect/positive_kernel_2d.py':['kernel_value'],
        'research/reconnect/t1_uniform_core.py':['h_indicator'],
    }
    found=[]
    for name,names in selected.items():
        source=(ROOT/name).read_text();tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                found.append({'path':name,'name':node.name,'line':node.lineno,'sha256':digest(ROOT/name),
                    'text':ast.get_source_segment(source,node)})
    assert len(found)==8
    return found


def preserve():
    old=read(PARENT/'preservation.json');formal=old['formal_manifests_verified']+[{
        'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(PARENT/'sha256.json'),
        'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal)==115
    for item in formal:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for name,h in entries.items():assert digest(ROOT/name)==h,name
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for name,h in originals.items():assert digest(ROOT/name)==h,name
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
    for item in old['historical_diagnostic_differences_retained']:
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries']
        differences=[{'path':k,'saved':v,'current':digest(ROOT/k)} for k,v in entries.items() if digest(ROOT/k)!=v]
        assert sorted(differences,key=lambda x:x['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda x:x['path'])
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],'all_prior_scientific_files_unchanged':True}


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def timeout(*_):raise TimeoutError('registered120s read-only definition cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='read_only_smooth_T1_same_backend_density_control_definition' and not gate['already_executed']
        assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},'read_only_definition':True})
        copies=[]
        for name in p['saved_data_files']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
            copies.append({'source':name,'copy':str(target.relative_to(out)),'sha256':digest(target)})
        save(out/'source-copies.json',copies);save(out/'input-inventory.json',inventory());save(out/'symbolic-checks.json',algebra())
        save(out/'source-excerpts.json',excerpts())
        save(out/'definition.json',{'date':'2026-11-07','timezone':'Etc/UTC','decision':'single_T1_density_calibration_contract_defined_not_executed',
            'contract':p['contract'],'contract_sha256':digest(ROOT/p['contract']),'proposed_case':p['proposed_case'],
            'kernel_definition':'unchanged 2D radial kernel,continuous normalization only; fixed h=1/16; self term and original masses',
            'ledger_definition':'new minimal fixed oriented fan-area ratio,not an execution of legacy ledger_v3/v4',
            'ledger_proxy_anchor_mass_is_not_original_mass':True,'F_correction_enabled':False,'legacy_drivers_imported_or_called':False,
            'static_time_rule_inherited_from_T1_not_Becker_H_waiver':True,'density_bias_physical_tolerance_introduced':False,
            'ledger_path_reuse_requires_exact_coefficient_identity':True,'new_density_arrays_or_coefficients_generated':False,
            'new_actual_or_independent_paths':0,'new_geometry_graphs':0,'new_reference_PDE_solves':0,'new_software_checks':0,
            'recorded_checks_total':502,'future_software_checks_if_both_pass':504,'general_density_validation_completed':False,
            'all_Becker_unresolved_gates_preserved':True,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
        save(out/'preservation.json',preserve());elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and peak<1024
        save(out/'execution.json',{'completed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,
            'flow_geometry_density_or_PDE_functions_called':False,'old_paths_rerun':False,'published_or_pushed':False})
        print('T1 density-source contract defined; no new density,coefficient,graph,state or PDE.',flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
