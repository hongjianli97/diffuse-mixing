"""One frozen static T1 N32 density comparison, three new kernel paths only."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,resource,shutil,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from t1_density_sources import kernel_density,fixed_rings,ledger_density,reweight
from audit_t1_density_calibration import audit_packets
from t1_density_calibration_core import symbols,compare
from shear_union64_study import native_run,arrays

PARENT=ROOT/'runs/reconnect-t1-density-definition'
PROTOCOL=ROOT/'research/reconnect/t1_density_calibration_protocol.json'


def main(directory):
    out=Path(directory).resolve();p=read(PROTOCOL);lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/name)==h,name
    assert read(out/'tests.json')['successful'] and read(out/'tests.json')['checks_run']==2
    assert read(out/'tests.json')['elapsed_seconds']<120
    assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
    for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
    # Copy bytes, without deserializing any scalar, reference or old path.
    copies=[]
    for name in p['sources']:
        if not name.startswith('runs/reconnect-t1-uniform/'):continue
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        copies.append({'source':name,'copy':str(target.relative_to(out)),'sha256':digest(target)})
    save(out/'source-copies.json',copies)
    started=time.monotonic()
    def guard(stage,*args):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>300 or peak>1024:raise RuntimeError(f'frozen resource cap exceeded:{elapsed}s/{peak}MiB')
        print(f'{stage}: {elapsed:.3f}s,peak {peak:.2f}MiB',flush=True)
    def timeout(*_):raise TimeoutError('frozen300second science cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    try:
        src=out/'source/runs/reconnect-t1-uniform'
        with np.load(src/'builder-inputs.npz',allow_pickle=False) as z:
            ids=z['n32_ids'];x=z['n32_positions'];rho0=z['n32_rho']
        with np.load(src/'n32/material-reference.npz',allow_pickle=False) as z:m=z['mass']
        original=read(src/'n32/geometry.json');assert original['D_exact']==str(F(.02)) and original['ids']==ids.tolist()
        for a in (ids,x,rho0,m):a.setflags(write=False)
        for source in ('kernel','ledger'):(out/source).mkdir()
        rhoK=kernel_density(ids,x,ids,m,1/16,[1.,1.])
        rings=fixed_rings(ids,x,original);current=x.copy()
        if not np.array_equal(current,x):raise ValueError('this scientific driver only admits the saved static trajectory')
        rhoL,ring_record=ledger_density(ids,x,current,rho0,rings,[1.,1.])
        save(out/'ledger/rings.json',ring_record)
        for source,rho in (('kernel',rhoK),('ledger',rhoL)):
            np.savez_compressed(out/source/'density.npz',rho=rho)
            save(out/source/'coefficients.json',reweight(original,ids,rho,m))
        certificate=audit_packets(out,ids,x,m,rho0,original)
        save(out/'density-coefficient-audit.json',certificate)
        files={str(q.relative_to(out)):digest(q) for source in ('kernel','ledger') for q in (out/source).iterdir() if q.is_file()}
        files['density-coefficient-audit.json']=digest(out/'density-coefficient-audit.json')
        save(out/'coefficient-seal.json',{'files':files,'all_coefficients_sealed_before_scalar_loading':True,
            'new_density_fields':2,'new_coefficient_packets':2,'new_geometry_graphs':0,'kernel_received_oracle_density':False,
            'ledger_given_initial_density_only':True,'material_mass_npz_read_keys_before_seal':['mass'],
            'original_geometry_sha256':digest(src/'n32/geometry.json'),'mass_c0_file_sha256':digest(src/'n32/material-reference.npz')})
        guard('two density/coefficient packets independently checked and sealed')
        # First deserialization of scalar, reference and old propagation data.
        d=arrays(src/'n32/material-reference.npz');np.testing.assert_array_equal(m,d['mass'])
        parameters=read(src/'n32/analytic-parameters.json');sigma=read(src/'results.json')['common_initial_fluctuation_norm']
        runs={};paths={};old_paths={}
        for K in (4,8,16):
            folder=src/'n32'/f'k{K}';run=read(folder/'native-results.json');path=arrays(folder/'trajectory-states.npz')
            for source in ('oracle','ledger'):runs[source,K]=run;paths[source,K]=path
            old_paths[str(K)]={'file':str((folder/'trajectory-states.npz').relative_to(out)),
                'sha256':digest(folder/'trajectory-states.npz'),'run_sha256':digest(folder/'native-results.json'),
                'reason':'same original m,c0,exact shared g,static generator and dt sequence','rerun':False}
        save(out/'ledger-reuse.json',{'all2048_exact_coefficients_equal':certificate['ledger']['exact_coefficients_equal_oracle'],
            'old_paths':old_paths,'old_oracle_runs':0,'new_ledger_runs':0,'proxy_anchor_mass_over_original_m_exact':'2',
            'proxy_mass_replaces_material_mass':False,'motion_density_validated':False})
        packets={'oracle':original,**{s:read(out/s/'coefficients.json') for s in ('kernel','ledger')}}
        density={'oracle':rho0,'kernel':rhoK,'ledger':rhoL}
        symbol_records,symbol_arrays=symbols(m,d['c0'],original['D_exact'],packets,density,parameters['reference_evaluation_indicator_absolute'])
        save(out/'symbol-records.json',symbol_records);np.savez_compressed(out/'symbol-observations.npz',**symbol_arrays)
        index={}
        for K in (4,8,16):
            (out/'kernel'/f'k{K}').mkdir()
            for step in range(K):index[f'k{K}-step{step:02d}']={'file':'coefficients.json','sha256':digest(out/'kernel/coefficients.json'),
                'midpoint_time_saved':float((2*step+1)*.05/(2*K))}
        save(out/'kernel/graph-index.json',index)
        for K in (4,8,16):
            run,path=native_run(out/'kernel',K,d,index,guard)
            save(out/'kernel'/f'k{K}'/'native-results.json',run)
            np.savez_compressed(out/'kernel'/f'k{K}'/'trajectory-states.npz',**path)
            runs['kernel',K]=run;paths['kernel',K]=path
        result=compare(m,d['c0'],d['reference'],parameters['reference_evaluation_indicator_absolute'],sigma,runs,paths,symbol_records,symbol_arrays)
        save(out/'results.json',result)
        if not result['implementation_controls_passed']:raise ArithmeticError('frozen implementation controls failed or unresolved; saved and stopped')
        guard('frozen source comparison completed')
        save(out/'execution.json',{'completed':True,'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'date':'2026-11-07','timezone':'Asia/Shanghai','new_actual_paths':3,'new_independent_paths':3,'new_action_intervals':28,
            'new_density_fields':2,'new_coefficient_packets':2,'new_geometry_graphs':0,'new_reference_PDE_solves':0,
            'old_oracle_or_ledger_paths_rerun':0,'new_software_groups':2,'recorded_checks_total':504,'seed':20261101,
            'coefficient_seal_preceded_scalar_loading':True,'published_or_pushed':False})
        save(out/'stage3-status.json',{'date':'2026-11-07','timezone':'Asia/Shanghai','single_T1_density_calibration_executed':True,
            'implementation_controls_passed':True,'general_density_physical_acceptance':False,'all_Becker_unresolved_gates_preserved':True,
            'recorded_checks_total':504,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
        print(json.dumps({'implementation_controls_passed':True,'density':symbol_records,'comparisons':result['source_comparisons']},indent=2),flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-started,'partial_data_preserved':True,'repair_or_automatic_extra_run':False});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
