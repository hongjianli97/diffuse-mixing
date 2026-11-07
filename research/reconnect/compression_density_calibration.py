"""One frozen moving N64 comparison; all96 rates sealed before scalar loading."""
from pathlib import Path
from fractions import Fraction as F
import argparse,resource,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from compression_density_sources import periodic_winding,anchor_rings,kernel_density,ledger_density
from compression_density_audit_core import bin_density,verify_lifts,verify_ring_density,verify_coefficients
from compression_density_compare import compare
from t1_density_sources import reweight
from shear_union64_study import native_run,arrays

PARENT=ROOT/'runs/reconnect-compression-density-definition'
SOURCE=ROOT/'runs/reconnect-compression-union64'


def allowed_inputs():
    with np.load(SOURCE/'trajectory-source.npz',allow_pickle=False) as z:ids=z['ids'];labels=z['labels'];rho0=z['rho0_point']
    with np.load(SOURCE/'union-definition.npz',allow_pickle=False) as z:mass=z['mass']
    with np.load(SOURCE/'builder-inputs.npz',allow_pickle=False) as z:
        assert np.array_equal(ids,z['ids'])
        b={f'k{K}_{name}':z[f'k{K}_{name}'] for K in (16,32) for name in ('time','positions')}
    assert len(ids)==4096 and np.array_equal(ids,np.arange(4096)) and np.all(mass>0) and np.isfinite(mass).all()
    for name in ('time','positions'):np.testing.assert_array_equal(b['k16_'+name],b['k32_'+name][::2])
    for K in (16,32):
        np.testing.assert_array_equal(b[f'k{K}_positions'][0],labels)
        assert b[f'k{K}_time'].shape==(2*K+1,) and b[f'k{K}_time'][0]==0 and b[f'k{K}_time'][-1]==.05
        # Saved samples use exactly the old evenly spaced binary64 grid.
        np.testing.assert_array_equal(b[f'k{K}_time'],np.linspace(0,.05,2*K+1))
    for a in (ids,labels,rho0,mass,*b.values()):a.setflags(write=False)
    return ids,labels,rho0,mass,b


def density_diagnostics(out,ids,labels,rho0,m,b,index):
    # Only after every coefficient has been audited and sealed.
    with np.load(SOURCE/'builder-inputs.npz',allow_pickle=False) as z:oracle={K:z[f'k{K}_rho'] for K in (16,32)}
    anchor=read(out/'anchor-rings.json');ratio=np.array([float(F(float(r))*F(a)/F(float(w))) for r,a,w in zip(rho0,anchor['initial_areas_exact'],m)])
    rows=[];s2=np.sin(2*np.pi/64)/(2*np.pi/64)
    for K in (16,32):
        for step in range(K):
            frame=2*step+1;t=float(b[f'k{K}_time'][frame]);ref=oracle[K][frame];alpha=.25*t/.05
            for source in ('kernel','ledger'):
                rho=arrays(out/source/f'k{K}/density-step{step:02d}.npz')['rho'];diff=rho-ref
                row={'source':source,'K':K,'step':step,'time':t,'raw_density_minimum':float(rho.min()),'raw_density_maximum':float(rho.max()),
                     'point_error_Linf':float(np.max(abs(diff))),'point_error_mass_norm':float(np.sqrt(m@(diff*diff))),
                     'mass_weighted_density_bias':float(m@diff/m.sum()),'spatial_density_range':float(np.ptp(rho))}
                if source=='ledger':
                    secant=rho0/(1+alpha*s2*np.cos(2*np.pi*labels[:,0]));deviation=float(np.max(abs(rho-secant)))
                    assert deviation<1e-10
                    row['postseal_secant_formula_max_difference']=deviation
                rows.append(row)
    return {'rows':rows,'oracle_density_first_loaded_after_all96_coefficient_seal':True,
        'proxy_anchor_mass_over_original_mass_minimum':float(ratio.min()),'proxy_anchor_mass_over_original_mass_maximum':float(ratio.max()),
        'proxy_anchor_mass_replaced_original_mass':False,'density_acceptance_threshold_introduced':False}


def main(directory):
    out=Path(directory).resolve();p=read(out/'protocol.json');lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/name)==h,name
    test=read(out/'tests.json');assert test['successful'] and test['checks_run']==2 and test['elapsed_seconds']<120
    assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
    for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
    started=time.monotonic()
    def guard(stage,*args):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>=700 or peak>=1024:raise RuntimeError(f'frozen resource cap:{elapsed}s/{peak}MiB')
        if not args or args[0]%4==0:print(f'{stage} {args}: {elapsed:.2f}s,peak{peak:.1f}MiB',flush=True)
    def timeout(*_):raise TimeoutError('frozen700second science cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(700)
    try:
        ids,labels,rho0,m,b=allowed_inputs();index=read(SOURCE/'graph-index.json');wind={};lift_audits={}
        for K in (16,32):
            wind[K]=periodic_winding(ids,b[f'k{K}_time'],b[f'k{K}_positions'],np.linspace(0,.05,2*K+1))
            lift_audits[str(K)]=verify_lifts(labels,b[f'k{K}_time'],b[f'k{K}_positions'],wind[K])
        np.testing.assert_array_equal(wind[16],wind[32][::2]);np.savez_compressed(out/'integer-winding.npz',k16=wind[16],k32=wind[32])
        save(out/'lift-audit.json',lift_audits)
        first=read(SOURCE/index['k16-step00']['file']);rings,a0=anchor_rings(ids,labels,first['edges']);del first
        expected=sorted({tuple(sorted((i,j))) for i in range(4096) for j in ((i//64)*64+(i%64+1)%64,((i//64+1)%64)*64+i%64)})
        for i,ring in enumerate(rings):
            expected_neighbors={(i//64)*64+(i%64+1)%64,(i//64)*64+(i%64-1)%64,((i//64+1)%64)*64+i%64,((i//64-1)%64)*64+i%64}
            assert {v[0] for v in ring}==expected_neighbors and a0[i]==F(1,2048)
        save(out/'anchor-rings.json',{'rings':rings,'initial_areas_exact':list(map(str,a0)),'images_fixed_from_t0_labels':True,'physical_capacity_replaced':False})
        indices={s:{} for s in ('kernel','ledger')};receipts=[];sealed={}
        for s in indices:
            for K in (16,32):(out/s/f'k{K}').mkdir(parents=True)
        guard('explicit lifts and fixed anchors independently verified')
        for K in (16,32):
            for step in range(K):
                tag=f'k{K}-step{step:02d}';olditem=index[tag];original=read(SOURCE/olditem['file'])
                assert digest(SOURCE/olditem['file'])==olditem['sha256'] and sorted(map(tuple,original['edges']))==expected
                assert original['ids']==ids.tolist() and original['D_exact']==str(F(.02)) and original['periods_exact']==['1','1'] and original['thickness_exact']=='1'
                frame=2*step+1;x=b[f'k{K}_positions'][frame];w=wind[K][frame];t=float(b[f'k{K}_time'][frame])
                assert olditem['frame']==frame and olditem['midpoint_time_saved']==t
                rhoK,krecord=kernel_density(ids,x,ids,m,1/32);rhoL,area=ledger_density(x,w,rho0,rings,a0)
                independent,counts=bin_density(ids,x,m,1/32);diff=float(np.max(abs(rhoK-independent)))
                assert np.all(abs(rhoK-independent)<=1e-10*np.maximum(1.,abs(independent)))
                assert krecord['support_count_min']==min(counts) and krecord['support_count_max']==max(counts)
                verify_ring_density(x,w,rho0,rings,a0,rhoL,area)
                for s,rho in (('kernel',rhoK),('ledger',rhoL)):
                    folder=out/s/f'k{K}';density_file=folder/f'density-step{step:02d}.npz';coefficient_file=folder/f'coefficients-step{step:02d}.json'
                    np.savez_compressed(density_file,rho=rho)
                    if s=='ledger':save(folder/f'areas-step{step:02d}.json',{'current_areas_exact':list(map(str,area))})
                    packet=reweight(original,ids,rho,m);identical=verify_coefficients(original,packet,ids,rho);save(coefficient_file,packet);del packet
                    indices[s][tag]={'file':str(coefficient_file.relative_to(out/s)),'sha256':digest(coefficient_file),'midpoint_time_saved':t}
                    receipts.append({'source':s,'K':K,'step':step,'all8192_rates_exactly_verified':True,'all_other_geometry_fields_unchanged':True,
                        'coefficient_identity_with_oracle':identical,'kernel_independent_difference':diff if s=='kernel' else None,
                        'kernel_support':krecord if s=='kernel' else None,'scalar_or_reference_loaded':False})
                del original;guard('density and coefficients preseal',step+1,K)
        save(out/'density-coefficient-audit.json',receipts)
        for s in indices:save(out/s/'graph-index.json',indices[s])
        for q in sorted(out.rglob('*')):
            if q.is_file() and (q.parts[len(out.parts)] in ('kernel','ledger') or q.name in ('anchor-rings.json','integer-winding.npz','lift-audit.json','density-coefficient-audit.json')):
                sealed[str(q.relative_to(out))]=digest(q)
        save(out/'coefficient-seal.json',{'files':sealed,'all_coefficients_sealed_before_scalar_loading':True,
            'new_density_fields':96,'new_coefficient_packets':96,'new_geometry_graphs':0,'kernel_received_oracle_density':False,
            'ledger_given_initial_density_only':True,'material_npz_preseal_keys':['mass'],'source_path_prefix':str(SOURCE.relative_to(ROOT))})
        guard('all96 coefficients independently verified and sealed')
        # Information boundary: first scalar/reference/future oracle/old paths loading.
        d=arrays(SOURCE/'union-definition.npz');np.testing.assert_array_equal(m,d['mass'])
        save(out/'density-diagnostics.json',density_diagnostics(out,ids,labels,rho0,m,b,index))
        runs={};paths={}
        for K in (16,32):
            runs['oracle',K]=read(SOURCE/f'k{K}/native-results.json');paths['oracle',K]=arrays(SOURCE/f'k{K}/trajectory-states.npz')
        for s in ('kernel','ledger'):
            for K in (16,32):
                run,path=native_run(out/s,K,d,indices[s],guard);save(out/s/f'k{K}/native-results.json',run)
                np.savez_compressed(out/s/f'k{K}/trajectory-states.npz',**path);runs[s,K]=run;paths[s,K]=path
                guard(f'{s} K{K} complete')
        result=compare(d,runs,paths);save(out/'results.json',result);guard('frozen comparison complete')
        save(out/'execution.json',{'completed':True,'elapsed_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'new_density_fields':96,'new_coefficient_packets':96,'new_actual_paths':4,'new_independent_paths':4,'new_action_intervals':96,
            'new_geometry_graphs':0,'new_reference_PDE_solves':0,'old_paths_rerun':0,'seed':20261101,
            'coefficient_seal_preceded_scalar_loading':True,'time_order_certified':False,'published_or_pushed':False})
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-started,'partial_data_preserved':True,'repair_or_automatic_extra_run':False});raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
