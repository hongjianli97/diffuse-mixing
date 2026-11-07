"""Execute the single frozen N64 joint control once, preserving any failure."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,resource,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from compression_joint_core import ORIGINAL,CALIBRATION,nested_samples,save_npz,snapshots,compare
from compression_joint_reference import recorded_reference_path
from compression_density_sources import ledger_density
from compression_density_audit_core import verify_lifts,verify_ring_density,verify_coefficients
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from audit_shear_voronoi_time import audit_geometry
from t1_density_sources import reweight
from shear_union64_study import native_run,arrays
from material_union import observe_path

PARENT=ROOT/'runs/reconnect-compression-joint-definition'


def main(directory):
    out=Path(directory).resolve();start=time.monotonic();gate=read(out/'source-gate-spec.json')
    def guard(stage,*args):
        elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>=1600 or rss>=1024:raise RuntimeError(f'frozen main cap {elapsed}s/{rss}MiB')
        if not args or args[0]%8==0:print(f'{stage} {args}: {elapsed:.2f}s,peak{rss:.1f}MiB',flush=True)
    def alarm(*_):raise TimeoutError('frozen1600second main cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(1600)
    try:
        assert digest(PARENT/'next-gate-spec.json')==read(out/'protocol.json')['source_gate_sha256']
        for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
        for name,h in read(out/'protocol.json')['sources'].items():assert digest(ROOT/name)==h,name
        test=read(out/'tests.json');assert test['successful'] and test['checks_run']==2 and test['elapsed_seconds']<120
        assert gate['N']==64 and gate['new_candidate_K']==[64] and gate['new_reference_cases']==[[2048,1024],[2048,2048]]
        with np.load(ORIGINAL/'trajectory-source.npz',allow_pickle=False) as z:ids,labels,rho0=z['ids'],z['labels'],z['rho0_point']
        with np.load(ORIGINAL/'union-definition.npz',allow_pickle=False) as z:m=z['mass']
        with np.load(ORIGINAL/'builder-inputs.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(ids,z['ids']);old_t,old_x=z['k32_time'],z['k32_positions']
        b=nested_samples(ids,labels,rho0,old_t,old_x)
        old_w=arrays(CALIBRATION/'integer-winding.npz')['k32']
        np.testing.assert_array_equal(b['winding'][::2],old_w)
        save(out/'lift-audit.json',verify_lifts(labels,b['time'],b['positions'],b['winding']))
        save_npz(out/'builder-inputs.npz',**b)
        anchor=read(CALIBRATION/'anchor-rings.json');save(out/'anchor-rings.json',anchor)
        assert all(F(a)==F(1,2048) for a in anchor['initial_areas_exact'])
        probes=read(ROOT/'research/reconnect/compression_union64_protocol.json')['fixed_geometry_probe_ids']
        expected=sorted({tuple(sorted((i,j))) for i in range(4096) for j in ((i//64)*64+(i%64+1)%64,((i//64+1)%64)*64+i%64)})
        indices={s:{} for s in ('oracle','ledger')};receipts=[]
        (out/'geometry').mkdir()
        for s in indices:(out/s/'k64').mkdir(parents=True)
        guard('shared frames and integer lifts verified')
        for step in range(64):
            x=b['positions'][2*step+1];rhoO=b['midpoint_rho_oracle'][step]
            graph=build_periodic_faces(ids,x,rhoO,.02,(1,1),1)
            packet=coefficient_packet(graph);del graph
            assert sorted(map(tuple,packet['edges']))==expected
            gp=out/f'geometry/step{step:02d}.json';save(gp,packet)
            _,_,_,certificate=audit_geometry(packet,x,rhoO,ids,probes)
            save(out/f'geometry/audit-step{step:02d}.json',{'cache_sha256':digest(gp),'certificate':certificate,
                'pure_geometry_no_mass_c0_reference_loaded':True})
            rhoL,area=ledger_density(x,b['winding'][2*step+1],rho0,anchor['rings'],anchor['initial_areas_exact'])
            verify_ring_density(x,b['winding'][2*step+1],rho0,anchor['rings'],anchor['initial_areas_exact'],rhoL,area)
            save(out/f'ledger/k64/areas-step{step:02d}.json',{'current_areas_exact':list(map(str,area))})
            for s,rho in (('oracle',rhoO),('ledger',rhoL)):
                folder=out/s/'k64';save_npz(folder/f'density-step{step:02d}.npz',rho=rho)
                coefficients=reweight(packet,ids,rho,m)
                identical=verify_coefficients(packet,coefficients,ids,rho)
                file=folder/f'coefficients-step{step:02d}.json';save(file,coefficients);del coefficients
                indices[s][f'k64-step{step:02d}']={'file':str(file.relative_to(out/s)),'sha256':digest(file),
                    'midpoint_time_saved':float(b['time'][2*step+1])}
                receipts.append({'source':s,'step':step,'all8192_rates_verified':True,'oracle_identity':identical,
                    'geometry_sha256':digest(gp),'scalar_or_reference_loaded':False})
            del packet;gc.collect();guard('geometry and both coefficients verified',step+1)
        for s,index in indices.items():save(out/s/'graph-index.json',index)
        save(out/'density-coefficient-audit.json',receipts)
        sealed={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and
            (p.relative_to(out).parts[0] in ('oracle','ledger','geometry') or p.name in
             ('builder-inputs.npz','anchor-rings.json','lift-audit.json','density-coefficient-audit.json'))}
        save(out/'coefficient-seal.json',{'files':sealed,'new_geometry_graphs':64,'new_coefficient_packets':128,
            'all_pure_geometry_and_rates_audited_before_scalar_or_reference_loading':True,'material_npz_preseal_keys':['mass']})
        guard('all64 geometry and128 coefficients sealed before scalars')
        # Information boundary. All future reference/scalar loads occur below.
        d=arrays(ORIGINAL/'union-definition.npz');np.testing.assert_array_equal(m,d['mass'])
        with np.load(ORIGINAL/'original-parent-reference.npz',allow_pickle=False) as z:pm,pc=z['mass'],z['c0']
        (out/'reference').mkdir();refs={}
        for K in (1024,2048):
            r,a=recorded_reference_path(2048,K,128,pm)
            save_npz(out/f'reference/n2048-k{K}.npz',**r)
            a['initial_N128_c0_max_difference']=float(np.max(abs(r['projected_states'][0]-pc)))
            save(out/f'reference/n2048-k{K}-audit.json',a)
            assert a['initial_N128_c0_max_difference']<=3e-16
            assert a['B_energy_max_increase']<0 and a['CN_energy_balance_max_abs_residual']<1e-15
            refs[K]=observe_path(r['projected_states'],pm,m);del r;gc.collect();guard(f'reference n2048-k{K} complete')
        new={'mass':m,'c0':d['c0'],'reference':refs[2048],
            'reference_space_coarse':d['reference'],'reference_space_fine':refs[1024],
            'reference_time_check':refs[1024],'space':d['reference']-refs[1024],
            'time':refs[1024]-refs[2048]}
        save_npz(out/'comparison-inputs.npz',**new)
        for s in ('oracle','ledger'):
            run,path=native_run(out/s,64,new,indices[s],guard)
            save(out/s/'k64/native-results.json',run);save_npz(out/s/'k64/trajectory-states.npz',**path)
            del run,path;gc.collect();guard(f'{s} complete actual and independent paths')
        states,independent,mt=snapshots(out);result=compare(new,states,independent,mt);save(out/'results.json',result)
        guard('all registered comparisons saved')
        save(out/'execution.json',{'completed':True,'elapsed_seconds':time.monotonic()-start,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'new_position_frames':64,
            'new_geometry_graphs':64,'new_density_fields':128,'new_coefficient_packets':128,
            'new_actual_paths':2,'new_independent_paths':2,'new_action_intervals':128,'new_reference_PDE_solves':2,
            'new_reference_CN_intervals':3072,'old_paths_rerun':0,'kernel_rerun':False,'published_or_pushed':False})
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-start,'partial_data_preserved':True,'automatic_retry':False});raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
