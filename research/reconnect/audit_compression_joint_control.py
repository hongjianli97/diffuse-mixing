"""Audit saved arrays; no reference solve, geometry builder or actual path rerun."""
from pathlib import Path
from fractions import Fraction as F
import argparse,resource,signal,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from compression_joint_core import ORIGINAL,CALIBRATION,snapshots,compare
from compression_density_audit_core import verify_lifts,verify_ring_density,verify_coefficients
from compression_joint_interval_audit import audit_intervals
from compression_joint_budget_audit import audit_budgets
from shear_union64_study import arrays


def audit_references(out,d):
    with np.load(ORIGINAL/'original-parent-reference.npz',allow_pickle=False) as z:pm,pc=z['mass'],z['c0']
    reports=[]
    for K in (1024,2048):
        r=arrays(out/f'reference/n2048-k{K}.npz');M=2048;B=r['B_history'];mass=r['mass_1d'];dt=.05/K
        assert B.shape==(K+1,M) and np.isfinite(B).all()
        np.testing.assert_array_equal(B[::K//2],r['B_snapshots']);np.testing.assert_array_equal(B[0],r['initial_B'])
        np.testing.assert_array_equal(r['step_time'],np.linspace(0,.05,K+1));np.testing.assert_array_equal(r['snapshot_time'],[0.,.025,.05])
        axis=(np.arange(M)+.5)/M;C=np.sinc(1/M)*np.cos(2*np.pi*axis)
        C2=(1+np.sinc(2/M)*np.cos(4*np.pi*axis))/2;rho=1+.25*C
        np.testing.assert_array_equal(mass,rho/M);np.testing.assert_allclose(B[0],(C+.25*C2)/rho,rtol=0,atol=1e-15)
        maximum=0.;face=(np.arange(M)+1)/M;cos=np.cos(2*np.pi*face)
        for j in range(K):
            before,after=B[j],B[j+1];g=.02*M*(1+.25*cos)/(1+(.25*(j+.5)/K)*cos)**2
            average=(before+after)/2;flux=g*(np.roll(average,-1)-average)
            residual=mass*(after-before)-dt*(flux-np.roll(flux,1))
            scaled=float(np.max(abs(residual))/(mass.max()*max(1.,np.max(abs(before)),np.max(abs(after)))))
            maximum=max(maximum,scaled);assert scaled<1e-10
            assert r['B_energy'][j+1]<r['B_energy'][j]
            assert abs(r['CN_energy_balance_residual'][j])<1e-15
            # Independent face energy and inventory observations, no sparse solve.
            energy=float(.5*mass@(after*after));inv=float(mass@after)
            assert energy==r['B_energy'][j+1] and inv==r['B_inventory'][j+1]
            balance=energy-r['B_energy'][j]+dt*float(g@((np.roll(average,-1)-average)**2))
            assert abs(balance)<1e-15
        observed=[];max_projection=0.
        for k,t in enumerate((0.,.025,.05)):
            child=mass.reshape(128,-1).sum(axis=1);component=(mass*B[k*K//2]).reshape(128,-1).sum(axis=1)
            Y=np.sinc(1/128)*np.cos(2*np.pi*(np.arange(128)+.5)/128)
            parent=((.5*child[None,:]/128+.2*np.exp(-.02*(2*np.pi)**2*t)*Y[:,None]*component[None,:]/128)/pm.reshape(128,128)).ravel()
            delta=float(np.max(abs(parent-r['projected_states'][k])));assert delta<1e-14;max_projection=max(max_projection,delta)
            # Independent four-child observation from the SAVED parent means.
            saved=r['projected_states'][k].reshape(128,128);mm=pm.reshape(128,128)
            q=mm*saved
            obs=(q[0::2,0::2]+q[0::2,1::2]+q[1::2,0::2]+q[1::2,1::2]).ravel()/d['mass']
            observed.append(obs)
        observed=np.array(observed);saved=d['reference_space_fine'] if K==1024 else d['reference']
        assert np.max(abs(observed-saved))<=64*np.finfo(float).eps
        assert np.max(abs(r['projected_states'][0]-pc))<=3e-16
        reports.append({'columns':M,'steps':K,'all_CN_intervals_verified':True,'maximum_scaled_residual':maximum,
            'maximum_independent_parent_projection_difference':max_projection,'observed_initial_bridge_preserved':True})
    return reports


def main(directory,write=False):
    out=Path(directory).resolve();start=time.monotonic()
    def guard(stage):
        elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<400 and rss<1024
        print(f'{stage}: {elapsed:.2f}s,peak{rss:.1f}MiB',flush=True)
    def alarm(*_):raise TimeoutError('frozen400second saved audit cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(400)
    try:
        for key in ('implementation-lock.json','protocol.json'):
            for name,h in read(out/key)['files' if key.startswith('implementation') else 'sources'].items():assert digest(ROOT/name)==h,name
        ex=read(out/'execution.json');assert ex['completed'] and ex['elapsed_seconds']<1600 and ex['peak_RSS_MiB']<1024
        assert read(out/'tests.json')['successful'] and read(out/'tests.json')['checks_run']==2
        seal=read(out/'coefficient-seal.json');assert seal['all_pure_geometry_and_rates_audited_before_scalar_or_reference_loading']
        for name,h in seal['files'].items():assert digest(out/name)==h,name
        b=arrays(out/'builder-inputs.npz');d=arrays(out/'comparison-inputs.npz');old=arrays(ORIGINAL/'union-definition.npz')
        for key in ('mass','c0'):np.testing.assert_array_equal(d[key],old[key])
        np.testing.assert_array_equal(d['reference_space_coarse'],old['reference'])
        np.testing.assert_array_equal(d['reference_time_check'],d['reference_space_fine'])
        np.testing.assert_array_equal(d['space'],d['reference_space_coarse']-d['reference_space_fine'])
        np.testing.assert_array_equal(d['time'],d['reference_time_check']-d['reference'])
        with np.load(ORIGINAL/'trajectory-source.npz',allow_pickle=False) as z:ids,labels,rho0=z['ids'],z['labels'],z['rho0_point']
        with np.load(ORIGINAL/'builder-inputs.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(b['time'][::2],z['k32_time']);np.testing.assert_array_equal(b['positions'][::2],z['k32_positions'])
        np.testing.assert_array_equal(b['ids'],ids)
        np.testing.assert_array_equal(b['winding'][::2],arrays(CALIBRATION/'integer-winding.npz')['k32'])
        assert verify_lifts(labels,b['time'],b['positions'],b['winding'])==read(out/'lift-audit.json')
        anchor=read(out/'anchor-rings.json');assert anchor==read(CALIBRATION/'anchor-rings.json')
        receipts=[]
        for step in range(64):
            geometry=out/f'geometry/step{step:02d}.json';packet=read(geometry)
            cert=read(out/f'geometry/audit-step{step:02d}.json');assert cert['cache_sha256']==digest(geometry)
            assert cert['certificate']['shared_positive_faces']==8192 and cert['pure_geometry_no_mass_c0_reference_loaded']
            x=b['positions'][2*step+1];t=b['time'][2*step+1]
            expected=rho0/(1+(.25*t/.05)*np.cos(2*np.pi*labels[:,0]))
            for s in ('oracle','ledger'):
                folder=out/s/'k64';rho=arrays(folder/f'density-step{step:02d}.npz')['rho']
                if s=='oracle':np.testing.assert_array_equal(rho,expected)
                else:verify_ring_density(x,b['winding'][2*step+1],rho0,anchor['rings'],anchor['initial_areas_exact'],rho,
                    read(folder/f'areas-step{step:02d}.json')['current_areas_exact'])
                identical=verify_coefficients(packet,read(folder/f'coefficients-step{step:02d}.json'),ids,rho)
                receipts.append({'source':s,'step':step,'all8192_rates_verified':True,'oracle_identity':identical,
                    'geometry_sha256':digest(geometry),'scalar_or_reference_loaded':False})
            if (step+1)%16==0:guard(f'saved density and coefficients {step+1}/64')
        assert receipts==read(out/'density-coefficient-audit.json')
        references=audit_references(out,d);guard('3072 saved reference intervals verified')
        intervals=audit_intervals(out,d['mass'],d['c0']);guard('128 saved action intervals verified')
        states,independent,mt=snapshots(out);result=read(out/'results.json');assert compare(d,states,independent,mt)==result
        budgets=audit_budgets(d,states,independent,mt,result);guard('80/100 digit budgets and time diagnostics verified')
        findings={'passed':True,'references':references,'intervals':intervals,'high_precision':budgets,
            'geometry_preseal_audit_certificates_verified':64,'coefficient_packets_verified':128,
            'no_actual_path_reference_or_geometry_reruns':True,'old_rows_unchanged':True,
            'recorded_checks_total':508,'stage3':'incomplete','fluid_orc_enabled':False}
        if write:save(out/'audit.json',{'passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-start,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
        else:assert findings==read(out/'audit.json')['findings']
    except BaseException as exc:
        if write:save(out/'failed-audit.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
