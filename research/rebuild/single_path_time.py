"""One actual nonuniform-time BE path through all four observation times."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from scipy.linalg import eigh_tridiagonal
from fixed_be_time import FixedBackwardEuler
from fixed_time_study import time_indicators
from spatial_study import normalized_error


def modal_path(eigenvalues,U,m,c0,endpoints,counts):
    root=np.sqrt(m);coeff=U.T@(root*c0);log_transfer=np.zeros_like(eigenvalues)
    forecasts=[];last=0.
    for t,n in zip(endpoints,counts):
        dt=(t-last)/n
        if np.any(1+dt*eigenvalues<=0):raise ValueError('invalid modal denominators')
        log_transfer-=n*np.log1p(dt*eigenvalues)
        log_exact=-t*eigenvalues;top=np.maximum(log_transfer,log_exact);bottom=np.minimum(log_transfer,log_exact)
        diff=np.sign(log_transfer-log_exact)*np.exp(top)*(-np.expm1(bottom-top))
        forecasts.append(((U@(np.exp(log_transfer)*coeff))/root,(U@(diff*coeff))/root))
        last=t
    return forecasts


def advance_path(c0,m,g,endpoints,counts):
    endpoints=np.asarray(endpoints,dtype=float);counts=np.asarray(counts)
    if len(endpoints)!=len(counts) or len(counts)==0 or not np.isfinite(endpoints).all() or np.any(np.diff(np.r_[0.,endpoints])<=0):
        raise ValueError('strictly increasing positive observation endpoints required')
    if np.any(counts<=0) or np.any(counts!=counts.astype(int)):raise ValueError('positive integer segment counts required')
    c=np.asarray(c0).copy();m=np.asarray(m);g=np.asarray(g);mean=float(m@c/m.sum());initial_mass=float(m@c)
    tiny=np.finfo(float).tiny;mass_scale=max(abs(initial_mass),float(m@abs(c)),tiny)
    lo=initial_lo=float(c.min());hi=initial_hi=float(c.max());drift=residual=balance=0.;states=[];audits=[];last=0.;cumulative=0
    for t,n in zip(endpoints,counts.astype(int)):
        dt=(t-last)/n;solver=FixedBackwardEuler(m,g,dt)
        for _ in range(n):
            nxt=solver.advance(c);dc=nxt-c;flux=dt*g*np.diff(nxt);r=m*dc;r[:-1]-=flux;r[1:]+=flux
            residual=max(residual,float(np.linalg.norm(r)/max(float(np.linalg.norm(m*c)),tiny)))
            drift=max(drift,abs(float(m@(nxt-c0)))/mass_scale)
            E=lambda z:.5*float(m@((z-mean)**2))
            identity=E(nxt)-E(c)+dt*float(g@(np.diff(nxt)**2))+.5*float(m@(dc*dc))
            balance=max(balance,abs(identity)/max(float(m@(c*c)),tiny))
            lo=min(lo,float(nxt.min()));hi=max(hi,float(nxt.max()));c=nxt
        if max(drift,residual,balance)>1e-10 or lo<initial_lo-1e-10 or hi>initial_hi+1e-10:raise RuntimeError('single path invariant failure')
        cumulative+=n;states.append(c.copy());audits.append({'endpoint':float(t),'segment_steps':int(n),'segment_dt':float(dt),'cumulative_steps':int(cumulative),
                    'maximum_relative_mass_drift':drift,'maximum_relative_solve_residual':residual,'maximum_relative_energy_balance_residual':balance,
                    'minimum_all_steps':lo,'maximum_all_steps':hi})
        last=t
    return np.asarray(states),audits


def main(source,uniform_study,output):
    source=Path(source).resolve();study=Path(uniform_study).resolve();output=Path(output);root=Path(__file__).resolve().parents[2]
    prior=json.loads((source/'sha256.json').read_text())
    for path,digest in prior.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('source evidence mismatch: '+path)
    output.mkdir(parents=True,exist_ok=False)
    old=json.loads((study/'results.json').read_text());times=np.array([x['time'] for x in old['runs']])
    uniform_counts=np.array([x['selected_steps'] for x in old['runs']]);maxdt=times/uniform_counts
    counts=np.ceil(np.diff(np.r_[0.,times])/maxdt).astype(int)
    with np.load(source/'mesh-q8.npz') as z:m=z['child_mass'];c0=z['child_c0'];g=z['g'];A=z['A'];mp=z['parent_mass'];cp=z['parent_c0'];faces=z['faces']
    right=faces[:-1]>=0;diag=np.zeros(len(m));diag[:-1]+=g;diag[1:]+=g
    vals,U=eigh_tridiagonal(diag/m,-g/np.sqrt(m[:-1]*m[1:]))
    spatial=json.loads((source/'results.json').read_text());limit=json.loads((study/'config.json').read_text())['time_to_spatial_diagnostic_limit']
    config={'study':'single continuous BE path through all selected observation times','source':str(source),'uniform_time_study':str(study),
            'endpoints':times.tolist(),'initial_segment_counts':counts.tolist(),
            'selection_input_sha256':{name:hashlib.sha256((study/name).read_bytes()).hexdigest() for name in ('config.json','results.json')},
            'selection':'ceil(segment length / previous independent horizon dt); at most three global binary refinements if modal check requires',
            'diagnostic_limit_unchanged':limit,'no_state_reinitialization_at_checkpoints':True,'fluid_orc_enabled':False,'third_stage_complete':False}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    trials=[]
    for attempt in range(4):
        forecast=modal_path(vals,U,m,c0,times,counts);indicators=[]
        for t,(state,delta) in zip(times,forecast):
            s=next(x for x in spatial['runs'] if x['resolution']==8 and x['time']==t)
            indicators.append(time_indicators(delta,m,c0,A,mp,cp,right,s['right_inventory_reference'],s))
        maximum=max(x['maximum_ratio'] for x in indicators);trials.append({'counts':counts.tolist(),'max_predicted_time_to_spatial_ratio':maximum})
        if maximum<=limit:break
        counts*=2
    else:raise RuntimeError('bounded single-path time selection did not meet unchanged diagnostic')
    start=time.monotonic();states,audits=advance_path(c0,m,g,times,counts);wall=time.monotonic()-start;rows=[]
    for i,t in enumerate(times):
        s=next(x for x in spatial['runs'] if x['resolution']==8 and x['time']==t)
        with np.load(source/f'result-q8-t{t:g}.npz') as z:semi=z['child_solution'];ref=z['child_reference'];pref=z['parent_reference']
        actual=states[i];indicator=time_indicators(actual-semi,m,c0,A,mp,cp,right,s['right_inventory_reference'],s)
        if indicator['maximum_ratio']>limit:raise RuntimeError('actual accumulated time error exceeds unchanged diagnostic')
        gap=float(np.max(abs(actual-forecast[i][0])))
        if gap>1e-10:raise RuntimeError('single path modal/direct disagreement')
        row={'time':float(t),'audit':audits[i],**indicator,'modal_vs_actual_max_abs_difference':gap,
             'full_total_error':normalized_error(m,c0,actual-ref),'parent_total_error':normalized_error(mp,cp,A@actual-pref),
             'right_inventory':float(m[right]@actual[right]),'right_inventory_reference':s['right_inventory_reference'],
             'right_inventory_signed_relative_error':float((m[right]@actual[right]-s['right_inventory_reference'])/s['right_inventory_reference'])}
        rows.append(row)
        np.savez_compressed(output/f'checkpoint-t{t:g}.npz',c=actual,mass=m,c0=c0,reference=ref,semidiscrete=semi,parent_c=A@actual,parent_reference=pref,parent_mass=mp,parent_c0=cp)
        print('single path',t,'steps',audits[i]['cumulative_steps'],'max time/spatial',indicator['maximum_ratio'],flush=True)
    (output/'results.json').write_text(json.dumps({'selection_trials':trials,'final_segment_counts':counts.tolist(),'total_steps':int(counts.sum()),
                'wall_seconds':wall,'runs':rows,'all_actual_error_separation_diagnostics_met':True,'third_stage_physical_acceptance':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('uniform_study');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.uniform_study,a.output)
