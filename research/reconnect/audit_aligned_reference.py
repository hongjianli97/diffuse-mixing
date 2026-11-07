"""Independent exact-arithmetic checks of saved synthetic equations; no time march."""
from fractions import Fraction as F
from pathlib import Path
import argparse,json,resource,time
import mpmath as mp
import numpy as np
from audit_route_decision import ROOT,read,save,digest


def exact_step(mass,old,g,dt):
    """Rational Thomas elimination, independent of scipy and production assembly."""
    m=list(map(lambda x:F(float(x)),mass));u=list(map(lambda x:F(float(x)),old))
    h=[F(float(dt))*F(float(x))/2 for x in g];n=len(m)
    diagonal=[m[i]+(h[i-1] if i else 0)+(h[i] if i<n-1 else 0) for i in range(n)]
    rhs=[m[i]*u[i]+(h[i-1]*(u[i-1]-u[i]) if i else 0)+(h[i]*(u[i+1]-u[i]) if i<n-1 else 0) for i in range(n)]
    right_diagonal=[m[i]-(h[i-1] if i else 0)-(h[i] if i<n-1 else 0) for i in range(n)]
    for i in range(1,n):
        factor=-h[i-1]/diagonal[i-1]
        diagonal[i]-=factor*(-h[i-1]);rhs[i]-=factor*rhs[i-1]
        assert diagonal[i]>0
    out=[F(0)]*n;out[-1]=rhs[-1]/diagonal[-1]
    for i in range(n-2,-1,-1):out[i]=(rhs[i]+h[i]*out[i+1])/diagonal[i]
    assert sum(a*b for a,b in zip(m,out))==sum(a*b for a,b in zip(m,u))
    return out,right_diagonal


def exact_error(stored,exact):
    return max(float(abs(F(float(a))-b)) for a,b in zip(stored,exact))


def aggregate(edges,parents,values):
    e=list(map(lambda x:F(float(x)),edges));p=list(map(lambda x:F(float(x)),parents))
    v=list(map(lambda x:F(float(x)),values));ans=[]
    for a,b in zip(p[:-1],p[1:]):
        left=e.index(a);right=e.index(b)
        assert sum(e[j+1]-e[j] for j in range(left,right))==b-a
        ans.append(sum((e[j+1]-e[j])*v[j] for j in range(left,right))/(b-a))
    return ans


def compute(directory):
    out=Path(directory);lock=read(out/'implementation-lock.json')
    # Structure of the frozen implementation lock is intentionally checked here.
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert digest(ROOT/'research/rebuild/reference.py')==lock['legacy_reference_sha256']
    assert digest(ROOT/'research/rebuild/becker.py')==lock['legacy_initial_sha256']
    r=read(out/'results.json');tests=read(out/'tests.json');calls=read(out/'calls.json')
    assert r['recorded_checks_total']==502 and r['software_checks_passed']==2
    assert r['new_Becker_reference_solves']==r['new_candidate_transient_runs']==0
    assert r['stage3_physical_acceptance']=='incomplete' and not r['fluid_orc_enabled']
    assert len(calls)==7 and sum(c['linear_solves'] for c in calls)==33
    assert tests['successful'] and tests['checks_run']==2
    eps=np.finfo(float).eps;allowance=64*eps;cases={};reconstructions=0
    for name in ('uniform','nonuniform','constant'):
        z=np.load(out/(name+'.npz'),allow_pickle=False);n=len(z['mass']);steps=len(z['states'])-1
        assert n<=32 and steps<=16
        e=z['edges'];mass=z['mass'];p=z['parent_edges'];states=z['states']
        assert np.array_equal(mass,np.diff(e)) and np.array_equal(e[z['starts']],p)
        assert np.array_equal(states[0],z['c0']) and np.array_equal(states[-1],z['c'])
        maximum=0.;coefficient_error=0.
        for k in range(steps):
            assert float(z['times'][k])==(k+.5)*float(z['dt'])
            exact,right=exact_step(mass,states[k],z['g'][k],float(z['dt']));reconstructions+=1
            assert min(right)>=0
            local=exact_error(states[k+1],exact);maximum=max(maximum,local);assert local<=allowance
            # Exact constant and conservation follow from the reconstructed equations.
            if name=='constant' and np.all(states[k]==states[k][0]):
                assert all(v==F(float(states[k][0])) for v in exact)
            for j,g in enumerate(z['g'][k]):
                face=F(float(e[j+1]));t=F(float(z['times'][k]))
                rho=F(1) if name=='uniform' else 1+face/4+t/8
                intended=rho*rho*F(.02)/(F(float(z['centres'][j+1]))-F(float(z['centres'][j])))
                error=float(abs(F(float(g))-intended)/abs(intended));coefficient_error=max(coefficient_error,error)
                assert error<=allowance
        initial_error=exact_error(z['observed_initial'],aggregate(e,p,z['c0']))
        final_error=exact_error(z['observed_final'],aggregate(e,p,z['c']))
        assert max(initial_error,final_error)<=allowance
        cases[name]={'saved_equations_reconstructed':steps,'maximum_one_step_rounding_difference':maximum,
          'maximum_coefficient_relative_rounding_difference':coefficient_error,'initial_observation_rounding':initial_error,
          'final_observation_rounding':final_error,'exact_mass_conservation_each_reconstructed_equation':True,
          'sufficient_positivity_all_saved_steps':True}
    old=np.load(out/'legacy-uniform.npz',allow_pickle=False);new=np.load(out/'uniform.npz',allow_pickle=False)
    for key in ('edges','c0','c'):assert np.array_equal(old[key],new[key])
    zero=np.load(out/'D0.npz',allow_pickle=False);assert np.array_equal(zero['states'],np.tile(zero['c0'],(9,1)))
    assert len(zero['g'])==len(zero['times'])==0
    a=np.load(out/'aggregation.npz',allow_pickle=False)
    assert np.array_equal(a['base_edges'],a['fine_edges'][::2])
    direct=aggregate(a['fine_edges'],a['parent_edges'],a['arbitrary_fine_values'])
    assert exact_error(a['direct'],direct)<=allowance and exact_error(a['composed'],direct)<=allowance
    precisions=[]
    for digits in (80,100):
        with mp.workdps(digits):
            errors={}
            for key,values in (('parent_edges','parent_c0'),('base_edges','base_c0'),('fine_edges','fine_c0')):
                edges=[mp.mpf(float(x)) for x in a[key]];means=[]
                for left,right in zip(edges[:-1],edges[1:]):
                    means.append(mp.mpf('.5')+(mp.sin(mp.pi*right)-mp.sin(mp.pi*left))/(4*mp.pi*(right-left)))
                error=max(abs(mp.mpf(float(x))-v) for x,v in zip(a[values],means));assert error<allowance
                errors[values]=float(error)
            precisions.append({'digits':digits,'maximum_initial_mean_rounding_by_grid':errors})
    assert precisions[0]['maximum_initial_mean_rounding_by_grid']==precisions[1]['maximum_initial_mean_rounding_by_grid']
    strict=np.load(out/'strict-rejected.npz',allow_pickle=False);monitor=np.load(out/'monitor-rejected.npz',allow_pickle=False)
    centres=(monitor['edges'][1:]+monitor['edges'][:-1])/2;g=1/np.diff(centres)
    exact,margin=exact_step(monitor['mass'],monitor['c0'],g,10.);reconstructions+=1
    assert min(margin)<0 and min(exact)<0
    assert exact_error(strict['margin'],margin)<=allowance*max(1,max(abs(float(x)) for x in margin))
    assert len(strict['raw_state'])==0
    rejection_error=exact_error(monitor['raw_state'],exact);assert rejection_error<=allowance
    byname={c['name']:c for c in calls}
    assert byname['strict-rejected']['linear_solves']==0 and byname['monitor-rejected']['linear_solves']==1
    assert byname['strict-rejected']['outcome']=='StepRestrictionError' and byname['monitor-rejected']['outcome']=='RawStateError'
    assert all(not c.get('states_clipped_or_normalized',False) for c in calls)
    assert reconstructions==25
    return {'passed':True,'independent_exact_rational_equation_reconstructions':reconstructions,
      'new_time_integrations':0,'new_Becker_reference_solves':0,'new_candidate_transient_runs':0,
      'allowance_from_frozen_synthetic_contract':allowance,'cases':cases,'uniform_legacy_bitwise':True,
      'D0_bitwise_identity':True,'complete_child_mass_closure_exact_on_synthetic_grids':True,
      'nested_observation_exact_arithmetic_verified':True,'initial_mean_precision_checks':precisions,
      'rejected_monitor_raw_minimum':float(monitor['raw_state'].min()),'rejected_monitor_rounding_difference':rejection_error,
      'sufficient_condition_failure_and_expected_rejections_verified':True,
      'no_claim_of_physical_reference_accuracy':True,
      'input_NPZ_sha256':{p.name:digest(p) for p in sorted(out.glob('*.npz'))}}


def main(directory,write=False):
    start=time.monotonic();findings=compute(directory)
    if write:
        save(Path(directory)/'audit.json',{'passed':True,'first_attempt_passed':True,'findings':findings,
          'elapsed_seconds':time.monotonic()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    else:assert findings==read(Path(directory)/'audit.json')['findings']
    print(json.dumps(findings,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--write',action='store_true')
    args=parser.parse_args();main(args.directory,args.write)
