"""Independent expanded-moment and beta-integral audit of saved-data diagnostics."""
from pathlib import Path
from fractions import Fraction as F
import argparse, ast, math, resource, signal, time, traceback
import numpy as np
import mpmath as mp
from audit_route_decision import ROOT, read, save, digest


def independent_acc(indices, m, u, o):
    ids=list(map(int,indices)); values={}
    dot=lambda x,y:sum((m[i]*x[i]*y[i] for i in ids),F(0))
    linear=lambda x:sum((m[i]*x[i] for i in ids),F(0))
    values['M']=sum((m[i] for i in ids),F(0))
    values['S']=linear(u)-linear(o)
    values['Q']=dot(u,u)-2*dot(u,o)+dot(o,o)
    values['R']=dot(o,o);values['T']=linear(o)
    for name,predicate in [('positive',lambda i:u[i]>o[i]),('negative',lambda i:u[i]<o[i])]:
        selected=[i for i in ids if predicate(i)]
        values[name]=len(selected);values[name+'_mass']=sum((m[i] for i in selected),F(0))
    values['zero']=len(ids)-values['positive']-values['negative']
    relative={i:u[i]/o[i]-1 for i in ids}
    lo=min(ids,key=lambda i:(relative[i],i));hi=max(ids,key=lambda i:(relative[i],-i))
    values.update(relative_min=relative[lo],relative_max=relative[hi],minimum_id=lo,maximum_id=hi)
    return values


def verify_acc(actual,expected):
    assert actual=={k:str(v) if isinstance(v,F) else v for k,v in expected.items()}


def audit(out):
    p=read(out/'protocol.json');lock=read(out/'implementation-lock.json')
    for entries in (p['sources'],lock['files']):
        for name,h in entries.items():assert digest(ROOT/name)==h,name
    assert lock['locked_before_diagnostics']
    parent=ROOT/'runs/reconnect-density-snapshot'
    assert digest(parent/'next-gate-spec.json')==p['source_gate_sha256']
    gate=read(out/'source-gate-spec.json')
    assert gate['new_density_fields_authorized']==0 and gate['parent_decision']=='stop_no_expansion'
    # No adapter/solver imports or NPZ writes in this read-only producer.
    tree=ast.parse((ROOT/'research/reconnect/density_error_structure.py').read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom):assert n.module in ('pathlib','fractions','audit_route_decision')
        if isinstance(n,ast.Import):assert {x.name for x in n.names}<={'argparse','math','resource','signal','time','traceback','numpy','mpmath','matplotlib','matplotlib.pyplot'}
        if isinstance(n,ast.Attribute):assert n.attr not in ('savez','savez_compressed')
    rows=read(out/'exact-error-structure.json');result=read(out/'results.json')
    index=read(parent/'density-index.json');verified=[]
    assert [(r['N'],r['source']) for r in rows]==[(n,s) for n in (64,128,256) for s in ('kernel','ledger')]
    for case in p['cases']:
        N=case['N']
        with np.load(ROOT/case['trajectory'],allow_pickle=False) as z:labels,rho0=z['labels'],z['rho0_point']
        with np.load(ROOT/case['mass_file'],allow_pickle=False) as z:m=list(map(F,z['mass'].tolist()))
        with np.load(ROOT/case['builder'],allow_pickle=False) as z:
            assert z['k32_time'][63].hex()=='0x1.9333333333334p-5';oracle=z['k32_rho'][63]
        assert np.all(oracle.reshape(N,N)==oracle[:N]);o=list(map(F,oracle.tolist()))
        for source in ('kernel','ledger'):
            with np.load(ROOT/index[str(N)]['fields'][source]['file'],allow_pickle=False) as z:u=list(map(F,z['rho'].tolist()))
            row=next(r for r in rows if r['N']==N and r['source']==source)
            parts=[]
            for c,column in enumerate(row['columns']):
                assert len(row['columns'])==N and column['column']==c
                q=independent_acc(range(c,N*N,N),m,u,o);parts.append(q)
                extras={'column','a_exact','relative_column_mean_exact'}
                verify_acc({k:v for k,v in column.items() if k not in extras},q)
                assert F(column['a_exact'])==F(float(labels[c,0]))==F(2*c+1,2*N)
                assert F(column['relative_column_mean_exact'])==q['S']/q['T']
            total={k:sum((q[k] for q in parts),F(0)) for k in ('M','S','Q','R','T','positive_mass','negative_mass')}
            for k in ('positive','negative','zero'):total[k]=sum(q[k] for q in parts)
            lo=min(parts,key=lambda q:(q['relative_min'],q['minimum_id']))
            hi=max(parts,key=lambda q:(q['relative_max'],-q['maximum_id']))
            total.update(relative_min=lo['relative_min'],minimum_id=lo['minimum_id'],relative_max=hi['relative_max'],maximum_id=hi['maximum_id'])
            g=row['global_exact'];additional={'normalized_bias_square','normalized_centered_square','normalized_error_square','centered_error_square'}
            verify_acc({k:v for k,v in g.items() if k not in additional},total)
            centered=total['Q']-total['S']**2/total['M'];assert centered>=0
            assert F(g['centered_error_square'])==centered
            assert F(g['normalized_bias_square'])==total['S']**2/total['M']/total['R']
            assert F(g['normalized_centered_square'])==centered/total['R']
            assert F(g['normalized_error_square'])==total['Q']/total['R']
            for name,mask in [('expansion',rho0>1),('compression',rho0<1)]:
                q=independent_acc(np.flatnonzero(mask),m,u,o);verify_acc(row['groups'][name],q)
                assert q['positive']+q['negative']+q['zero']==N*N//2
                for i in np.flatnonzero(mask):
                    expanding=labels[i,0]<.25 or labels[i,0]>.75
                    assert expanding==(name=='expansion')
            mismatches=sum((u[i]>o[i])!=(rho0[i]>1) or u[i]==o[i] for i in range(N*N))
            assert mismatches==row['sign_mismatch_with_expansion']
            assert row['extrema_material_labels']=={key:labels[total[key]].tolist() for key in ('minimum_id','maximum_id')}
            assert F(row['maximum_within_column_relative_range_exact'])==max(q['relative_max']-q['relative_min'] for q in parts)
            verified.append(row)
    precision=[]
    for digits in (80,100):
        with mp.workdps(digits):
            def v(x):
                f=F(x);return mp.mpf(f.numerator)/f.denominator
            observations=[];exact={}
            for row in verified:
                g=row['global_exact'];exact[row['N'],row['source']]=g
                total=v(g['normalized_error_square']);b=v(g['normalized_bias_square']);c=v(g['normalized_centered_square'])
                metrics={'E2':mp.sqrt(total),'signed_mass_bias':v(g['S'])/v(g['M']),
                    'bias_square_fraction':b/total,'centered_square_fraction':c/total,
                    'centered_relative_norm':mp.sqrt(c),'normalized_bias_square':b,
                    'normalized_centered_square':c,'maximum_within_column_relative_range':v(row['maximum_within_column_relative_range_exact'])}
                groups={}
                for name,q in row['groups'].items():
                    groups[name]={k:mp.nstr(x,65) for k,x in {
                        'mass_fraction':v(q['M'])/v(g['M']),'signed_mean':v(q['S'])/v(q['M']),
                        'error_square_fraction':v(q['Q'])/v(g['Q']),
                        'relative_min':v(q['relative_min']),'relative_max':v(q['relative_max'])}.items()}
                    groups[name].update({k:q[k] for k in ('positive','negative','zero','minimum_id','maximum_id')})
                expected={'N':row['N'],'source':row['source'],**{k:mp.nstr(x,65) for k,x in metrics.items()},
                    'positive':g['positive'],'negative':g['negative'],'zero':g['zero'],
                    'sign_mismatch_with_expansion':row['sign_mismatch_with_expansion'],
                    'extrema_material_labels':row['extrema_material_labels'],'groups':groups}
                assert expected==next(r for r in result['observations'] if r['N']==row['N'] and r['source']==row['source'])
                observations.append(expected)
            pairs=[]
            for source in ('kernel','ledger'):
                for C,N in ((64,128),(128,256)):
                    d={k:F(exact[N,source][k])-F(exact[C,source][k]) for k in ('normalized_error_square','normalized_bias_square','normalized_centered_square')}
                    assert d['normalized_error_square']==d['normalized_bias_square']+d['normalized_centered_square']
                    pairs.append({'source':source,'coarse_N':C,'fine_N':N,**{k+'_change':mp.nstr(v(x),65) for k,x in d.items()},
                        'centered_increase_exceeds_bias_decrease':d['normalized_centered_square']>0 and d['normalized_bias_square']<0 and d['normalized_error_square']>0,
                        'components_are_causal_error_allocation':False})
            assert pairs==result['native_grid_square_error_changes'];precision.append((observations,pairs))
    assert precision[0]==precision[1]
    # Independent beta integrals, without expanding the kernel polynomial.
    def integral(power):
        return sum((coefficient*F(2)**(power+j+1)*math.factorial(power+j)*math.factorial(6)/math.factorial(power+j+7)
            for j,coefficient in enumerate((F(1),F(3),F(35,12)))),F(0))
    moments=read(out/'radial-moments.json');norm=F(9,2)*integral(1);second=F(9,2)*integral(3)
    assert norm==F(moments['normalization_exact'])==1
    assert second==F(moments['radial_second_moment_exact'])==F(14,33)
    assert F(moments['smoothing_laplacian_coefficient_exact'])==second/4==F(7,66)
    assert F(moments['per_axis_second_moment_exact'])==second/2
    assert not moments['new_kernel_sums_or_fields_evaluated']
    old=read(parent/'results.json');assert old['decision']=='stop_no_expansion'==result['parent_decision']
    assert not result['old_cases_rejudged'] and not result['numerical_limit_field_or_convolution_generated']
    for field in ('new_density_fields','new_geometry_or_coefficients','new_scalar_or_reference_solves','new_software_tests'):assert result[field]==0
    assert result['recorded_checks_total']==508 and result['stage3']=='incomplete' and not result['fluid_orc_enabled']
    assert not list(out.rglob('*.npz'))
    preserved=read(out/'preservation.json');assert len(preserved['formal_manifests_verified'])==125
    assert preserved['originals_verified']==782 and preserved['attachments_verified']==15
    for item in preserved['formal_manifests_verified']:assert digest(ROOT/item['manifest'])==item['manifest_sha256']
    ex=read(out/'execution.json');assert ex['completed'] and ex['elapsed_seconds']<120 and ex['peak_RSS_MiB']<1024
    return {'passed':True,'six_saved_fields_exactly_verified':True,'column_profiles_verified':896,
        'two_geometric_regions_per_field_verified':True,'four_square_error_change_identities_verified':True,
        'digits':[80,100],'reported_significant_digits':65,'independent_beta_integrals_verified':True,
        'analytic_limit_proof_is_written_not_numeric_field_evaluation':True,
        'old_stop_preserved':True,'new_fields_or_solves':0,'new_software_tests':0,'recorded_checks_total':508,
        'stage3':'incomplete','fluid_orc_enabled':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--output')
    args=parser.parse_args();start=time.monotonic()
    def alarm(*_):raise TimeoutError('frozen120second structure audit cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        r=audit(Path(args.directory).resolve());elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and rss<1024;r.update(elapsed_seconds=elapsed,peak_RSS_MiB=rss)
        if args.output:save(args.output,r)
        print(r)
    except BaseException as exc:
        if args.output:save(Path(args.output).with_name('failed-audit.json'),{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise
    finally:signal.alarm(0)
