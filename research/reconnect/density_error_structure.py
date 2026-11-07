"""Read-only signed error decomposition, regional accounting and column profiles."""
from pathlib import Path
from fractions import Fraction as F
import argparse, math, resource, signal, time, traceback
import numpy as np
import mpmath as mp
from audit_route_decision import ROOT, read, save, digest

PARENT = ROOT/'runs/reconnect-density-snapshot'
PROTOCOL = ROOT/'research/reconnect/density_error_structure_protocol.json'
SUMS = ('M','S','Q','R','T','positive_mass','negative_mass')


def empty():
    return {**{k:F(0) for k in SUMS}, 'positive':0,'negative':0,'zero':0,
            'relative_min':None,'relative_max':None,'minimum_id':None,'maximum_id':None}


def add(acc, i, m, e, o, rel, terms):
    for key, term in zip(('M','S','Q','R','T'), terms): acc[key] += term
    if e > 0: acc['positive'] += 1; acc['positive_mass'] += m
    elif e < 0: acc['negative'] += 1; acc['negative_mass'] += m
    else: acc['zero'] += 1
    if acc['relative_min'] is None or rel < acc['relative_min']:
        acc['relative_min'],acc['minimum_id'] = rel,i
    if acc['relative_max'] is None or rel > acc['relative_max']:
        acc['relative_max'],acc['maximum_id'] = rel,i


def serialized(acc):
    return {k:str(v) if isinstance(v,F) else v for k,v in acc.items()}


def analyze(N, source, labels, rho0, mass, oracle, density):
    assert np.all(oracle.reshape(N,N) == oracle[:N])
    groups = {'expansion':empty(),'compression':empty()}; columns = [empty() for _ in range(N)]
    total = empty(); disagreement = 0
    for i,(w,z,u,r0) in enumerate(zip(mass.tolist(),oracle.tolist(),density.tolist(),rho0.tolist())):
        m,o,v = F(w),F(z),F(u); e = v-o; rel = e/o
        a = F(float(labels[i,0])); expanding = a < F(1,4) or a > F(3,4)
        assert a not in (F(1,4),F(3,4)) and expanding == (r0 > 1)
        assert r0 != 1 and m > 0 and o > 0
        group = 'expansion' if expanding else 'compression'
        terms = (m,m*e,m*e*e,m*o*o,m*o)
        for acc in (total,groups[group],columns[i%N]): add(acc,i,m,e,o,rel,terms)
        disagreement += int((e > 0) != expanding or e == 0)
    for key in SUMS:
        assert total[key] == sum((g[key] for g in groups.values()),F(0))
        assert total[key] == sum((g[key] for g in columns),F(0))
    center = total['Q']-total['S']**2/total['M']; assert center >= 0
    global_exact = {**serialized(total),
        'normalized_bias_square':str(total['S']**2/total['M']/total['R']),
        'normalized_centered_square':str(center/total['R']),
        'normalized_error_square':str(total['Q']/total['R']),
        'centered_error_square':str(center)}
    return {'N':N,'source':source,'global_exact':global_exact,
        'groups':{k:serialized(v) for k,v in groups.items()},
        'extrema_material_labels':{key:labels[total[key]].tolist() for key in ('minimum_id','maximum_id')},
        'sign_mismatch_with_expansion':disagreement,
        'columns':[{'column':i,'a_exact':str(F(float(labels[i,0]))),
            'relative_column_mean_exact':str(q['S']/q['T']),**serialized(q)} for i,q in enumerate(columns)],
        'maximum_within_column_relative_range_exact':str(max(q['relative_max']-q['relative_min'] for q in columns))}


def evaluate(rows, digits):
    with mp.workdps(digits):
        def value(x):
            q=F(x); return mp.mpf(q.numerator)/q.denominator
        report=[]; fractions={}
        for row in rows:
            g=row['global_exact']; b=value(g['normalized_bias_square']); c=value(g['normalized_centered_square'])
            total=value(g['normalized_error_square']); fractions[row['N'],row['source']]=g
            metrics={'E2':mp.sqrt(total),'signed_mass_bias':value(g['S'])/value(g['M']),
                'bias_square_fraction':b/total,'centered_square_fraction':c/total,
                'centered_relative_norm':mp.sqrt(c),'normalized_bias_square':b,
                'normalized_centered_square':c,'maximum_within_column_relative_range':value(row['maximum_within_column_relative_range_exact'])}
            groups={}
            for name,q in row['groups'].items():
                groups[name]={k:mp.nstr(x,65) for k,x in {
                    'mass_fraction':value(q['M'])/value(g['M']),
                    'signed_mean':value(q['S'])/value(q['M']),
                    'error_square_fraction':value(q['Q'])/value(g['Q']),
                    'relative_min':value(q['relative_min']),'relative_max':value(q['relative_max'])}.items()}
                groups[name].update({k:q[k] for k in ('positive','negative','zero','minimum_id','maximum_id')})
            report.append({'N':row['N'],'source':row['source'],**{k:mp.nstr(v,65) for k,v in metrics.items()},
                'positive':g['positive'],'negative':g['negative'],'zero':g['zero'],
                'sign_mismatch_with_expansion':row['sign_mismatch_with_expansion'],
                'extrema_material_labels':row['extrema_material_labels'],'groups':groups})
        pairs=[]
        for source in ('kernel','ledger'):
            for coarse,fine in ((64,128),(128,256)):
                c,f=fractions[coarse,source],fractions[fine,source]
                delta={k:F(f[k])-F(c[k]) for k in ('normalized_error_square','normalized_bias_square','normalized_centered_square')}
                assert delta['normalized_error_square']==delta['normalized_bias_square']+delta['normalized_centered_square']
                pairs.append({'source':source,'coarse_N':coarse,'fine_N':fine,
                    **{k+'_change':mp.nstr(value(v),65) for k,v in delta.items()},
                    'centered_increase_exceeds_bias_decrease':delta['normalized_centered_square']>0 and
                        delta['normalized_bias_square']<0 and delta['normalized_error_square']>0,
                    'components_are_causal_error_allocation':False})
        return {'observations':report,'native_grid_square_error_changes':pairs}


def radial_moments():
    polynomial=[F(0) for _ in range(9)]
    for k in range(7):
        for j,c in enumerate((F(1),F(3),F(35,12))):
            polynomial[k+j] += F(math.comb(6,k))*(-F(1,2))**k*c
    integral=lambda power:sum((c*F(2)**(i+power+1)/(i+power+1) for i,c in enumerate(polynomial)),F(0))
    normalization=F(9,2)*integral(1); second=F(9,2)*integral(3)
    assert normalization==1 and second==F(14,33)
    return {'phi_polynomial_coefficients':list(map(str,polynomial)),
        'normalization_exact':str(normalization),'radial_second_moment_exact':str(second),
        'per_axis_second_moment_exact':str(second/2),'smoothing_laplacian_coefficient_exact':str(second/4),
        'new_kernel_sums_or_fields_evaluated':False}


def plot(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True,sharey=True,constrained_layout=True)
    colors={64:'#244b81',128:'#167b6b',256:'#c2691d'}
    for ax,source in zip(axes,('kernel','ledger')):
        ax.axvspan(.25,.75,color='#dce4ef',alpha=.5,label='Compression region')
        ax.axhline(0,color='#505050',linewidth=.7)
        for row in rows:
            if row['source']==source:
                x=[float(F(c['a_exact'])) for c in row['columns']]
                y=[float(F(c['relative_column_mean_exact'])) for c in row['columns']]
                ax.plot(x,y,color=colors[row['N']],label=f"N={row['N']}",linewidth=1.5)
        ax.set_title(source); ax.set_xlabel('Material coordinate a'); ax.set_xlim(0,1)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0)); ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='lower right')
    axes[0].set_ylabel('Signed relative density error (mass-weighted column mean)')
    fig.suptitle('Saved snapshot only: h / material spacing = 2',fontsize=12)
    fig.savefig(out/'signed-error-profiles.png',dpi=160)
    fig.savefig(out/'signed-error-profiles.svg')
    plt.close(fig)


def preserve():
    old=read(PARENT/'preservation.json'); p=PARENT/'sha256.json'
    formal=old['formal_manifests_verified']+[{'manifest':str(p.relative_to(ROOT)),
        'manifest_sha256':digest(p),'entries_verified':len(read(p))}]
    assert len(formal)==125
    for item in formal:
        p=ROOT/item['manifest']; assert digest(p)==item['manifest_sha256']
        entries=read(p); assert len(entries)==item['entries_verified']
        for name,h in entries.items(): assert digest(ROOT/name)==h,name
    originals=read('/workspace/original-handoff-sha256.json'); attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for name,h in originals.items(): assert digest(ROOT/name)==h,name
    for item in attachments:
        p=Path('/workspace')/item['path']; assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
    for item in old['historical_diagnostic_differences_retained']:
        p=ROOT/item['manifest']; assert digest(p)==item['manifest_sha256']
        diff=[{'path':k,'saved':v,'current':digest(ROOT/k)} for k,v in read(p).items() if digest(ROOT/k)!=v]
        assert sorted(diff,key=lambda x:x['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda x:x['path'])
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],
        'all_prior_scientific_files_unchanged':True}


def main(directory):
    out=Path(directory).resolve();out.mkdir(exist_ok=False);start=time.monotonic()
    def alarm(*_):raise TimeoutError('frozen120second read-only structure cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='read_only_saved_density_error_structure_and_consistency_review'
        assert gate['new_density_fields_authorized']==0 and gate['parent_decision']=='stop_no_expansion'
        assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        save(out/'protocol.json',p);save(out/'source-gate-spec.json',gate)
        save(out/'implementation-lock.json',{'files':{s:digest(ROOT/s) for s in p['implementation_files']},'locked_before_diagnostics':True})
        save(out/'preservation.json',preserve())
        index=read(PARENT/'density-index.json'); previous=read(PARENT/'exact-moments.json');rows=[]
        for case in p['cases']:
            N=case['N']
            with np.load(ROOT/case['trajectory'],allow_pickle=False) as z:ids,labels,rho0=z['ids'],z['labels'],z['rho0_point']
            with np.load(ROOT/case['mass_file'],allow_pickle=False) as z:m=z['mass']
            with np.load(ROOT/case['builder'],allow_pickle=False) as z:
                assert np.array_equal(ids,z['ids']) and z['k32_time'][63].hex()=='0x1.9333333333334p-5'
                o=z['k32_rho'][63]
            assert np.array_equal(ids,np.arange(N*N)) and labels.shape==(N*N,2)
            for source in ('kernel','ledger'):
                f=ROOT/index[str(N)]['fields'][source]['file']
                with np.load(f,allow_pickle=False) as z:u=z['rho']
                row=analyze(N,source,labels,rho0,m,o,u)
                old=next(q['moments'] for q in previous if q['N']==N and q['source']==source)
                for k,name in [('M','mass'),('S','signed_error_mass'),('Q','error_square'),('R','oracle_square')]:
                    assert row['global_exact'][k]==old[name]
                rows.append(row)
            print(f'N{N} saved signed errors accounted: {time.monotonic()-start:.2f}s',flush=True)
        save(out/'exact-error-structure.json',rows)
        r80,r100=evaluate(rows,80),evaluate(rows,100);assert r80==r100
        r80.update(parent_decision='stop_no_expansion',old_cases_rejudged=False,
            new_density_fields=0,new_geometry_or_coefficients=0,new_scalar_or_reference_solves=0,
            new_software_tests=0,recorded_checks_total=508,stage3='incomplete',fluid_orc_enabled=False,
            numerical_limit_field_or_convolution_generated=False)
        save(out/'results.json',r80);save(out/'radial-moments.json',radial_moments());plot(out,rows)
        for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
        elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<120 and rss<1024
        save(out/'execution.json',{'completed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':rss,
            'new_physical_fields_or_solves':0,'new_software_tests':0,'old_tests_rerun':0,'published_or_pushed':False})
        print('Saved error decomposition and finite-stencil consistency review complete.')
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc(),'partial_data_preserved':True,'automatic_retry':False});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    main(parser.parse_args().directory)
