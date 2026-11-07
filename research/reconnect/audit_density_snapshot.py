"""Independent pre-oracle density audit and post-seal exact-moment verification."""
from pathlib import Path
from fractions import Fraction as F
import argparse, ast, gc, hashlib, resource, signal, time, traceback
import numpy as np
import mpmath as mp
from audit_route_decision import ROOT, read, save, digest
from compression_density_audit_core import bin_density, verify_lifts, verify_ring_density

PARENT = ROOT/'runs/reconnect-density-snapshot-definition'
OLD = ROOT/'runs/reconnect-compression-density-calibration'


def preseal_audit(out, protocol, index, guard):
    """Uses no future oracle. Saves six verification arrays, not new physical samples."""
    start = time.monotonic(); rows = []
    original_receipts = read(OLD/'density-coefficient-audit.json')
    for case in protocol['cases']:
        N = case['N']; item = index[str(N)]; guard(f'N{N} independent density audit begins')
        with np.load(ROOT/case['trajectory'],allow_pickle=False) as z:
            ids,labels,rho0 = z['ids'],z['labels'],z['rho0_point']
        with np.load(ROOT/case['mass_file'],allow_pickle=False) as z: m = z['mass']
        with np.load(ROOT/case['builder'],allow_pickle=False) as z:
            assert np.array_equal(z['ids'],ids)
            t,x = z['k32_time'],z['k32_positions']
        assert np.array_equal(ids,np.arange(N*N)) and np.array_equal(t,np.linspace(0,.05,65))
        assert t[63].hex() == '0x1.9333333333334p-5'
        with np.load(ROOT/item['winding_file'],allow_pickle=False) as z: w = z[item['winding_key']]
        lifts = verify_lifts(labels,t,x,w)
        anchor = read(ROOT/item['anchor_file']); rings = anchor['rings']; a0 = anchor['initial_areas_exact']
        assert len(rings) == N*N and len(a0) == N*N
        assert anchor['images_fixed_from_t0_labels'] and not anchor['physical_capacity_replaced']
        for i,ring in enumerate(rings):
            expected = {i//N*N+(i+1)%N,i//N*N+(i-1)%N,(i+N)%(N*N),(i-N)%(N*N)}
            assert {v[0] for v in ring} == expected and F(a0[i]) == F(2,N*N)
            for j,sx,sy in ring:
                dx = F(float(labels[j,0]))+sx-F(float(labels[i,0]))
                dy = F(float(labels[j,1]))+sy-F(float(labels[i,1]))
                assert abs(dx)+abs(dy) == F(1,N)
        verify_ring_density(labels,np.zeros(labels.shape,dtype=np.int64),rho0,rings,a0,rho0,a0)
        area = read(ROOT/item['area_file'])['current_areas_exact']
        for source in ('kernel','ledger'):
            path = ROOT/item['fields'][source]['file']
            assert digest(path) == item['fields'][source]['sha256']
            with np.load(path,allow_pickle=False) as z: u = z['rho']
            assert u.shape == m.shape == rho0.shape == (N*N,)
            assert np.isfinite(u).all() and np.min(u)>0 and np.min(m)>0
            if source == 'kernel':
                a,counts = bin_density(ids,x[63],m,2/N)
                support = (item['kernel_support'] if N != 64 else next(q['kernel_support']
                    for q in original_receipts if q['source']=='kernel' and q['K']==32 and q['step']==31))
                assert support['h'] == 2/N and not support['normalization'] and support['self_retained']
                assert support['support_count_min'] == min(counts) and support['support_count_max'] == max(counts)
                assert np.max(abs(u-a)) <= 1e-10*max(1.,float(np.max(abs(a))))
            else:
                verify_ring_density(x[63],w[63],rho0,rings,a0,u,area)
                a = np.array([float(F(float(r))*F(initial)/F(current))
                    for r,initial,current in zip(rho0,a0,area)])
                assert np.array_equal(a,u)
            target = out/f'n{N}/{source}-independent.npz'
            assert not target.exists(); np.savez_compressed(target,rho=a)
            rows.append({'N':N,'source':source,'field_file':str(path.relative_to(ROOT)),
                'field_sha256':digest(path),'new_production_field':item['fields'][source]['new_production_field'],
                'independent_file':str(target.relative_to(ROOT)),'independent_sha256':digest(target),
                'maximum_evaluation_difference':float(np.max(abs(u-a))),
                'original_algebra_guard':1e-10,'oracle_or_scalar_loaded':False,
                'exact_ledger_single_rounding_identity':True if source=='ledger' else None})
        save(out/f'n{N}/lift-audit.json',lifts)
        guard(f'N{N} independent lifts, rings and density passed')
        del ids,labels,rho0,m,t,x,w,rings,a0,area,u,a
        gc.collect()
    assert sum(r['new_production_field'] for r in rows) == 4 and len(rows) == 6
    save(out/'density-audit.json',{'passed':True,'rows':rows,
        'elapsed_seconds':time.monotonic()-start,'all_density_checks_preceded_oracle_loading':True,
        'verification_arrays_are_additional_physical_samples':False,
        'new_software_tests':0,'recorded_checks_total':508})


def independent_moments(mass, oracle, u, a):
    m,o,u,a = [list(map(F,v.tolist())) for v in (mass,oracle,u,a)]
    def bilinear(x,y): return sum((w*v*z for w,v,z in zip(m,x,y)),F(0))
    def linear(x): return sum((w*v for w,v in zip(m,x)),F(0))
    # Expanded exact moments provide an independent path without float cancellation.
    oo,uu,aa = bilinear(o,o),bilinear(u,u),bilinear(a,a)
    rel = [v/z-1 for v,z in zip(u,o)]
    diff = [abs(v/z-w/z) for v,w,z in zip(u,a,o)]
    values = {'mass':sum(m,F(0)),'oracle_square':oo,
        'error_square':uu-2*bilinear(u,o)+oo,
        'evaluation_square':uu-2*bilinear(u,a)+aa,
        'signed_error_mass':linear(u)-linear(o),'relative_Linf':max(map(abs,rel)),
        'evaluation_Linf':max(diff),'relative_minimum':min(rel),'relative_maximum':max(rel),
        'density_minimum':min(u),'density_maximum':max(u),'oracle_minimum':min(o),'oracle_maximum':max(o),
        'mass_minimum':min(m),'mass_maximum':max(m)}
    assert values['error_square']>=0 and values['evaluation_square']>=0
    return values


def postseal_audit(out):
    protocol = read(out/'protocol.json'); locked = read(out/'implementation-lock.json')
    for entries in (protocol['sources'],locked['files']):
        for name,h in entries.items(): assert digest(ROOT/name)==h,name
    assert locked['locked_before_fields']
    assert digest(PARENT/'next-gate-spec.json')==protocol['source_gate_sha256']
    gate = read(PARENT/'next-gate-spec.json'); assert protocol['metrics']==gate['metrics']
    seal = read(out/'density-seal.json'); receipts = read(out/'density-audit.json')
    assert receipts['passed'] and receipts['all_density_checks_preceded_oracle_loading']
    assert seal['all_six_fields_independently_verified_before_oracle']
    for name,h in seal['files'].items(): assert digest(ROOT/name)==h,name
    index = read(out/'density-index.json'); assert set(index)=={'64','128','256'}
    assert sum(f['new_production_field'] for item in index.values() for f in item['fields'].values())==4
    assert len(list(out.glob('n*/kernel.npz')))==len(list(out.glob('n*/ledger.npz')))==2
    for receipt in receipts['rows']:
        for file_key,hash_key in (('field_file','field_sha256'),('independent_file','independent_sha256')):
            assert digest(ROOT/receipt[file_key])==receipt[hash_key]
        assert receipt['original_algebra_guard']==1e-10 and not receipt['oracle_or_scalar_loaded']
    access = read(out/'input-access.json'); scoring_started = False; oracle_reads = 0
    allowed = {'trajectory':{'ids','labels','rho0_point'},'mass':{'mass'},
        'builder':{'ids','k32_time','k32_positions'},'oracle':{'k32_rho'}}
    for event in access:
        assert set(event['keys'])<=allowed[event['role']]
        if event['phase']=='postseal_scoring': scoring_started=True
        if event['role']=='oracle':
            oracle_reads+=1
            assert scoring_started and event['density_seal_sha256']==digest(out/'density-seal.json')
        elif event['phase']=='production': assert not scoring_started
    assert oracle_reads==3
    # AST check binds the explicit input boundary and prevents undeclared producers.
    tree = ast.parse((ROOT/'research/reconnect/density_snapshot_study.py').read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    assert imports=={'pathlib','fractions','audit_route_decision','compression_density_sources',
        'density_snapshot_metrics','audit_density_snapshot'}
    saved = read(out/'exact-moments.json'); exact=[]
    assert [(q['N'],q['source']) for q in saved]==[(n,s) for n in (64,128,256) for s in ('kernel','ledger')]
    for case in protocol['cases']:
        n=case['N']
        with np.load(ROOT/case['mass_file'],allow_pickle=False) as z: m=z['mass']
        with np.load(ROOT/case['builder'],allow_pickle=False) as z: o=z['k32_rho'][63]
        for source in ('kernel','ledger'):
            with np.load(ROOT/index[str(n)]['fields'][source]['file'],allow_pickle=False) as z: u=z['rho']
            with np.load(out/f'n{n}/{source}-independent.npz',allow_pickle=False) as z: a=z['rho']
            q=independent_moments(m,o,u,a)
            entry=next(v for v in saved if v['N']==n and v['source']==source)
            assert {k:str(v) for k,v in q.items()}==entry['moments']
            exact.append((n,source,q))
    result=read(out/'results.json'); precision=[]
    for digits in (80,100):
        with mp.workdps(digits):
            def convert(q): return mp.mpf(q.numerator)/q.denominator
            values={}; strings=[]
            for n,source,mom in exact:
                q={k:convert(v) for k,v in mom.items()}
                v={'E2':mp.sqrt(q['error_square']/q['oracle_square']),
                    'Einf':q['relative_Linf'],'B2':mp.sqrt(q['evaluation_square']/q['oracle_square']),
                    'Binf':q['evaluation_Linf'],'signed_mass_bias':q['signed_error_mass']/q['mass']}
                v.update({k:q[k] for k in ('relative_minimum','relative_maximum','density_minimum',
                    'density_maximum','oracle_minimum','oracle_maximum','mass_minimum','mass_maximum')})
                row={'N':n,'source':source,**{k:mp.nstr(x,65) for k,x in v.items()}}
                assert row==next(x for x in result['observations'] if x['N']==n and x['source']==source)
                strings.append(row);values[n,source]=v
            decisions=[]; assert len(result['spatial_pairs'])==8
            assert len({(q['source'],q['coarse_N'],q['fine_N'],q['metric'])
                for q in result['spatial_pairs']})==8
            for row in result['spatial_pairs']:
                source,coarse,fine,metric=row['source'],row['coarse_N'],row['fine_N'],row['metric']
                assert source in ('kernel','ledger') and (coarse,fine) in ((64,128),(128,256))
                c,f=values[coarse,source],values[fine,source]; b={'E2':'B2','Einf':'Binf'}[metric]
                gain=c[metric]-f[metric]; budget=c[b]+f[b]
                decision=('decrease_larger_than_evaluation_indicator' if gain>budget else
                    'increase_larger_than_evaluation_indicator' if gain< -budget else 'numerically_unresolved')
                ratio=c[metric]/f[metric] if f[metric] else None
                expected={'source':source,'coarse_N':coarse,'fine_N':fine,'metric':metric,
                    'gain':mp.nstr(gain,65),'evaluation_indicator_sum':mp.nstr(budget,65),'decision':decision,
                    'coarse_over_fine_error':None if ratio is None else mp.nstr(ratio,65),
                    'observed_log2_ratio':None if ratio is None or ratio<=0 else mp.nstr(mp.log(ratio,2),65)}
                assert row==expected;decisions.append(expected)
            precision.append({'observations':strings,'pairs':decisions})
    assert precision[0]==precision[1]
    all_down=all(x['decision']=='decrease_larger_than_evaluation_indicator' for x in result['spatial_pairs'])
    assert all_down==result['all_pairs_decrease_beyond_evaluation_indicators']
    assert result['decision']==('diagnostic_complete_review_only' if all_down else 'stop_no_expansion')
    assert result['new_density_fields']==4 and result['N64_fields_reused']==2
    for field in ('new_geometry','new_coefficients','new_scalar_paths','new_reference_PDE','new_software_tests'):
        assert result[field]==0
    assert result['recorded_checks_total']==508 and result['stage3']=='incomplete'
    assert not result['fluid_orc_enabled'] and not result['general_order_or_PDE_bound_certified']
    ex=read(out/'execution.json'); assert ex['completed'] and ex['elapsed_seconds']<600 and ex['peak_RSS_MiB']<1024
    preservation=read(out/'preservation.json')
    assert len(preservation['formal_manifests_verified'])==124
    for item in preservation['formal_manifests_verified']:
        assert digest(ROOT/item['manifest'])==item['manifest_sha256']
    assert preservation['originals_verified']==782 and preservation['attachments_verified']==15
    return {'passed':True,'all_six_field_seals_and_input_boundary_verified':True,
        'independent_expanded_exact_moments_verified':6,'precision_digits':[80,100],
        'reported_significant_digits':65,'all_eight_pair_decisions_verified':True,
        'decision':result['decision'],'preseal_density_audit_repeated':False,
        'reason':'preseal independent density verification retained; identical bytes reverified',
        'new_production_fields_in_main':4,'new_software_tests':0,'recorded_checks_total':508,
        'stage3':'incomplete','fluid_orc_enabled':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--output');args=parser.parse_args();start=time.monotonic()
    def alarm(*_): raise TimeoutError('frozen600second independent snapshot audit cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        result=postseal_audit(Path(args.directory).resolve())
        elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed<600 and rss<1024
        result.update(elapsed_seconds=elapsed,peak_RSS_MiB=rss)
        if args.output: save(args.output,result)
        print(result)
    except BaseException as exc:
        if args.output:
            save(Path(args.output).with_name('failed-audit.json'),{'type':type(exc).__name__,
                'message':str(exc),'traceback':traceback.format_exc(),
                'partial_data_preserved':True,'automatic_retry':False})
        raise
    finally: signal.alarm(0)
