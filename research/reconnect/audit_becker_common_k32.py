"""Independent saved sequence/complete material union/80 and100 digit budget audit."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext
import argparse, json, math, resource, time
import numpy as np
from audit_route_decision import ROOT, read, digest
from audit_becker_spatial import save, Q, norm, arrays
from audit_becker_fine_time import audit_actions_k32, verify_action_specialization, variance_exact, squared
from audit_aligned_becker import near
from becker_fine_time_core import verify_sampling_rule
from becker_common_k32_core import analyze, reused_grid
from affine_shear_completion import plain_json


def observations(out,r):
    src = out/'source/old'
    old = read(src/'results.json')
    fine = read(out/'source/fine/results.json')
    assert r['by_grid']['128'] == reused_grid(fine)
    base = arrays(src/'n32/column.npz')
    mp,cp = base['mass'],base['c0']
    common = arrays(out/'common-observations.npz')
    oldcommon = arrays(src/'common-observations.npz')
    reference = arrays(src/'reference/observations-l1-k16384.npz')
    calculated = {}
    for N in (32,64,128):
        row = r['by_grid'][str(N)]
        folder = (out if N != 128 else out/'source/fine')/f'n{N}'
        data,col = arrays(folder/'material.npz'),arrays(src/f'n{N}/column.npz')
        m,c0,mc = data['mass'],data['c0'],col['mass']
        np.testing.assert_array_equal(m,np.tile(mc/8,8))
        np.testing.assert_array_equal(c0,np.tile(col['c0'],8))
        path = arrays(folder/'k32/trajectory-states.npz')
        u,v = path['states'][-1],path['independent_states'][-1]
        u16 = arrays(src/f'n{N}/k16/trajectory-states.npz')['states'][-1]
        pu,p16 = common[f'n{N}_k32'],oldcommon[f'n{N}_k16']
        exact,capacity = [],[]
        ratio = N//32
        for i in range(32):
            ids = [a*N+i*ratio+j for a in range(8) for j in range(ratio)]
            exact.append(sum((Q(m[j])*Q(u[j]) for j in ids),F(0))/Q(mp[i]))
            capacity.append(sum((Q(m[j]) for j in ids),F(0))/Q(mp[i]))
        rounding = np.array([float(Q(a)-b) for a,b in zip(pu,exact)])
        assert np.max(abs(rounding)) <= 64*np.finfo(float).eps*max(1.,float(np.max(abs(u))))
        assert norm(mp,rounding) == row['J_star_parts']['new_K32_common_rounding']
        assert norm(mp,np.array([float(abs(q-1))*float(np.max(abs(u))) for q in capacity])) == row['J_star_parts']['new_K32_capacity_final_bound']
        factor = math.nextafter(math.sqrt(float(max(capacity))),math.inf)
        assert factor == row['capacity']['norm_factor_upper']
        assert float(max(abs(q-1) for q in capacity)) == row['capacity']['defect']
        bridge = norm(mc,(u.reshape(8,N)*(mc/8)).sum(axis=0)/mc-u.reshape(8,N)[0])
        assert bridge == row['B_star_parts']['new_K32_transverse_bridge']
        assert row['B_star_parts'][f'retained_B{N}'] == old['time_controls'][str(N)]['budget_absolute']
        for key,value in old['spatial_budget_by_grid'][str(N)]['common_parts'].items():
            if key not in ('observed8_16','capacity_times_original_budget'):
                assert row['J_star_parts']['retained_'+key] == value
        last = read(folder/'k32/native-results.json')['steps'][-1]
        assert row['numeric_K32_parts']['machine'] == last['cumulative_machine_indicator_absolute']
        assert row['numeric_K32_parts']['tail'] == last['cumulative_tail_indicator_absolute']
        ref = np.tile(reference[f'n{N}_projected'],8)
        hx,hpx = 1-variance_exact(m,u)/variance_exact(m,c0),1-variance_exact(mp,pu)/variance_exact(mp,cp)
        assert abs(float(hx)-row['native_H_K32']) <= 64*np.finfo(float).eps
        assert abs(float(hpx)-row['common_H_K32']) <= 64*np.finfo(float).eps
        for k,val in [('native_H',abs(row['native_H_K32']-old['observations'][str(N)]['native_reference_H'])),
                      ('common_H',abs(row['common_H_K32']-old['observations'][str(N)]['common_reference_H']))]:
            assert row['errors'][k] == val
        calculated[N] = {'squared':{'native_error':squared(m,u,ref),'common_error':squared(mp,pu,reference['n32_projected']),
            'd16_32':squared(m,u16,u),'common_d16_32':squared(mp,p16,pu),'control':squared(m,u,v)},
            'A':2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0)))),
            'V0':float(m@((c0-float(m@c0/m.sum()))**2)),
            'Ap':2*math.sqrt(float(mp.sum()))*max(1.,float(np.max(abs(cp)))),
            'Vp':float(mp@((cp-float(mp@cp/mp.sum()))**2))}
    checks = []
    for digits in (80,100):
        with localcontext() as ctx:
            ctx.prec = digits
            D = lambda x: Decimal.from_float(float(x))
            sigma = D(r['common_initial_fluctuation_norm'])
            gridchecks = []
            for N,z in calculated.items():
                row = r['by_grid'][str(N)]
                roots = {k:(Decimal(q.numerator)/Decimal(q.denominator)).sqrt() for k,q in z['squared'].items()}
                near(roots['native_error']/sigma,row['errors']['native'])
                near(roots['common_error']/sigma,row['errors']['common'])
                near(roots['d16_32'],row['state_time_control']['new16_32'])
                near(roots['common_d16_32'],row['spatial_parts']['common']['observed16_32'])
                near(roots['control'],row['numeric_K32_parts']['complete_sequence_difference'])
                numeric = sum(map(D,row['numeric_K32_parts'].values()),Decimal(0))
                near(2*numeric,row['B_star_parts']['twice_new_path_numeric'])
                B = sum(map(D,row['B_star_parts'].values()),Decimal(0))
                J = D(row['capacity']['norm_factor_upper'])*B+sum(map(D,row['J_star_parts'].values()),Decimal(0))
                near(B,row['B_star']);near(J,row['common_norm_budget'])
                hb,hpb = (D(z['A'])*B+B*B)/D(z['V0']),(D(z['Ap'])*J+J*J)/D(z['Vp'])
                near(hb,row['shifted_H_time_control']['current_original_H_indicator'])
                near(hpb,row['spatial_parts']['common_H']['original_H_indicator'])
                assert all(v >= 0 for v in row['B_star_parts'].values()) and all(v >= 0 for v in row['J_star_parts'].values())
                tc = old['time_controls'][str(N)]
                state = D(tc['state8_16'])-roots['d16_32']
                near(state,row['state_time_control']['decrease'])
                assert (state > B) == row['state_time_control']['resolved']
                hd = abs(D(row['native_H_K32'])-D(old['observations'][str(N)]['native_H']))
                near(hd,row['shifted_H_time_control']['new16_32'])
                dec = D(tc['H8_16'])-hd
                near(dec,row['shifted_H_time_control']['decrease'])
                assert (dec > hb) == row['shifted_H_time_control']['resolved']
                gridchecks.append({'columns':N,'B_star':str(B),'common_norm_budget':str(J),
                    'H_indicator':str(hb),'H_decrease':str(dec),'H_resolved':dec > hb,
                    'exact_saved_norms':{k:str(v) for k,v in roots.items()}})
            comparisons = []
            for item in r['adjacent_spatial_comparisons']:
                C,N = item['coarse_columns'],item['fine_columns']
                for k,parts in item['positive_components'].items():
                    assert parts == {f'n{Q}_{name}':value for Q in (C,N) for name,value in r['by_grid'][str(Q)]['spatial_parts'][k].items()}
                    assert all(v >= 0 for v in parts.values())
                    budget = sum(map(D,parts.values()),Decimal(0))/(sigma if k in ('native','common') else 1)
                    gain = D(r['by_grid'][str(C)]['errors'][k])-D(r['by_grid'][str(N)]['errors'][k])
                    near(budget,item['fixed_indicators'][k]);near(gain,item['gains'][k])
                    decision = 'fine_better_resolvably' if gain > budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
                    assert decision == item['decisions'][k]
                    comparisons.append({'columns':[C,N],'observable':k,'gain':str(gain),'budget':str(budget),'decision':decision})
            checks.append({'digits':digits,'grids':gridchecks,'comparisons':comparisons})
    assert not r['by_grid']['128']['shifted_H_time_control']['resolved']
    return {'retained_old_positive_terms_verified':True,'complete_mass_unions_verified':True,'precision_checks':checks}


def main(directory,write=False):
    started = time.monotonic()
    out = Path(directory)
    r,ex,seal = read(out/'results.json'),read(out/'execution.json'),read(out/'coefficient-seal.json')
    protocol = read(out/'protocol.json')
    assert ex['completed'] and ex['new_graphs'] == 64 and ex['new_primary_paths'] == ex['new_independent_paths'] == 2
    assert ex['elapsed_seconds'] < 900 and ex['peak_RSS_MiB'] < 1024
    assert seal['graphs'] == 64 and seal['all64_sealed_before_mass_scalar_oldstates_reference'] and not seal['internal_material_faces_given_to_builder']
    for p,h in read(out/'implementation-lock.json')['files'].items():
        assert digest(ROOT/p) == h,p
    for item in read(out/'source-copies.json'):
        assert digest(out/item['copy']) == item['sha256'] == protocol['source_files'][item['source']]
    verify_sampling_rule();verify_action_specialization()
    for N in (32,64):
        assert digest(out/f'n{N}/graph-index.json') == seal['index_sha256'][str(N)]
        world = read(out/f'n{N}/pure-inputs.json')
        prior = read(out/f'source/old/n{N}/pure-inputs.json')
        frames = world['cases']['32']['frames']
        assert set(world['cases']) == {'32'} and len(frames) == 65
        assert all(F(f['time_exact']) == F(i,640) for i,f in enumerate(frames))
        assert all(frames[2*i] == f for i,f in enumerate(prior['cases']['16']['frames']))
    actions = [audit_actions_k32(out,N,write) for N in (32,64)]
    budget = observations(out,r)
    current,common = analyze(out)
    assert plain_json(current) == r
    saved = arrays(out/'common-observations.npz')
    for k,v in common.items():
        np.testing.assert_array_equal(v,saved[k])
    assert not r['fluid_orc_enabled'] and not r['new_full_spatial_or_Stage3_pass_allowed'] and not r['registered_H_check_waived']
    findings = {'passed':True,'actions':actions,'observations_and_positive_budgets':budget,
        'frozen_analysis_reproduced':True,'old_source_hashes_verified':True,
        'reused128_budget_and_all_outcomes_unchanged':True,'comparison_decision':r['comparison_decision'],
        'all_shifted_H_outcomes_retained':True,'new_primary_or_reference_solves_in_audit':0,
        'new_software_checks':0,'recorded_checks_total':502}
    if write:
        save(out/'audit.json',{'passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__))})
    else:
        assert read(out/'audit.json')['findings'] == findings
    print(json.dumps({'passed':True,'actions_verified':64,'comparison_decision':r['comparison_decision'],
        'elapsed_seconds':time.monotonic()-started}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True)
    p.add_argument('--write',action='store_true')
    args = p.parse_args()
    main(args.directory,args.write)
