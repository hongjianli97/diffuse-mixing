"""One nested clock and separately registered K64 joint comparison; no repair."""
import math
from pathlib import Path
import numpy as np
from audit_route_decision import ROOT, read
from compression_voronoi_core import sample_points
from compression_density_sources import periodic_winding
from compression_density_compare import classification, reference_parts
from voronoi_time_action import metrics
from shear_union64_study import arrays

ORIGINAL = ROOT/'runs/reconnect-compression-union64'
CALIBRATION = ROOT/'runs/reconnect-compression-density-calibration'


def nested_samples(ids, labels, rho0, old_time, old_positions):
    """Only new odd frames are sampled; old even frames are copied verbatim."""
    if len(old_time) < 3 or len(old_time) % 2 != 1:
        raise ValueError('complete old half-step clock required')
    if not np.array_equal(old_time, np.linspace(0., .05, len(old_time))):
        raise ValueError('unchanged saved clock required')
    if old_positions.shape != (len(old_time), len(ids), 2):
        raise ValueError('complete saved frames required')
    t = np.linspace(0., .05, 2*len(old_time)-1)
    np.testing.assert_array_equal(t[::2], old_time)
    q = sample_points(labels, ids, rho0, t[1::2])
    x = np.empty((len(t), len(ids), 2)); x[::2] = old_positions; x[1::2] = q['positions']
    w = periodic_winding(ids, t, x, t)
    return {'ids': ids.copy(), 'time': t, 'positions': x,
            'midpoint_rho_oracle': q['rho'], 'winding': w}


def save_npz(path, **items):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **items)


def snapshots(out):
    states = {}; independent = {}; mt = {}
    for s in ('oracle', 'ledger'):
        for K in (16, 32, 64):
            folder = (out/s if K == 64 else ORIGINAL if s == 'oracle' else CALIBRATION/s)/f'k{K}'
            p = arrays(folder/'trajectory-states.npz'); r = read(folder/'native-results.json')
            states[s,K] = p['states'][::K//2]
            independent[s,K] = p['independent_states'][::K//2]
            mt[s,K] = [0.] + [r['steps'][j-1]['cumulative_machine_indicator_absolute'] +
                               r['steps'][j-1]['cumulative_tail_indicator_absolute'] for j in (K//2, K)]
    return states, independent, mt


def compare(data, states, independent, mt):
    m, c0 = data['mass'], data['c0']; V0 = metrics(m,c0,c0)['variance']; sigma = math.sqrt(V0)
    norm = lambda v: float(np.sqrt(m@(v*v)))
    H = lambda v: metrics(m,v,c0)['H']
    hm = lambda e: (2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*e+e*e)/V0
    numerical = {}; observations = {}; time_rows = []
    for s in ('oracle','ledger'):
        observations[s] = []
        for k,t in enumerate((0.,.025,.05)):
            for K in (16,32,64):
                u,v = states[s,K][k],independent[s,K][k]
                numerical[s,K,k] = (mt[s,K][k],norm(u-v),abs(H(u)-H(v)))
            u = states[s,64][k]; ref = data['reference'][k]
            observations[s].append({'time':t,'concentration_error_absolute':norm(u-ref),
                'concentration_error_relative':norm(u-ref)/sigma,'H':H(u),'reference_H':H(ref),
                'H_error':abs(H(u)-H(ref)),'raw_mass_change':float(m@(u-c0)),
                'minimum':float(u.min()),'maximum':float(u.max())})
            if not k: continue
            u,v,w = (states[s,K][k] for K in (16,32,64))
            C = [norm(u-v),norm(v-w)]; HH = [abs(H(u)-H(v)),abs(H(v)-H(w))]
            e = sum(weight*sum(numerical[s,K,k][:2]) for K,weight in ((16,1),(32,2),(64,1)))
            a = sum(weight*numerical[s,K,k][2] for K,weight in ((16,1),(32,2),(64,1)))
            total = sum(weight*numerical[s,K,k][0] for K,weight in ((16,1),(32,2),(64,1)))
            eh = a+hm(total)
            time_rows.append({'source':s,'time':t,'state_difference_16_32':C[0],'state_difference_32_64':C[1],
                'H_difference_16_32':HH[0],'H_difference_32_64':HH[1],
                'concentration_decrease_indicator':e,'H_decrease_indicator':eh,
                'concentration_decrease_resolved':C[0]-C[1]>e,'H_decrease_resolved':HH[0]-HH[1]>eh,
                'middle_K32_weight':2,'time_order_certified':False})
    rows = []
    for k,t in enumerate((0.,.025,.05)):
        if not k: continue
        cp,hp = reference_parts(data,k,norm,H,c0); total = 0.
        for s in ('ledger','oracle'):
            a,b = states[s,32][k],states[s,64][k]
            cp[s+'_32_64'] = norm(a-b); hp[s+'_32_64'] = abs(H(a)-H(b))
            for K in (32,64):
                machine,delta,dh = numerical[s,K,k]
                cp[f'{s}_e{K}'] = machine+delta; hp[f'{s}_complete_sequence_H{K}'] = dh; total += machine
        hp['both_machine_tail_to_H'] = hm(total)
        assert all(v>=0 for v in (*cp.values(),*hp.values()))
        bc,bh = sum(cp.values())/sigma,sum(hp.values())
        a,b = observations['ledger'][k],observations['oracle'][k]
        gc,gh = a['concentration_error_relative']-b['concentration_error_relative'],a['H_error']-b['H_error']
        rows.append({'source':'ledger','time':t,'concentration_parts_absolute':cp,'H_parts':hp,
            'concentration_indicator':bc,'H_indicator':bh,'concentration_error_change':gc,'H_error_change':gh,
            'concentration_decision':classification(gc,bc),'H_decision':classification(gh,bh),
            'actual_source_state_difference_absolute':norm(states['ledger',64][k]-states['oracle',64][k])})
    stop = any(not r[c] for r in time_rows for c in ('concentration_decrease_resolved','H_decrease_resolved'))
    stop |= any(r[c]!='error_smaller_resolvably' for r in rows for c in ('concentration_decision','H_decision'))
    return {'case':'K64_with_K32_control_and_n2048_k2048_reference','common_sigma64':sigma,
        'observations':observations,'time_differences':time_rows,'source_comparisons':rows,
        'stop_expansion':stop,'old_rows_rejudged':False,'time_order_certified':False,
        'indicators_are_certified_PDE_error_bounds':False,'recorded_checks_total':508,
        'stage3':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
