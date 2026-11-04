"""allcases.py <case> -- the eight cases of the LHD report recomputed with the LTN channel framework (ltn_dyn.LTN),
plus mainstream baselines on the same tracers: DTFE density, FTLE, DBS (least-squares, Haller-Karrasch-Kogelbauer),
encounter number (Rypina & Pratt), LAVD (from the network's face circulations vs exact), AFRC barrier intensity.
cases: dg | lo | sod | blast | sv | sb | sbv | rm | rmv      Reference fields are read only for scoring."""
import numpy as np, sys, json, time, pickle
from scipy.stats import spearmanr
sys.path.insert(0, '/home/claude/w/ltn'); sys.path.insert(0, '/home/claude/w/LHD_v3.2/lhd'); sys.path.insert(0, '/home/claude/w'); sys.path.insert(0, '/home/claude/w/cmp')
import ltn_dyn as LD, lhdx as X, cases as CS, compare as CB
G_ = 1.4; CV = 1 / (G_ - 1); case = sys.argv[1]; t0 = time.time(); ref = {}; mir = {}

# ------------------------------------------------------------------ load
if case == 'dg':
    c = CS.case_dg(); fr, dt, h = c['frames'], c['dt'], c['h']; rho0 = np.ones(fr.shape[1]); ref['om'] = c['om_ex']; ref['lavd'] = c['lavd_ex']; S_eval = [25, 49]
if case == 'lo':
    c = CS.case_lo(); fr, dt, h = c['frames'], c['dt'], c['h']; rho0 = np.ones(fr.shape[1]); ref['lavd'] = c['lavd_ex']; S_eval = [20, 39]
if case == 'sod':
    c = CS.case_sod(S=41); fr, dt, h = c['frames'], c['dt'], c['h']; rho0 = c['rho'][0]; ref.update(rho=c['rho'], p=c['p']); mir = dict(mirror_band=0.12); S_eval = [25, 39]
if case == 'blast':
    c = CS.case_blast(); fr, dt, h = c['frames'], c['dt'], c['h']; ref['lnJ'] = c['lnJ_ex']; rho0 = np.ones(fr.shape[1]); S_eval = [30, 59]
    for k in ('rho', 'p'):
        if k in c: ref[k] = c[k]
    if 'rho' in c: rho0 = c['rho'][0]
if case == 'sv':
    o = pickle.load(open('/home/claude/w/recalc/sv_dns.pkl', 'rb')); fr = np.asarray(o['frames'], float); dt = float(o['dt']); h = 0.005
    rho0 = np.asarray(o['E']['rho'][0], float); ref.update({k: np.asarray(o['E'][k], float) for k in ('rho', 'p', 'om')}); S_eval = [20, 40]
if case in ('sb', 'sbv'):
    if case == 'sb':
        o = pickle.load(open('/home/claude/w/LHD_v3.2/sb_dns_dx0.005.pkl', 'rb')); fr = o['frames']; Ed = o['E']
    else:
        o = pickle.load(open('/home/claude/w/visc/sbv_ck.pkl', 'rb'))['R']; fr = np.array(o['frames']); Ed = o
    dt, h = 0.02, 0.005; rho0 = np.asarray(Ed['rho'][0], float); ref.update({k: np.asarray(Ed[k], float) for k in ('rho', 'p', 'om')}); mir = dict(mirror_y=True); S_eval = [30, 50]
if case in ('rm', 'rmv'):
    o = pickle.load(open({'rm': '/home/claude/w/rm/rm_single_ck.pkl', 'rmv': '/home/claude/w/rm/rmv_single_ck.pkl'}[case], 'rb'))['rec']
    fr = np.array(o['frames']); dt, h = 0.05, 0.005; rho0 = np.array(o['rho'][0]); ref.update(rho=np.array(o['rho']), p=np.array(o['p'])); mir = dict(mirror_band=0.5); S_eval = [40, 80]
fr = np.asarray(fr, float); S, N0 = fr.shape[:2]
G = LD.LTN(fr, dt, rho0, k=12, **mir); N = G.N; org = np.arange(N) < N0
print(case, 'S=%d N0=%d N=%d |E_G|=%d  built %.0fs' % (S, N0, N, len(G.keys), time.time() - t0), flush=True)
res = dict(case=case, S=S, N0=N0, E_G=int(len(G.keys))); fields = {}

def bnodes(c):
    ntri = np.asarray(abs(c['B2']).sum(1)).ravel(); b = np.zeros(N, bool); b[c['E'][ntri == 1].ravel()] = True; return b
def node_avg(c, vT):
    v = np.zeros(N); w = np.zeros(N)
    for k in range(3): np.add.at(v, c['tri'][:, k], vT * c['A']); np.add.at(w, c['tri'][:, k], c['A'])
    return v / np.maximum(w, 1e-300)
def relerr(a, b, m): e = np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-12); return float(np.median(e)), float(np.percentile(e, 95))

for s in S_eval:
    ch = G.channels(s); c = ch['c']; hot = G.bad[:, s]; bn = bnodes(c); inner = org & ~bn & ~hot; r = {'t': float(G.t[s]) if hasattr(G, 't') else s * dt}
    # mass channel: ledger vs DTFE
    rl = G.rho(s); dtfe = 3 * G.m / np.maximum(c['Ast'] * 3, 1e-300)
    if 'rho' in ref:
        rr = np.r_[[ref['rho'][s]] * (N // N0)].ravel() if N > N0 else ref['rho'][s]
        r['density ledger (median, 95%)'] = relerr(rl, rr, inner); r['density DTFE (median, 95%)'] = relerr(dtfe, rr, inner)
    elif 'lnJ' in ref:
        Jl = np.exp(G.lnJ[s]); Jr = np.exp(np.r_[[ref['lnJ'][s]] * (N // N0)].ravel()); r['J ledger (median, 95%)'] = relerr(Jl, Jr, inner)
        r['J DTFE (median, 95%)'] = relerr(c['Ast'] / np.maximum(X._star(G.fr[0], G.K[0]['tri'], N) / 3 * 1, 1e-300) * 1, Jr, inner)
    else:
        r['density ledger rel. dev. from 1 (median, 95%)'] = relerr(rl, np.ones(N), inner); r['density DTFE rel. dev. from 1 (median, 95%)'] = relerr(dtfe, np.ones(N), inner)
    # vorticity channel
    omN = node_avg(c, ch['Gam'] / c['A'])
    if 'om' in ref:
        orf = np.r_[ref['om'][s], -ref['om'][s]] if N == 2 * N0 else ref['om'][s][:N0] if N == N0 else None
        m = inner & np.isfinite(omN)
        r['vorticity corr'] = float(np.corrcoef(omN[:N0][m[:N0]], np.asarray(ref['om'][s])[m[:N0]])[0, 1])
    # momentum channel: Euler consistency, pressure, viscosity
    ip = lambda x, y: float((c['w'] * x * y).sum())
    r['curl share of rho*a (Euler: ~0)'] = ip(ch['Hf']['c'], ch['Hf']['c']) / max(ip(ch['fe'], ch['fe']), 1e-300); r['mu_E'] = ch['mu_E']
    if 'p' in ref:
        pr = np.r_[[ref['p'][s]] * (N // N0)].ravel(); p = ch['p'] + (pr[inner] - ch['p'][inner]).mean()
        r['pressure rms/std (off shock)'] = float(np.sqrt(np.mean((p - pr)[inner] ** 2)) / pr[inner].std())
        r['pressure rms/std (all)'] = float(np.sqrt(np.mean((p - pr)[org & ~bn] ** 2)) / pr[org & ~bn].std())
        if 'rho' in ref:
            se = CV * (np.log(np.maximum(pr, 1e-9)) - G_ * np.log(rr)); sn = CV * (np.log(np.maximum(p, 1e-9)) - G_ * np.log(rl)); mm = inner & np.isfinite(sn)
            r['entropy corr'] = float(np.corrcoef(sn[mm], se[mm])[0, 1]); r['entropy rms / range'] = float(np.sqrt(np.mean((sn - se)[mm] ** 2)) / (np.percentile(se[mm], 99) - np.percentile(se[mm], 1) + 1e-12))
            fields['p_%d' % s] = p[:N0]; fields['s_%d' % s] = sn[:N0]; fields['sref_%d' % s] = se[:N0]
    # strain channel: two activations
    ps = G.pair_strain(s, rmax=2.5 * h * (1.5 if case in ('lo', 'blast') else 1)); Sf = 0.5 * (ch['Lg'] + ch['Lg'].transpose(0, 2, 1))
    Sn = np.stack([node_avg(c, Sf[:, 0, 0]), node_avg(c, Sf[:, 0, 1]), node_avg(c, Sf[:, 1, 1])], 1); m = inner & np.isfinite(ps['S']).all(1)
    r['strain corr (pair vs three-point)'] = float(np.corrcoef(ps['S'][m].ravel(), Sn[m].ravel())[0, 1])
    res['s%d' % s] = r; fields['om_%d' % s] = omN[:N0]; fields['rho_%d' % s] = rl[:N0]; fields['dtfe_%d' % s] = dtfe[:N0]
    fields['P_%d' % s] = G.fr[s][:N0]
    print(s, {k: (('%.3g' % v) if isinstance(v, float) else v) for k, v in r.items()}, '%.0fs' % (time.time() - t0), flush=True)

# ------------------------------------------------------------------ transport structures over the whole window
traj = fr; T = dt * (S - 1)
fields['ftle'] = CB.ftle(traj, T); fields['dbs'] = CB.dbs_ls(traj, np.arange(S) * dt, compressible=True)
fields['enc'] = CB.encounter_number(traj, 1.5 * h).astype(float); fields['afrc'] = np.asarray(G.robust_stretch())[:N0]
lav = np.zeros(N); 
for s in range(1, S - 1):
    cs = G.active(s); V = G.node_kin(s)[0]; om = node_avg(cs, (cs['B2'].T @ G.cochain(cs, V)) / cs['A'])
    wbar = np.average(om[:N0], weights=np.maximum(cs['Ast'][:N0], 1e-12)); lav += np.abs(om - wbar) * dt
fields['lavd_net'] = lav[:N0]
st = {}
if 'lavd' in ref:
    le = np.asarray(ref['lavd'], float); m = np.isfinite(le) & np.isfinite(lav[:N0])
    st['LAVD network vs exact: Spearman'] = float(spearmanr(lav[:N0][m], le[m])[0])
    q = np.percentile(le[m], 90); qn = np.percentile(lav[:N0][m], 90); A_ = le[m] > q; B_ = lav[:N0][m] > qn
    st['LAVD top-10% Jaccard'] = float((A_ & B_).sum() / (A_ | B_).sum()); fields['lavd_ex'] = le
for a, b in (('afrc', 'ftle'), ('afrc', 'dbs'), ('ftle', 'dbs'), ('dbs', 'enc'), ('lavd_net', 'ftle')):
    m = np.isfinite(fields[a]) & np.isfinite(fields[b]); st['Spearman %s~%s' % (a, b)] = float(spearmanr(fields[a][m], fields[b][m])[0])
res['structures'] = st; fields['P0'] = fr[0]; fields['PT'] = fr[-1]
print(st, '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open('/home/claude/w/recalc/res_%s.json' % case, 'w'), indent=1, default=float)
np.savez_compressed('/home/claude/w/recalc/fields_%s.npz' % case, **fields)
print('done %.0fs' % (time.time() - t0))
