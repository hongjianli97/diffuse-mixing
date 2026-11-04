"""AFRC_H with the paper's post-processing (part2 cells 19/29): per-scale q95 normalisation, max-combine,
graph soft-closing despeckle (k=8, 2 iters, soft-erode q=0.3), graph-TV (lambda=0.1, 200 it, k=16)."""
import json, pickle, time, numpy as np, cases, lhdx as X, afrc_h_engine as AH, paper_post as PP
def afrcH_paper(traj, censor, T, P, ks=(12, 16, 24), post=True):
    per = []
    for k in ks:
        B, _, _ = AH.afrc_h_field(traj, k=k, dt=T, censor=censor)
        B = np.maximum(B - np.median(B), 0.0)                      # display stretch: remove the static background
        pos = B[B > 0]; per.append(B / (np.quantile(pos, 0.95) if pos.size else 1.0))
    c = np.max(np.vstack(per), 0)
    if not post:
        return c
    c = np.maximum(PP.graph_despeckle(c.astype(np.float64), P, verbose=False), 0.0)
    c = np.maximum(PP.graph_tv_denoise(c.astype(np.float64), P, lam=PP.FORMAN_TV_LAMBDA, iters=PP.FORMAN_TV_ITERS, k=PP.FORMAN_TV_K, verbose=False), 0.0)
    return c
if __name__ == "__main__":
    out = {}; t0 = time.time()
    c = cases.case_dg(); fr = c['frames']; traj = np.ascontiguousarray(np.transpose(fr, (1, 0, 2))); T = c['t'][-1]
    P0 = fr[0]; inner = (P0[:, 0] > .05) & (P0[:, 0] < 1.95) & (P0[:, 1] > .05) & (P0[:, 1] < .95)
    raw = afrcH_paper(traj, None, T, P0, post=False); pst = afrcH_paper(traj, None, T, P0)
    out['DG'] = dict(P=P0, raw=raw, post=pst, ftle=c['ftle_ex'])
    print('DG sp raw %.3f post %.3f  (%.0fs)' % (X.spearman(raw, c['ftle_ex'], inner), X.spearman(pst, c['ftle_ex'], inner), time.time() - t0), flush=True)
    D = pickle.load(open('sb_dns.pkl', 'rb')); fr = D['frames']; Ed = D['E']; dlp = np.diff(np.log(Ed['p']), axis=0); del D
    cst = (dlp > 0.15).T; cs = cst.copy(); cs[:, 1:] |= cst[:, :-1]; cs[:, :-1] |= cst[:, 1:]
    trf = np.ascontiguousarray(np.transpose(fr[50:71], (1, 0, 2))); trb = np.ascontiguousarray(np.transpose(fr[20:51][::-1], (1, 0, 2)))
    Ps = fr[50]
    out['SB'] = dict(P=Ps, fwd=afrcH_paper(trf, cs[:, 50:70], 0.4, Ps), bwd=afrcH_paper(trb, cs[:, 20:50][:, ::-1], 0.6, Ps),
                     ftle_f=X.ls_ftle(fr[50], fr[70], 0.4), ftle_b=X.ls_ftle(fr[50], fr[20], 0.6))
    print('SB done (%.0fs)' % (time.time() - t0), flush=True)
    S = pickle.load(open('sv_dns.pkl', 'rb'))
    for mode in ('in', 'off'):
        fr = S['fr_' + mode][10:31]; traj = np.ascontiguousarray(np.transpose(fr, (1, 0, 2)))
        out['SV_' + mode] = dict(P=fr[-1], bwd=afrcH_paper(np.ascontiguousarray(traj[:, ::-1]), None, 0.4, fr[-1]),
                                 ftle_b=X.ls_ftle(fr[-1], fr[0], 0.4))
    print('SV done (%.0fs)' % (time.time() - t0), flush=True)
    pickle.dump(out, open('res_afrcH_paper.pkl', 'wb'))
