"""Stage 2 of the shock-bubble case: AFRC_E (paper input / Stokes input) and AFRC_H (paper / coexact weights)."""
import gc, json, pickle, time, numpy as np, lhdx as X, afrc_engine as AF, afrc_h_engine as AH
D = pickle.load(open("sb_dns.pkl", "rb")); fr, t = D["frames"], D["t"]; N = fr.shape[1]; del D; gc.collect()
Z = np.load("sb_stage1.npz"); sc, E, cs, dt = Z["sc"], Z["E"], Z["cs"], float(Z["dt"])
o = {"vort_c": Z["vort_c"]}; R = {}
traj = np.ascontiguousarray(np.transpose(fr, (1, 0, 2)))
t1 = time.time(); BH, _, kf = AH.afrc_h_field(traj, k=16, dt=t[-1] - t[0], censor=cs); R["t_afrcH"] = time.time() - t1
AH._FORMAN_CENSOR_CURRENT = cs
keepE = AH.coherent_edge_keep(traj, E); AH._FORMAN_CENSOR_CURRENT = None
Ek = E[keepE]; sce = np.abs(np.nanmedian(sc[:, keepE], 0)); sce = np.where(np.isfinite(sce), sce, 0.0)
sce = np.maximum(sce, 1e-3 * np.median(sce[sce > 0]))
BH_rb = AH._forman_node_from_weights(N, Ek, sce, AH._node_weight(N, Ek, sce))
BH_rb[np.bincount(Ek.ravel(), minlength=N) < 3] = 0.0
t1 = time.time(); FE = AF.afrc_fields(traj); R["t_afrcE"] = time.time() - t1
orig_turn = AF._frc5_edge_turning
TH = 0.5 * np.nan_to_num(o["vort_c"][:-1] + o["vort_c"][1:]) * 0.5 * dt
def stokes_turning(Pg, Ee, chunk=AF.FRC5_CHUNK_E):
    _, rln = orig_turn(Pg, Ee, chunk)
    return np.ascontiguousarray((0.5 * (TH[:, Ee[:, 0]] + TH[:, Ee[:, 1]])).T.astype(np.float32)), rln
AF._frc5_edge_turning = stokes_turning; FE2 = AF.afrc_fields(traj); AF._frc5_edge_turning = orig_turn

np.savez_compressed("sb_afrc.npz", AFRC_E=FE["core_med"], AFRC_E_stokes=FE2["core_med"], AFRC_H=BH, AFRC_H_rebuilt=BH_rb)
json.dump(R, open("sb_afrc_times.json", "w"))
print("afrc done", R)
