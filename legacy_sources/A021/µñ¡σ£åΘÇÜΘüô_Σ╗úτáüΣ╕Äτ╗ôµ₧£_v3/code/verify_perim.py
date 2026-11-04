"""Is the network perimeter growth real?  Advect the t0 boundary polygons densely with the exact velocity."""
import pickle, json, numpy as np, cases, cases_mix as CM, afe_cases as AC
def dens(Q, n):
    Qc = np.vstack([Q, Q[:1]]); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Qc, axis=0), axis=1))]
    u = np.linspace(0, s[-1], n, endpoint=False); return np.c_[np.interp(u, s, Qc[:, 0]), np.interp(u, s, Qc[:, 1])]
def perim(Q): return float(np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1).sum())
def adv(vel, Q, t, frames, nsub):
    out = {0: perim(Q)}; tt = t[0]
    for s in range(1, max(frames) + 1):
        hh = (t[s] - t[s - 1]) / nsub
        for _ in range(nsub):
            k1 = vel(Q, tt); k2 = vel(Q + .5*hh*k1, tt + .5*hh); k3 = vel(Q + .5*hh*k2, tt + .5*hh); k4 = vel(Q + hh*k3, tt + hh)
            Q = Q + hh*(k1 + 2*k2 + 2*k3 + k4)/6; tt += hh
        if s in frames: out[s] = perim(Q)
    return max(np.log(out[s]/out[0]) for s in out)
def _main():
    res = {}
    if __name__ == "__main__":
      pass
    
    for nm, vel in [("LO", CM.lo_vel), ("DG", cases.dg_fields()[0])]:
        F = pickle.load(open(f"tcx_fields_{nm}.pkl", "rb")); R = json.load(open(f"tcx_res_{nm}.json")); c = AC.load(nm)
        S = len(c["t"]); frames = set(np.unique(np.linspace(0, S-1, 11).round().astype(int)).tolist())
        for k in range(2):
            for q in ("cdef<=0.02", "lnLd<=0.1", "lnLd<=0.2"):
                Q = F["P11_polys"][k][q].get("net")
                if Q is None: continue
                ex = [adv(vel, dens(Q, n), c["t"], frames, 10) for n in (2000, 8000)]
                res[f"{nm} core{k} {q}"] = dict(network=R["P11_boundary"][k][q]["net_lnL"], exact_2k=ex[0], exact_8k=ex[1])
                print(nm, k, q, {a: round(b, 3) for a, b in res[f"{nm} core{k} {q}"].items()}, flush=True)
    json.dump(res, open("res_verify_perim.json", "w"), indent=1)
    
if __name__ == "__main__":
    _main()
