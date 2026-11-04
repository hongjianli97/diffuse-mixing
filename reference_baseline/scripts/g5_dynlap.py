"""T7 replacement: weighted FEM dynamic Laplacian (Froyland-Junge 2018 + Froyland-Kwok weights) on the trajectory complex,
mass matrix from the ledger; compare coherent sets with the old Ulam result."""
import numpy as np, json, cases, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from bcom import *
def stiff(P, tri, w=None):
    N = len(P); a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    A = 0.5*((b-a)[:, 0]*(c-a)[:, 1]-(b-a)[:, 1]*(c-a)[:, 0]); ok = np.abs(A) > 1e-14; tri, a, b, c, A = tri[ok], a[ok], b[ok], c[ok], A[ok]
    G = [np.c_[(b-c)[:, 1], -(b-c)[:, 0]], np.c_[(c-a)[:, 1], -(c-a)[:, 0]], np.c_[(a-b)[:, 1], -(a-b)[:, 0]]]   # grad lambda * 2A (rotated edges)
    w = np.ones(len(tri)) if w is None else w[ok]
    I, J, V = [], [], []
    for i in range(3):
        for j in range(3):
            I.append(tri[:, i]); J.append(tri[:, j]); V.append(w*(G[i]*G[j]).sum(1)/(4*np.abs(A)))
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(N, N))
def dynlap(fr, m, rho=None, k=4):
    S, N, _ = fr.shape; Kb = sp.csr_matrix((N, N))
    for s in range(S):
        tri = X.complex_peeled(fr[s])["tri"]; w = None if rho is None else rho[s][tri].mean(1)
        Kb = Kb+stiff(fr[s], tri, w)
    Kb = Kb/S; vals, vecs = eigsh(Kb, k=k, M=sp.diags(m), sigma=-1e-6, which="LM")
    o = np.argsort(vals); return vals[o], vecs[:, o]
def jacc(a, b): return float((a & b).sum()/max((a | b).sum(), 1))
out = {}
c = cases.case_lo(a=1.0, Rd=6.5, N=8000); fr = c["frames"]; m = X.star_area(fr[0])/3.0
vals, V = dynlap(fr, m); P0 = fr[0]; cen = c["centres"]
cores = [np.hypot(*(P0-np.array(cc)).T) < 1.5 for cc in cen]
best = []
for j in range(1, 4):
    v = V[:, j]
    for sgn in [1, -1]:
        thr = np.quantile(sgn*v, 0.9); sel = sgn*v >= thr
        best.append(max(jacc(sel, cores[0]), jacc(sel, cores[1]), jacc(sel, cores[0] | cores[1])))
out["lo"] = dict(eigs=[float(x) for x in vals], best_jaccard_core=float(max(best)), ulam_jaccard_old=0.11)
print("lo", out["lo"], flush=True)
c = cases.case_dg(nx=160, ny=80, T=10.0, S=51); fr = c["frames"]; m = X.star_area(fr[0])/3.0
vals, V = dynlap(fr, m); P0 = fr[0]; left = P0[:, 0] < 1.0
v = V[:, 1]; sel = v > np.median(v); out["dg"] = dict(eigs=[float(x) for x in vals], jaccard_left=float(max(jacc(sel, left), jacc(~sel, left))), ulam_jaccard_old=0.59)
print("dg", out["dg"], flush=True)
json.dump(out, open("res_g5_dynlap.json", "w"), indent=1)
