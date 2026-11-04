import numpy as np, json
from lhdp import *
rng = np.random.default_rng(0); out = {}
P0 = rng.uniform(-1, 1, (400, 2)); cx = X.complex_peeled(P0); tri = cx["tri"]
errs_c, errs_f = [], []
for trial in range(200):
    F = np.eye(2)+rng.normal(0, 0.6, (2, 2))
    if np.linalg.det(F) <= 0.05: continue
    c = rng.normal(0, 1, 2); P1 = P0@F.T+c
    circ, flux = disp_cochain_tri(P0, P1, tri); Fe, A0 = tri_F(P0, P1, tri); p, q = pq(Fe); A1 = tri_area(P1, tri)
    errs_c.append(np.max(np.abs(circ-2*A0*p.imag))/np.max(np.abs(A0)))
    errs_f.append(np.max(np.abs(flux-(A1-A0)))/np.max(np.abs(A0)))
out["thm_circ_maxrel"] = float(max(errs_c)); out["thm_flux_maxrel"] = float(max(errs_f))
P1 = P0+0.015*rng.normal(size=P0.shape)
circ, flux = disp_cochain_tri(P0, P1, tri); Fe, A0 = tri_F(P0, P1, tri); p, q = pq(Fe); A1 = tri_area(P1, tri)
ok = A1 > 0
out["nonaffine_circ"] = float(np.max(np.abs(circ-2*A0*p.imag)[ok])/np.max(np.abs(A0))); out["nonaffine_flux"] = float(np.max(np.abs(flux-(A1-A0)))/np.max(np.abs(A0)))
Pm = P0+0.05*rng.normal(size=P0.shape); Pp = P0+0.05*rng.normal(size=P0.shape)
d, dg, Gp, Gm = kelvin_pos(Pm, P0, Pp, tri, "q0"); d2, dg2, _, _ = kelvin_pos(Pm, P0, Pp, tri, "quarter")
sc = np.max(np.abs(d))
out["kelvin_q0_maxrel"] = float(np.max(np.abs(d-dg))/sc); out["kelvin_quarter_maxrel"] = float(np.max(np.abs(d2-dg2))/sc)
res = []
for dt in [0.2, 0.1, 0.05, 0.025]:
    traj = lambda t: np.c_[P0[:, 0]+0.3*np.sin(2*P0[:, 1]+t), P0[:, 1]+0.3*np.sin(2*P0[:, 0]+1.3*t)]
    Pm, Pc, Pp = traj(-dt), traj(0), traj(dt)
    d, dg, Gp, _ = kelvin_pos(Pm, Pc, Pp, tri, "q0"); d2, dg2, _, _ = kelvin_pos(Pm, Pc, Pp, tri, "quarter")
    res.append([dt, float(np.max(np.abs(d-dg))/np.max(np.abs(d))), float(np.max(np.abs(d2-dg2))/np.max(np.abs(d)))])
out["kelvin_dt_scan"] = res
F = np.eye(2)+rng.normal(0, .5, (2, 2)); b0, b1 = 0.3, 1.1
R = lambda b: np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
Fn = R(b1).T@F@R(b0); p, q = pq(F[None]); pn, qn = pq(Fn[None])
out["frame_p"] = float(abs(pn[0]-np.exp(-1j*(b1-b0))*p[0])); out["frame_q"] = float(abs(qn[0]-np.exp(-1j*(b1+b0))*q[0]))
errs = []
for _ in range(2000):
    F = np.eye(2)+rng.normal(0, 1, (2, 2))
    if np.linalg.det(F) <= 0.01: continue
    th = rng.uniform(0, np.pi); t0 = np.array([np.cos(th), np.sin(th)]); n0 = np.array([-t0[1], t0[0]])
    a = F@t0; b = F@n0; lam = np.linalg.norm(a); J = np.linalg.det(F)
    cot = (a@b)/J          # s/rho
    lhs = np.log(np.linalg.norm(b)); rhs = np.log(J)-np.log(lam)+0.5*np.log(1+cot**2)
    errs.append(abs(lhs-rhs))
out["qr_identity_max"] = float(max(errs))
sig = []
for e in np.linspace(1.05, 6, 30):
    th = rng.uniform(0, np.pi); n = np.array([np.cos(th), np.sin(th)]); F = np.eye(2)-(1-1/e)*np.outer(n, n)
    p, q = pq(F[None]); sig.append(abs(abs(q[0]/p[0])-np.tanh(0.5*abs(np.log(np.linalg.det(F))))))
out["uniaxial_sig_max"] = float(max(sig))
# Kabsch pullback objectivity of step readouts
Pa = P0; Pb = P0+0.05*np.c_[np.sin(3*P0[:, 1]), np.cos(2*P0[:, 0])]
Qa, Qb = R(0.7), R(1.9); Pa2 = Pa@Qa.T+[1, 2]; Pb2 = Pb@Qb.T+[-3, 0.5]
y1, _ = pullback(Pa, Pb); y2, _ = pullback(Pa2, Pb2)
r1 = step_readouts(Pa, y1, tri); r2 = step_readouts(Pa2, y2, tri)
out["kabsch_obj_psi"] = float(np.max(np.abs(r1["psi"]-r2["psi"]))); out["kabsch_obj_mu"] = float(np.max(np.abs(r1["mu"]-r2["mu"])))
r1 = step_readouts(Pa, Pb, tri); r2 = step_readouts(Pa2, Pb2, tri)
out["raw_psi_shift"] = float(np.median(r2["psi"]-r1["psi"]))
json.dump(out, open("res_n0.json", "w"), indent=1); print(json.dumps(out, indent=1))
