import pickle, numpy as np, cases, lhdx as X
from tcx_cases import _lo_omega, phi_reference
def load(name):
    if name == "DG":
        c = cases.case_dg(); fr = c["frames"]; P0 = fr[0]
        inner = (P0[:, 0] > .05) & (P0[:, 0] < 1.95) & (P0[:, 1] > .05) & (P0[:, 1] < .95)
        return dict(fr=fr, t=c["t"], dt=c["dt"], h=c["h"], om=c["om_ex"], inner=inner)
    if name == "LO":
        c = cases.case_lo(a=1.0, Rd=6.5, N=8000); fr = c["frames"]; P0 = fr[0]
        return dict(fr=fr, t=c["t"], dt=c["dt"], h=c["h"], om=_lo_omega(fr, c["t"]), inner=np.hypot(P0[:, 0], P0[:, 1]) < 6.0)
    if name == "SOD":
        c = cases.case_sod(); fr = c["frames"]; P0 = fr[0]
        return dict(fr=fr, t=c["t"], dt=c["dt"], h=c["h"], om=np.zeros(fr.shape[:2]), inner=c["interior"](P0) & (np.abs(P0[:, 0]) < 0.45))
    if name == "BLAST":
        c = cases.case_blast(); fr = c["frames"]; P0 = fr[0]
        return dict(fr=fr, t=c["t"], dt=c["dt"], h=c["h"], om=np.zeros(fr.shape[:2]), inner=c["interior"](P0))
    if name == "SB":
        D = pickle.load(open("sb_dns.pkl", "rb")); fr = D["frames"]; P0 = fr[0]
        inner = (P0[:, 0] > 0.32) & (P0[:, 0] < 1.23) & (P0[:, 1] < 0.40)
        return dict(fr=fr, t=D["t"], dt=D["dt"], h=D["h"], om=D["E"]["om"].astype(float), inner=inner)
    if name == "SBlate":                                   # window [0.8, 1.4]: after the roll-up has formed
        D = pickle.load(open("sb_dns.pkl", "rb")); fr = D["frames"][40:71]; P0 = fr[0]
        inner = (P0[:, 0] > 0.45) & (P0[:, 0] < 1.6) & (P0[:, 1] < 0.45)
        return dict(fr=fr, t=D["t"][40:71], dt=D["dt"], h=D["h"], om=D["E"]["om"][40:71].astype(float), inner=inner)
    if name == "SV":
        D = pickle.load(open("sv_dns.pkl", "rb")); k = 31; fr = D["fr_in"][:k]
        return dict(fr=fr, t=D["t"][:k], dt=D["dt"], h=0.005, om=D["Ev"]["om"][:k].astype(float), inner=np.ones(len(fr[0]), bool))
