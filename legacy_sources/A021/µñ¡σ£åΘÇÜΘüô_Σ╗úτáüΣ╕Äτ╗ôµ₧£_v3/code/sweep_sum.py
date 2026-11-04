import json, numpy as np
C = ["LO", "DG", "SV", "SB"]; NS = [2, 3, 4, 6, 8, 12, 16]; out = {}
rows = []
for c in C:
    R = json.load(open(f"sweep_{c}.json"))
    for v in R["vortices"]:
        for d, x in v["by_delta"].items():
            rows.append(dict(case=c, pos=v["pos"], delta=float(d), r=x["req_over_h"], clean=x["iou_clean"], noise={int(n): float(np.mean(y)) for n, y in x["iou_noise"].items()}))
def w(n): return 0.55*np.sqrt(n)        # kernel rms width / h (measured: 0.79h at n=2 ... 2.2h at n=16)
rules = {"fixed n=2": lambda r: 2, "fixed n=8 (noise rule)": lambda r: 8,
         "cap 4(r/3h)^2 (rev2)": lambda r: min(8, max(2, int(4*(r/3)**2))),
         "cap w<=r/3": lambda r: min(8, max(2, int((r/(3*0.55))**2))),
         "cap w<=r/4": lambda r: min(8, max(2, int((r/(4*0.55))**2))),
         "cap w<=r/6": lambda r: min(8, max(2, int((r/(6*0.55))**2)))}
def nearest(n): return min(NS, key=lambda k: abs(k-n))
res = {}
for d in (0.1, 0.2):
    sel = [x for x in rows if x["delta"] == d]
    res[d] = {name: round(float(np.mean([x["noise"][nearest(f(x["r"]))] for x in sel])), 3) for name, f in rules.items()}
    res[d]["oracle (best n per vortex)"] = round(float(np.mean([max(x["noise"].values()) for x in sel])), 3)
    res[d]["n_vortices"] = len(sel)
    print("delta", d, res[d])
    for x in sel:
        print("   ", x["case"], x["pos"], "r/h=%.1f" % x["r"], "best n", max(x["noise"], key=x["noise"].get), {k: round(v, 2) for k, v in x["noise"].items()})
# delta sensitivity table: per vortex r/h, clean IoU, noise IoU at n=8
dsum = {}
for x in rows:
    dsum.setdefault(f"{x['case']} {x['pos']}", {})[x["delta"]] = (round(x["r"], 1), round(x["clean"], 2), round(x["noise"][8], 2))
for k, v in dsum.items(): print(k, v)
json.dump(dict(rules=res, delta=dsum), open("sweep_summary.json", "w"), indent=1)
