import pickle, numpy as np
L = lambda g: pickle.load(open("/tmp/work/run/out_%s.pkl" % g, "rb"))
D, LO, SO, BL, SB, SV, TOY = (L(g) for g in ("dg", "lo", "sod", "blast", "sb", "sv", "toy"))
f = lambda x: ("%.3f" % x).replace("-", "$-$") if np.isfinite(x) else "--"
out = []

# ---------------- main summary table: interface-mean ln lambda
rows = []
def row(case, O, s, refkey, old):
    r = O["base"]["res"][s]; m = r["mean"]; ref = r[refkey]
    rows.append("%s & %.2f & %s & %s & %s & %s & \\textbf{%s} & %s & %s \\\\" % (
        case, r["t"], f(ref), old, f(m["pair"]), f(m["poly"]), f(m["vector"]), f(m["covector"]), f(m["combined"])))
row("双环流", D, 25, "exact_mean", "0.781"); row("双环流", D, 50, "exact_mean", "1.700")
row("LO 涡对", LO, 10, "exact_mean", "$-$0.06"); row("LO 涡对", LO, 20, "exact_mean", "0.23"); row("LO 涡对", LO, 40, "exact_mean", "0.78")
row("Sod", SO, 40, "exact_mean", "$-$0.008"); row("爆炸波", BL, 60, "exact_mean", "0.553")
row("激波–气泡", SB, 15, "ref_mean", "$-$0.07"); row("激波–气泡", SB, 30, "ref_mean", "[$-$0.08, 0.46]")
row("激波–气泡", SB, 50, "ref_mean", "[0.20, 0.77]"); row("激波–气泡", SB, 70, "ref_mean", "[0.60, 1.05]")
row("激波–涡（在线）", SV, 30, "ref_mean", "$-$0.18")
r = SV["off"]["res"][30]; m = r["mean"]
rows.append("激波–涡（离线） & %.2f & %s & $-$0.21 & %s & %s & \\textbf{%s} & %s & %s \\\\" % (r["t"], f(r["ref_mean"]), f(m["pair"]), f(m["poly"]), f(m["vector"]), f(m["covector"]), f(m["combined"])))
out.append("%% TABLE main\n" + "\n".join(rows))

# ---------------- per-tracer median abs error (analytic)
rows = []
for case, O, s in (("双环流", D, 25), ("双环流", D, 50), ("LO 涡对", LO, 20), ("LO 涡对", LO, 40), ("Sod", SO, 40), ("爆炸波", BL, 15), ("爆炸波", BL, 60)):
    r = O["base"]["res"][s]; e = r["medabs"]
    rows.append("%s & %.2f & %s & %s & \\textbf{%s} & %s & %s \\\\" % (case, r["t"], f(e["pair"]), f(e["poly"]), f(e["vector"]), f(e["covector"]), f(e["combined"])))
out.append("%% TABLE medabs\n" + "\n".join(rows))

# ---------------- convergence
rows = []
for case, items, s in (("双环流 $t=10$", [("80$\\times$40", D["nx80"]), ("160$\\times$80", D["base"]), ("320$\\times$160", D["nx320"])], 50),
                       ("LO $t=18.8$", [("$N$=2000", LO["N2000"]), ("$N$=8000", LO["base"]), ("$N$=32000", LO["N32000"])], 40),
                       ("爆炸波 $t=25$", [("$h$=0.044", BL["h0.044"]), ("$h$=0.022", BL["base"]), ("$h$=0.011", BL["h0.011"])], 60),
                       ("爆炸波 $t=7$（激波在环上）", [("$h$=0.044", BL["h0.044"]), ("$h$=0.022", BL["base"]), ("$h$=0.011", BL["h0.011"])], 15)):
    for lab, O in items:
        r = O["res"][s]; m = r["mean"]; e = r["medabs"]
        rows.append("%s & %s & %s & %s / %s & \\textbf{%s} / %s & %s / %s \\\\" % (case, lab, f(r["exact_mean"]), f(m["pair"]), f(e["pair"]), f(m["vector"]), f(e["vector"]), f(m["covector"]), f(e["covector"])))
        case = ""
out.append("%% TABLE conv\n" + "\n".join(rows))

# ---------------- noise
rows = []
for case, O, s in (("双环流 $t=10$", D, 50), ("LO $t=18.8$", LO, 40)):
    for key, lab in (("base", "0"), ("noise0.05", "0.05$h$"), ("noise0.10", "0.1$h$")):
        r = O[key]["res"][s]; m = r["mean"]; e = r["medabs"]
        rows.append("%s & %s & %s & %s / %s & \\textbf{%s} / %s & %s / %s & %s / %s & %s \\\\" % (case, lab, f(r["exact_mean"]), f(m["pair"]), f(e["pair"]), f(m["vector"]), f(e["vector"]),
                                                                         f(m["covector"]), f(e["covector"]), f(m["combined"]), f(e["combined"]), f(m["comp"])))
        case = ""
for key, lab in (("base", "0"), ("noise0.05", "0.05$h$"), ("noise0.10", "0.1$h$")):
    r = SB[key]["res"][70]; m = r["mean"]
    rows.append("%s & %s & %s & %s & \\textbf{%s} & %s & %s & %s \\\\" % ("激波–气泡 $t=1.4$" if key == "base" else "", lab, f(r["ref_mean"]), f(m["pair"]), f(m["vector"]), f(m["covector"]), f(m["combined"]), f(m["comp"])))
out.append("%% TABLE noise\n" + "\n".join(rows))

# ---------------- SB attribution
rows = []
for s, old_s, old_c in ((15, "$-$0.07", "0.32"), (30, "0.46 / $-$0.08", "0.48"), (50, "0.77 / 0.20", "0.49"), (70, "1.05 / 0.60", "0.45")):
    r = SB["base"]["res"][s]; m = r["mean"]; rm = SB["base_mass"]["res"][s]["mean"]
    side = [a for a, b in r["ref_side"]]
    tot = m["comp"] + m["vector"]
    rows.append("%.1f & %s & %s & %s & %s [%s, %s] & %s & %s & %s & %s \\\\" % (r["t"], old_c, old_s, f(m["comp"]), f(r["ref_mean"]), f(min(side)), f(max(side)), f(m["vector"]), f(rm["vector"]), f(tot), f(r["ref_total"])))
out.append("%% TABLE sbattr\n" + "\n".join(rows))

# ---------------- SB compression check
chk = SB["comp_check"]
rows = []
for s in (30, 50, 70):
    rows.append("%.1f & %s & %s & %s & %s \\\\" % (s * 0.02, f(chk["heavy"][s][0]), f(chk["heavy"][s][1]), f(chk["light"][s][0]), f(chk["light"][s][1])))
out.append("%% TABLE compchk\n" + "\n".join(rows))

# ---------------- toy
rows = []
for key, lab in (("A_h0.05_n0", "$h$=0.05"), ("A_h0.025_n0", "$h$=0.025"), ("A_h0.05_n0.05", "$h$=0.05，噪声 0.05$h$"), ("A_h0.05_n0.1", "$h$=0.05，噪声 0.1$h$")):
    v = TOY[key]; fr = sorted(v["res"]); gt = v["gt"]
    for target in (15, 30):
        s = min(fr, key=lambda q: abs(gt[q] - target)); m = v["res"][s]["mean"]
        rows.append("同心圆环 %s & $\\dot\\gamma t$=%d & 0 & %s & %s & %s & %s \\\\" % (lab, target, f(m["pair"]), f(m["vector"]), f(m["covector"]), f(m["combined"])))
for key, lab in (("B_h0.04", "$h$=0.04"), ("B_h0.02", "$h$=0.02")):
    v = TOY[key]
    for s in sorted(v["res"]):
        if v["res"][s]["t"] in (10.0, 40.0, 80.0):
            m = v["res"][s]["mean"]
            rows.append("偏心螺旋 %s & $t$=%d & %s & %s & %s & %s & %s \\\\" % (lab, v["res"][s]["t"], f(v["res"][s]["exact_mean"]), f(m["pair"]), f(m["vector"]), f(m["covector"]), f(m["combined"])))
out.append("%% TABLE toy\n" + "\n".join(rows))

# ---------------- compression summary + SV
rows = []
for case, O, s in (("Sod", SO, 40), ("爆炸波 $t=25$", BL, 60), ("爆炸波 $t=7$", BL, 15)):
    r = O["base"]["res"][s]; rows.append("%s & %s & %s \\\\" % (case, f(r["mean"]["comp"]), f(r["comp_ref"])))
for case, O in (("激波–涡（在线）$t=0.6$", SV["base"]), ("激波–涡（离线）$t=0.6$", SV["off"])):
    r = O["res"][30]; rows.append("%s & %s & %s \\\\" % (case, f(r["mean"]["comp"]), f(r["comp_ref"])))
out.append("%% TABLE comp\n" + "\n".join(rows))
open("/tmp/work/report/tables_auto.tex", "w").write("\n\n".join(out))
print("\n\n".join(out))
