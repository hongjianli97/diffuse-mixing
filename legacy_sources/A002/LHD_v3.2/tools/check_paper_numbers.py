"""Check that the paper quotes the numbers computed from results/*.json (tools/paper_numbers.py), and flag the
strings of known superseded claims.  usage: python tools/check_paper_numbers.py paper.tex"""
import sys
sys.path.insert(0, "tools"); from paper_numbers import numbers
s = open(sys.argv[1]).read(); N = numbers(); bad = 0
skip = {"diff_overshoot_max_sod", "diff_overshoot_max_blast", "diff_ledger_blast", "diff_area_blast", "planar_exact_max", "planar_block_max", "meta_max"}
for k, v in N.items():
    if k in skip: continue
    ok = str(v) in s; bad += not ok; print("%-24s %-28s %s" % (k, v, "ok" if ok else "MISSING"))
for old in ("median errors below $1\\%$", "4--10 times", "agrees with DNS to $1.5\\%$", "less than $0.03$ percentage points", "$0.0421$"):
    hit = old in s; bad += hit; print("superseded %-40s %s" % (old, "STILL PRESENT" if hit else "gone"))
import json, os
if os.path.exists("tools/v4_paper_keys.json") and "v3" in open(sys.argv[1]).read()[:3000]:
    import v4_numbers as V4; N4 = V4.numbers_v32(); N4["m_vel_maxdiff_all"] = "%.0f" % max(float(N4[k]) for k in N4 if k.endswith("_vel_maxdiff"))
    s = s.replace(r"\mbox{--}", "--")
    for k in json.load(open("tools/v4_paper_keys.json")):
        if k not in N4: continue
        ok = str(N4[k]) in s; bad += not ok; print("v4 %-24s %-20s %s" % (k, N4[k], "ok" if ok else "MISSING"))
print("RESULT:", "PASS" if bad == 0 else "%d problems" % bad); sys.exit(bad > 0)
