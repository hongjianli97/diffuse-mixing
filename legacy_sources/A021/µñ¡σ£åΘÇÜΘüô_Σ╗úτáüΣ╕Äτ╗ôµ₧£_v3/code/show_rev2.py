import json, sys
for c in sys.argv[1:]:
    r = json.load(open(f"tcx_res_{c}.json")); p = r["P10_persistence"]
    print(c, {k: p.get(k) for k in ["cores_ref","cores_net","cores_net_noise","cores_shrink","cores_shrink_noise","matched_shrink_noise",
          "sustained_rejected_clean","sustained_rejected_noise","shares_clean","vortices_ref","vortices_new","vortices_new_matched","vortices_new_noise","vortices_new_noise_matched","n_smooth_noise"]})
    for b in r["P11_boundary"]:
        o = {}
        for q in ["cdef<=0.02", "lnLd<=0.1", "lnLd<=0.2"]:
            d = b[q]; o[q] = (int(d["net_found"]), int(d["ref_found"]), int(d["noise_found"]), round(d.get("iou_ref", -1), 2), round(d.get("iou_noise", -1), 2), round(d.get("net_lnL", -1), 2), round(d.get("net_area", -1), 4))
        print("   ", [round(x, 3) for x in b["pos"]], o)
