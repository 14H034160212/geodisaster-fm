"""Fast offline: re-compare decision methods on cached per-building features,
adding the one-sided 'attractive' SDI variant. No model retraining.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, ".")
from geodisaster.dispatch.structured_decision import (
    SDIConfig, infer_affected, baseline_raw_threshold, baseline_any_intersection,
    baseline_prob_threshold, prf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/xbd_damage_sdi/features.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    feats = json.loads(open(args.features).read())["features"]
    by_chip = defaultdict(list)
    for f in feats:
        by_chip[f["chip"]].append(f)
    chips = sorted(by_chip); np.random.RandomState(args.seed).shuffle(chips)
    half = max(1, len(chips) // 2)
    tune = {k: by_chip[k] for k in chips[:half]}
    test = {k: by_chip[k] for k in chips[half:]}
    flat = lambda ch, key: np.concatenate([np.array([r[key] for r in rs]) for rs in ch.values()])

    def sdi(ch, cfg):
        preds, gts = [], []
        for rs in ch.values():
            prob = np.array([r["mean_prob"] for r in rs])
            cent = np.array([[r["cx"], r["cy"]] for r in rs])
            preds.append(infer_affected(prob, cent, cfg))
            gts.append(np.array([r["gt_affected"] for r in rs], bool))
        return np.concatenate(preds), np.concatenate(gts)

    # tune
    b3 = max(np.linspace(0.05, 0.9, 18),
             key=lambda t: prf(baseline_prob_threshold(flat(tune, "mean_prob"), t), flat(tune, "gt_affected"))["f1"])
    potts_l = max([0.5, 1.0, 1.5, 2.0, 3.0],
                  key=lambda l: prf(*sdi(tune, SDIConfig(mode="potts", lambda_smooth=l, radius_m=60, sigma_m=40)))["f1"])
    best = (0, None)
    for it in [0.2, 0.3, 0.4, 0.5]:
        for be in [0.5, 1.0, 1.5, 2.0, 3.0]:
            cfg = SDIConfig(mode="attractive", lambda_smooth=be, init_thresh=it, radius_m=60, sigma_m=40)
            f1 = prf(*sdi(tune, cfg))["f1"]
            if f1 > best[0]:
                best = (f1, cfg)
    attr_cfg = best[1]
    print(f"tuned: B3 thr={b3:.3f} | Potts lambda={potts_l} | "
          f"Attractive init={attr_cfg.init_thresh} beta={attr_cfg.lambda_smooth}")

    gt = flat(test, "gt_affected"); ffh = flat(test, "flood_frac_hard"); mp = flat(test, "mean_prob")
    res = {
        "B2_any_intersection": prf(baseline_any_intersection(ffh), gt),
        "B1_raw_threshold": prf(baseline_raw_threshold(ffh), gt),
        "B3_prob_threshold*": prf(baseline_prob_threshold(mp, b3), gt),
        "SDI_potts": prf(*sdi(test, SDIConfig(mode="potts", lambda_smooth=potts_l, radius_m=60, sigma_m=40))),
        "SDI_attractive": prf(*sdi(test, attr_cfg)),
    }
    print(f"\n=== test: {int(len(gt))} buildings, {int(gt.sum())} damaged ===")
    for k in ["B2_any_intersection", "B1_raw_threshold", "B3_prob_threshold*", "SDI_potts", "SDI_attractive"]:
        m = res[k]; print(f"  {k:22s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    bb = max(res[k]["f1"] for k in res if not k.startswith("SDI_attr"))
    print(f"\n  SDI_attractive F1={res['SDI_attractive']['f1']:.3f}  vs best-other {bb:.3f}  "
          f"({res['SDI_attractive']['f1']-bb:+.3f})")
    json.dump({"tuned_b3": float(b3), "potts_lambda": potts_l,
               "attr_init": attr_cfg.init_thresh, "attr_beta": attr_cfg.lambda_smooth,
               "methods": res}, open("outputs/xbd_damage_sdi/sdi_offline_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
