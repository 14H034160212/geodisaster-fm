"""One-time: extend chip_cache.pkl to all 10 Sen1Floods11 LOO regions (U-Net).

The original chip_cache.pkl only has the 4 hard regions (Pakistan, Somalia,
Paraguay, India) because the PPO sig experiment evaluated only on those.
For the leakage-free meta-train(6) / meta-test(4) protocol we need chip caches
for the 6 meta-train regions too: Ghana, Mekong, Nigeria, Sri-Lanka, USA, Spain.

Output: outputs/layer3_ppo/chip_cache_all10.pkl
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.eval_layer3_ppo_significance import cache_all_chips, _latest_ckpt
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("cache10")

ALL = ["Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay",
       "Somalia", "Spain", "Sri-Lanka", "USA"]
LOO_ROOT = "outputs/leave_one_region_out"
PATCHES = "data/processed/patches"
STATS = "data/processed/norm_stats_sen1floods11.yaml"
OUT = Path("outputs/layer3_ppo/chip_cache_all10.pkl")
SRC = Path("outputs/layer3_ppo/chip_cache.pkl")


def main():
    setup_logging()
    caches = pickle.loads(SRC.read_bytes()) if SRC.exists() else {}
    log.info("seed_cache", already_cached=list(caches))
    for region in ALL:
        if region in caches:
            continue
        ckdir = f"{LOO_ROOT}/test_{region}/checkpoints"
        ck = _latest_ckpt(ckdir)
        if ck is None:
            log.warning("no_ckpt", region=region); continue
        log.info("caching", region=region, ckpt=ck.name)
        caches[region] = cache_all_chips(region, ck, PATCHES, STATS)
    OUT.write_bytes(pickle.dumps(caches))
    log.info("saved", out=str(OUT), regions=list(caches),
             sizes={r: caches[r]["n"] for r in caches})


if __name__ == "__main__":
    sys.exit(main() or 0)
