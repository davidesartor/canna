# Source pairing: superseded (2026-11-02)

The question this report measured, which labelling of a base draw's sources onto the target's gives the shortest
flow-matching paths, no longer exists. Training now draws the base z0 standard normal per slot and independent of the
target, the target x1 sorted by f0 (`train.paths`), and the flow works on z = (x1 − μ)/σ in the encoder's frame, so
there is no X0 to pair. Sources are unordered in the prior; f0 order is only the slot labelling of the target.

The page and numbers below are the 2026-10-30 measurement (`visible_snr = 7` windows, 4 sources, 65536 windows per
band, commit 8310cb3) of X0-to-X1 coupling in raw flow coordinates, kept as history: f0 order had the shortest f0 paths
(0.26 against 0.49 for the nearest labelling and 0.67 for random labels) and was 5% (narrow-B) and 3% (full-B) shorter in
total than random labels, all of it from f0.

`scripts/pairing.py` imports `sort_by_snr`, which is gone, and is not ported: it would rebuild a comparison the training
no longer makes. The page [source-pairing.html](source-pairing.html) is unchanged and its numbers are not regenerated.
