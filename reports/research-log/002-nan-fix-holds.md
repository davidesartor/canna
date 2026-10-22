# F2 (2026-09-17 to 29): the NaN fix holds
- The x loss is now `log_map(target, pred)`; the old argument order scaled with the
  prediction's radius.
- 13264195 passed the old blow-up epochs (55, 62) and finished all 500 epochs finite. Early
  losses match the old run within ±0.003 by epoch 43.
