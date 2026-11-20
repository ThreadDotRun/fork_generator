# Fork generator (fork_generator.py, v9)

## Hot path: a residual readout per provenance class, relaxed toward a ridge solution against a base prior. Forgetting applies to evidence, and support gates the pull.
## Dissonance: a bump from surprise, bounded per survivor and capped in total at bump_max = 1.0.
## Ledger: open forks expire in field-time, and the identity n_arr = n_folded + n_discarded + n_open holds exactly.
## Mixing: π is softmax of logits, and antagonism updates the logits.
## Settlement: every outcome folds at max(agree, surprise). Dissent is never gated out.
## Rejection: non-finite values are rejected at fork, infer, observe, settle, and ingest. Settle validates before consuming the record.
## Breath phase: an option, off by default.

# Open issues
## Discard path. Settlement never discards
## Only expiry and input validation remove data, so the feeder carries all the garbage defense
### Whether that is the right rule is still open
## π can’t move on one stream (expected)
## Every settlement is observed, so the other two classes never get data, and update_pi returns early
## The antagonism can’t run on a single source

