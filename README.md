|
# Baseline: the v9 file in the previous code export. This changelog covers only the
# v10 re-render produced in the prior turn. It does not cover anything in
# hammer.py, proc_loop.py, datafeeder.py, or gradcheck.py — those are unchanged.
#
# ------------------------------------------------------------------------------
# 1. FILE HEADER (docstring block at top)
# ------------------------------------------------------------------------------
# Changed. The Settlement paragraph was rewritten. v9 said "Settlement never
# discards; discards come only from expiry and malformed input." v10 replaces
# that with the three discard paths (expiry, malformed input,
# hallucinated-for-shit under the declared valence) and states the valence as a
# bake-in.
#
# Changed. The Dissonance paragraph now declares the valence: "dissonance is
# pain. Pain is a hold signal for persistent surprise and an accelerate signal
# for acute surprise. The two-timescale split (acute/tonic) is not yet
# implemented; until it is, bump_p carries the tonic role."
#
# Added. A new "Honored class (v10)" paragraph distinguishing sense-1
# (Proposal.provenance, self-description) from sense-2 (the honored class on
# the ledger record, the accumulator the outcome folds into).
#
# Changed. The Rejection-boundary paragraph now includes the provenance header
# as a rejection point: "A provenance header naming a class not in PROVENANCE
# is rejected at fork."
#
# Unchanged. The Clocks, Hot state, Relaxation, Ledger, Antagonism, Breath
# phase, Bake-ins, and Consolidation-loss paragraphs are identical to v9.
#
# ------------------------------------------------------------------------------
# 2. DEFAULTS
# ------------------------------------------------------------------------------
# Added three keys:
# - gem_bump=0.5 — bump at or above this counts as persistent surprise.
# - gem_support=2.0 — support mass at or above this counts as kept-settling.
# - valence="tonic_hold" — the declared valence; swapping it swaps the discard
# rule.
#
# Added a comment block above DEFAULTS declaring the valence as a bake-in,
# stated not derived, with the two-timescale split named as not-yet-implemented.
#
# Unchanged. All prior keys (kappa, kappa_hot, lam, beta, s0, mu0, gamma,
# lam_b, diss_gain, bump_max, eta, kappa_pi, tau_keep, nov_thresh, ttl,
# phase_on, phase_base) keep their v9 values.
#
# ------------------------------------------------------------------------------
# 3. Proposal DATACLASS
# ------------------------------------------------------------------------------
# Changed comment only. The provenance field's inline comment now reads
# "# sense-1 self-description (not the honored class)". The field, its default
# ("inferred"), and its position are unchanged. Token is unchanged.
#
# ------------------------------------------------------------------------------
# 4. ForkGenerator.init
# ------------------------------------------------------------------------------
# Added a valence validation block after the beta > 0 check:
#
# if cfg["valence"] not in ("tonic_hold",):
# raise ValueError(f"valence {cfg['valence']!r} not implemented; "
# f"only 'tonic_hold' is available")
#
# The error message states that adding a second valence is a deliberate change
# to settle(), not a runtime switch.
#
# Unchanged. Every parameter, state array, ledger attribute, and bookkeeping
# counter is identical to v9. bump remains a single (C,) scalar array — the
# two-timescale split is not landed.
#
# ------------------------------------------------------------------------------
# 5. fork()
# ------------------------------------------------------------------------------
# Added a header-honoring step before the prediction:
#
# tags = dict(conditioning_tags or {})
# honored = tags.get("provenance", Proposal.provenance)
# if honored not in PROVENANCE:
# raise ValueError(f"fork: provenance header {honored!r} not in {PROVENANCE}")
#
# Changed the ledger record to include provenance=honored alongside x, phase,
# mu, unc, novelty, t0.
#
# Changed the returned Proposal to pass conditioning_tags=tags (the copied
# dict) rather than dict(conditioning_tags or {}). Functionally identical;
# cosmetic.
#
# Changed the docstring to state the header's role: the class the fork was
# asked to produce, honored at settle unless overridden, with the default being
# Proposal.provenance.
#
# Added a rejection path: a header naming a class not in PROVENANCE raises
# ValueError at the boundary. n_arr is not incremented on rejection (the raise
# happens before the ledger write).
#
# Unchanged. Everything else in fork() — the non-finite check, the prediction,
# the trajectory-id hash, the n_arr increment — is identical to v9.
#
# ------------------------------------------------------------------------------
# 6. settle()
# ------------------------------------------------------------------------------
# This is the largest change. Four distinct edits:
#
# Signature changed. provenance="observed" -> provenance=None. The silent
# default is gone.
#
# Added class resolution. After popping the ledger record:
#
# p_name = provenance if provenance is not None else rec["provenance"]
# if p_name not in PROVENANCE:
# raise ValueError(f"settle: override provenance {p_name!r} not in {PROVENANCE}")
# p = PROVENANCE.index(p_name)
#
# The honored class comes from the record unless the caller passes an explicit
# override. An override not in PROVENANCE is rejected — but after
# ledger.pop(tid), which means a bad override consumes the record. (v9's
# non-finite y check was deliberately before pop; the override check is after.
# This is an asymmetry: a bad y leaves the fork open, a bad override does not.
# Worth noting; may want to move the override check before pop in a follow-up,
# matching the y discipline.)
#
# Added the discard branch for hallucinated under tonic_hold:
#
# if p_name == "hallucinated" and c["valence"] == "tonic_hold":
# gem = (self.bump[p] >= c["gem_bump"]) or (self.M[p] >= c["gem_support"])
# if not gem:
# self.n_discarded += 1
# return dict(agree=agree, w=w, z=z, novel=novel,
# folded=False, discarded=True, gem=False)
#
# The gem check is the trembling-only approximation: persistent surprise (bump)
# OR repeated settling (M). A discarded hallucination still increments SN[p],
# SA[p], and (if applicable) SNA[p] — the antagonism statistics see the settled
# outcome — but skips observe(), so A, B, M, and bump are not updated.
#
# Changed the return dict to include discarded and gem keys. v9 returned
# dict(agree, w, z, novel, folded); v10 returns dict(agree, w, z, novel, folded,
# discarded, gem).
#
# Changed the docstring to state the honored-class rule, the valence, and the
# trembling-only limitation explicitly ("it treats tourmalines as gems").
#
# Unchanged. The math (agree, surprise, kept, w = max(agree, surprise), novel,
# SN/SA/SNA increments) is identical to v9. The folded = w > 0.0 branch for
# observed/inferred is identical.
