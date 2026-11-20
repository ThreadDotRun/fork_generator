#!/usr/bin/env bash
# /home/tdrsvr/Desktop/slop/tests/cli_test.sh
# cli_test.sh: black-box CLI test suite for fork_generator.py
#
# Usage:
#   cli_test.sh [path/to/fork_generator.py] [-v] [--quick] [--keep]
#     path     defaults to ../fork_generator.py relative to this script
#     -v       print stderr tail for failing commands
#     --quick  skip the full selftest (slowest case)
#     --keep   keep the working directory for inspection
#
# Exit: 0 all pass, 1 any failure, 2 setup error.
# Everything runs in a temp directory. The suite never touches the source tree.

set -u

VERBOSE=0
QUICK=0
KEEP=0
FG_ARG=""
for a in "$@"; do
  case "$a" in
    -v) VERBOSE=1 ;;
    --quick) QUICK=1 ;;
    --keep) KEEP=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) FG_ARG="$a" ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "$FG_ARG" ]; then
  if [ -f "$HERE/../fork_generator.py" ]; then FG_ARG="$HERE/../fork_generator.py"
  else FG_ARG="$HERE/fork_generator.py"; fi
fi
[ -f "$FG_ARG" ] || { echo "setup: cannot find $FG_ARG" >&2; exit 2; }
FG="$(cd "$(dirname "$FG_ARG")" && pwd)/$(basename "$FG_ARG")"
command -v python3 >/dev/null 2>&1 || { echo "setup: python3 not found" >&2; exit 2; }
python3 -c "import numpy" 2>/dev/null || { echo "setup: numpy not importable" >&2; exit 2; }

WORK="$(mktemp -d -t fgcli.XXXXXX)" || { echo "setup: mktemp failed" >&2; exit 2; }
cleanup() {
  if [ "$KEEP" -eq 1 ]; then echo "kept working dir: $WORK"; else rm -rf "$WORK"; fi
}
trap cleanup EXIT
cd "$WORK" || exit 2

PASS=0
FAIL=0
LASTERR=""

section() { printf '\n== %s\n' "$1"; }
ok()  { PASS=$((PASS + 1)); printf '  PASS  %s\n' "$1"; }
bad() {
  FAIL=$((FAIL + 1)); printf '  FAIL  %s\n' "$1"
  if [ "$VERBOSE" -eq 1 ] && [ -n "$LASTERR" ] && [ -s "$LASTERR" ]; then
    sed 's/^/          | /' "$LASTERR" | tail -n 12
  fi
}

# run NAME ARGS...   stdin from $IN if set. Writes $WORK/NAME.out and NAME.err. Sets RC.
run() {
  local name="$1"; shift
  local stdin="${IN:-/dev/null}"
  timeout "${TMO:-600}" python3 "$FG" "$@" <"$stdin" >"$WORK/$name.out" 2>"$WORK/$name.err"
  RC=$?
  LASTERR="$WORK/$name.err"
}

expect_rc() {        # expect_rc NAME WANT DESC
  if [ "$RC" -eq "$2" ]; then ok "$3 [rc=$RC]"; else bad "$3 (want rc=$2, got $RC)"; fi
}
expect_rc_nz() {     # expect_rc_nz DESC
  if [ "$RC" -ne 0 ]; then ok "$1 [rc=$RC]"; else bad "$1 (want nonzero rc)"; fi
}
has() {              # has NAME PATTERN DESC   (stdout or stderr)
  if grep -qE -- "$2" "$WORK/$1.out" "$WORK/$1.err"; then ok "$3"; else bad "$3 (no /$2/)"; fi
}
hasnt() {            # hasnt NAME PATTERN DESC (stdout)
  if grep -qE -- "$2" "$WORK/$1.out"; then bad "$3 (unexpected /$2/)"; else ok "$3"; fi
}
count() {            # count NAME PATTERN  -> prints number of stdout lines matching
  grep -cE -- "$2" "$WORK/$1.out" || true
}
expect_file() {      # expect_file PATH DESC
  if [ -s "$1" ]; then ok "$2"; else bad "$2 ($1 missing or empty)"; fi
}
eq() {               # eq GOT WANT DESC
  if [ "$1" = "$2" ]; then ok "$3"; else bad "$3 (got '$1', want '$2')"; fi
}
ne() {               # ne A B DESC
  if [ "$1" != "$2" ]; then ok "$3"; else bad "$3 (both '$1')"; fi
}

# jget NAME MARKER EXPR  : parse the JSON that follows MARKER in NAME.out; EXPR sees d
jget() {
  python3 - "$WORK/$1.out" "$2" "$3" <<'PY' 2>/dev/null || echo "<parse-error>"
import sys, json
txt = open(sys.argv[1]).read()
i = txt.find(sys.argv[2])
if i < 0:
    print("<missing>"); sys.exit(0)
i += len(sys.argv[2])
d, _ = json.JSONDecoder().raw_decode(txt[i:].lstrip())
print(eval(sys.argv[3], {}, {"d": d}))
PY
}

# jall NAME EXPR : evaluate EXPR over every JSON object in NAME.out (one line per object)
jall() {
  python3 - "$WORK/$1.out" "$2" <<'PY' 2>/dev/null
import sys, json
txt = open(sys.argv[1]).read()
dec = json.JSONDecoder()
i, objs = 0, []
while True:
    j = txt.find("{", i)
    if j < 0:
        break
    try:
        d, k = dec.raw_decode(txt, j)
    except ValueError:
        i = j + 1
        continue
    objs.append(d)
    i = k
for d in objs:
    print(eval(sys.argv[2], {}, {"d": d}))
PY
}

# ---------- fixtures ----------
make_inputs() {      # make_inputs FILE N  -> N valid JSON lines (16/8/8 features)
  python3 - "$1" "$2" <<'PY'
import sys, json, numpy as np
rng = np.random.default_rng(3)
with open(sys.argv[1], "w") as f:
    for _ in range(int(sys.argv[2])):
        f.write(json.dumps(dict(delta=rng.normal(size=16).round(4).tolist(),
                                hum=rng.uniform(0, 1, 8).round(4).tolist(),
                                state=rng.normal(size=8).round(4).tolist(),
                                phase=0.0)) + "\n")
PY
}

make_corpus() {      # make_corpus FILE KIND  KIND: legacy|badshape|badlen|negw|empty
  python3 - "$1" "$2" <<'PY'
import sys, pickle, numpy as np
path, kind = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
if kind == "legacy":
    items = [(rng.normal(size=32), rng.normal(size=6), None, float(rng.uniform(.5, 1)))
             for _ in range(40)]
elif kind == "badshape":
    items = [(np.zeros(31), np.zeros(6), 1.0)]
elif kind == "badlen":
    items = [(np.zeros(32), np.zeros(6))]
elif kind == "negw":
    items = [(np.zeros(32), np.zeros(6), -1.0)]
elif kind == "empty":
    items = []
else:
    raise SystemExit("unknown kind")
pickle.dump(items, open(path, "wb"))
PY
}

# =====================================================================
section "1. argument handling"
run argnone
expect_rc argnone 2 "no subcommand exits 2 (argparse)"

run argbogus bogus-subcommand
expect_rc argbogus 2 "unknown subcommand exits 2"

run helptop --help
expect_rc helptop 0 "--help exits 0"
has helptop "usage" "top-level help prints usage"

run helptrain train --help
expect_rc helptrain 0 "train --help exits 0"
has helptrain "--holdout" "train help lists --holdout"

run helpst selftest --help
expect_rc helpst 0 "selftest --help exits 0"
has helpst "--epochs" "selftest help lists --epochs"

run helpinf infer --help
expect_rc helpinf 0 "infer --help exits 0"

# =====================================================================
section "2. make-td"
run td1 make-td --out "$WORK/td1.dat" --n 32 --seed 7
expect_rc td1 0 "make-td writes a corpus"
has td1 "wrote 32 items" "make-td reports item count"
expect_file "$WORK/td1.dat" "corpus file non-empty"

run td2 make-td --out "$WORK/td2.dat" --n 32 --seed 7
if cmp -s "$WORK/td1.dat" "$WORK/td2.dat"; then ok "same seed gives byte-identical corpus"
else bad "same seed gives byte-identical corpus"; fi

run td3 make-td --out "$WORK/td3.dat" --n 32 --seed 8
if cmp -s "$WORK/td1.dat" "$WORK/td3.dat"; then bad "different seed gives different corpus"
else ok "different seed gives different corpus"; fi

mkdir -p "$WORK/defaults" && cd "$WORK/defaults" || exit 2
run tddef make-td --n 8
expect_rc tddef 0 "make-td default output path"
expect_file "$WORK/defaults/td.dat" "default output lands in cwd as td.dat"
cd "$WORK" || exit 2

# =====================================================================
section "3. train: outputs and bookkeeping"
run tr1 train --n 128 --epochs 3 --holdout 0.25 --seed 0 --out "$WORK/m1.pkl"
expect_rc tr1 0 "train runs end to end"
has tr1 '^\[save\] ' "save line printed"
has tr1 '^\[holdout\] ' "holdout report printed when --holdout > 0"
expect_file "$WORK/m1.pkl" "model pickle written"
eq "$(jget tr1 '[holdout] ' 'd["n"]')" 32 "holdout size = round(128 * 0.25)"
eq "$(jget tr1 '[diagnostics] ' 'd["train_n"]')" 96 "train_n = 128 - holdout"
eq "$(jget tr1 '[diagnostics] ' 'd["steps"]')" 9 "steps = epochs * ceil(96 / 32)"
eq "$(jget tr1 '[diagnostics] ' 'd["absorbed"]')" 1 "train absorbs hot state exactly once"
eq "$(jget tr1 '[diagnostics] ' 'd["ledger"]["balanced"]')" True "ledger balanced after train"
eq "$(jget tr1 '[diagnostics] ' 'd["field_time"]')" 0.0 "train advances no field-time"
eq "$(jget tr1 '[diagnostics] ' 'len(d["pi"])')" 3 "three provenance classes in pi"

run tr0 train --n 64 --epochs 2 --holdout 0 --out "$WORK/m0.pkl"
expect_rc tr0 0 "train with --holdout 0 runs"
hasnt tr0 '^\[holdout\]' "no holdout report when --holdout 0"

run ep0 train --n 32 --epochs 0 --holdout 0 --out "$WORK/mz.pkl"
expect_rc ep0 0 "zero epochs runs"
has ep0 '^\[diagnostics\] \{\}' "zero epochs gives empty diagnostics"

# =====================================================================
section "4. train: diagnostics, determinism, learning"
run trd train --n 64 --epochs 2 --holdout 0 --diagnose --out "$WORK/md.pkl"
expect_rc trd 0 "diagnose mode runs"
eq "$(count trd '^\[train\] ep=')" 2 "one per-epoch line per epoch"
has trd 'mu_rail=' "per-epoch line includes rail fractions"
has trd 'lu_rail=' "per-epoch line includes lu rail fraction"

run ra train --n 128 --epochs 4 --holdout 0.2 --seed 5 --out "$WORK/ra.pkl"
run rb train --n 128 --epochs 4 --holdout 0.2 --seed 5 --out "$WORK/rb.pkl"
eq "$(jget ra '[diagnostics] ' 'd["final_loss"]')" \
   "$(jget rb '[diagnostics] ' 'd["final_loss"]')" "same seed gives identical final loss"

run rc train --n 128 --epochs 4 --holdout 0.2 --seed 6 --out "$WORK/rc.pkl"
ne "$(jget ra '[diagnostics] ' 'd["final_loss"]')" \
   "$(jget rc '[diagnostics] ' 'd["final_loss"]')" "different seed gives different final loss"

run learn train --n 256 --epochs 20 --holdout 0 --seed 0 --out "$WORK/learn.pkl"
eq "$(jget learn '[diagnostics] ' 'd["improvement"] > 0')" True "training reduces loss over 20 epochs"

# =====================================================================
section "5. train: corpus input and rejection"
make_corpus "$WORK/legacy.pkl" legacy
run leg train --corpus "$WORK/legacy.pkl" --epochs 1 --holdout 0 --out "$WORK/mleg.pkl"
expect_rc leg 0 "legacy 4-tuple corpus loads"
has leg 'loaded .*legacy\.pkl' "loader reports source"
has leg ' 40 items' "legacy corpus item count correct"

run tdc train --corpus "$WORK/td1.dat" --epochs 1 --holdout 0 --out "$WORK/mtd.pkl"
expect_rc tdc 0 "make-td corpus loads into train"

run miss train --corpus "$WORK/does_not_exist.pkl" --epochs 1 --out "$WORK/x.pkl"
expect_rc_nz "missing corpus file fails"

make_corpus "$WORK/badshape.pkl" badshape
run bsh train --corpus "$WORK/badshape.pkl" --epochs 1 --holdout 0 --out "$WORK/x.pkl"
expect_rc_nz "wrong-shape item rejected"
has bsh 'shape' "shape error names the problem"

make_corpus "$WORK/badlen.pkl" badlen
run blen train --corpus "$WORK/badlen.pkl" --epochs 1 --holdout 0 --out "$WORK/x.pkl"
expect_rc_nz "two-field item rejected"
has blen 'expected \(x, y, w\)' "length error names expected form"

make_corpus "$WORK/negw.pkl" negw
run nw train --corpus "$WORK/negw.pkl" --epochs 1 --holdout 0 --out "$WORK/x.pkl"
expect_rc_nz "negative weight rejected"
has nw 'finite' "weight error names the rule"

make_corpus "$WORK/empty.pkl" empty
run em train --corpus "$WORK/empty.pkl" --epochs 1 --holdout 0 --out "$WORK/x.pkl"
expect_rc_nz "empty corpus rejected"
has em 'empty corpus' "empty-corpus error reported"

# =====================================================================
section "6. infer: stdin protocol and outputs"
make_inputs "$WORK/in3.jsonl" 3
IN="$WORK/in3.jsonl" run inf1 infer --model "$WORK/m1.pkl"
expect_rc inf1 0 "infer runs on three valid lines"
eq "$(count inf1 'trajectory_id')" 3 "one proposal per input line"
eq "$(jall inf1 'd["trajectory_id"][:5]' | sort -u)" "read-" "trajectory ids are read- prefixed"
eq "$(jall inf1 'len(d["latent_trajectory"])' | sort -u)" 6 "latent trajectory has d_fork entries"
eq "$(jall inf1 'd["uncertainty"] > 0' | sort -u)" True "uncertainty positive and finite"

IN="$WORK/in3.jsonl" run inf2 infer --model "$WORK/m1.pkl"
if cmp -s "$WORK/inf1.out" "$WORK/inf2.out"; then ok "same model and input give identical output"
else bad "same model and input give identical output"; fi

printf '%s\n' '{"delta": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "phase": 0.0}' \
  '{"delta": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "phase": 0.5}' > "$WORK/phase.jsonl"
IN="$WORK/phase.jsonl" run infph infer --model "$WORK/m1.pkl"
expect_rc infph 0 "phase field accepted"
eq "$(jall infph 'd["latent_trajectory"][0]' | sort -u | wc -l | tr -d ' ')" 1 \
   "phase has no effect when phase_on is False"

printf '%s\n' 'not json' '{"delta": [1,2,3]}' > "$WORK/bad.jsonl"
cat "$WORK/in3.jsonl" | head -n 1 >> "$WORK/bad.jsonl"
IN="$WORK/bad.jsonl" run infbad infer --model "$WORK/m1.pkl"
expect_rc infbad 0 "bad lines do not abort the stream"
eq "$(count infbad '\[infer\]\[error\]')" 2 "each bad line reports one error"
eq "$(count infbad 'trajectory_id')" 1 "valid line after bad lines still answered"

printf '\n\n' > "$WORK/blank.jsonl"
head -n 1 "$WORK/in3.jsonl" >> "$WORK/blank.jsonl"
printf '\n' >> "$WORK/blank.jsonl"
IN="$WORK/blank.jsonl" run infblank infer --model "$WORK/m1.pkl"
eq "$(count infblank 'trajectory_id')" 1 "blank lines are skipped"

IN="$WORK/in3.jsonl" run infmiss infer --model "$WORK/no_such_model.pkl"
expect_rc_nz "missing model fails"

printf 'junk bytes, not a pickle' > "$WORK/junk.pkl"
IN="$WORK/in3.jsonl" run infjunk infer --model "$WORK/junk.pkl"
expect_rc_nz "non-pickle model file fails"

# =====================================================================
section "7. pickle portability across import contexts"
mkdir -p "$WORK/elsewhere" && cd "$WORK/elsewhere" || exit 2
PYTHONPATH="$(dirname "$FG")" timeout 120 python3 -c '
import sys
import fork_generator as fg
m = fg.ForkGenerator.load(sys.argv[1])
print(m.step, m.cfg["hidden"], m.cfg["phase_on"])
' "$WORK/m1.pkl" >"$WORK/port.out" 2>"$WORK/port.err"
PRC=$?
LASTERR="$WORK/port.err"
if [ "$PRC" -eq 0 ]; then ok "script-run pickle loads from an importer"
else bad "script-run pickle loads from an importer (rc=$PRC)"; fi
eq "$(cat "$WORK/port.out" 2>/dev/null)" "9 128 False" "loaded model has trained step count and config"
cd "$WORK" || exit 2

# =====================================================================
section "8. selftest"
if [ "$QUICK" -eq 1 ]; then
  printf '  SKIP  selftest (--quick)\n'
else
  run st selftest --seed 0 --epochs 60
  expect_rc st 0 "selftest exits 0"
  has st '19/19 passed' "selftest reports 19/19"
  hasnt st '^\[FAIL\]' "no selftest FAIL lines"
fi

# =====================================================================
printf '\n== summary: %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
