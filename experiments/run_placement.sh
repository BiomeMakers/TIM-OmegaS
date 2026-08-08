#!/bin/bash
# =============================================================================
# run_placement.sh  --  the validation run behind the retention numbers.
#
# NO PENALTY IS INVOLVED. Both arms are plain LoRA. The question is placement
# alone, so this run does not depend on anything being measured about the
# regulariser: it can be run whatever those results turn out to be.
#
# ARM A (suggested): the quartile of largest relative drop from the diagnostic
#   in experiments/adapter_placement.py, on Llama-3-8B. 31 modules, of which 17
#   are o_proj, 10 q_proj and 4 k_proj. Not a single v_proj.
# ARM B (conventional): q_proj and v_proj, which is what everybody does, cut to
#   31 modules across 16 layers spaced through the model.
#
# THE CONTROL THAT MAKES THE COMPARISON VALID: both arms adapt EXACTLY 31
# modules, so what is compared is WHERE the adapter goes and not HOW MUCH of it
# there is. The harness prints how many it adapted; if the two numbers do not
# agree, stop.
#
# HYPOTHESIS, WRITTEN BEFORE RUNNING: if Steele's law (arXiv:2603.02224) is
# right and forgetting is governed by gradient-subspace overlap, the suggested
# arm should retain more. It would also be awkward for established practice,
# because v_proj is the type with the MOST overlap of the four and it is
# precisely the one the convention adapts.
#
# IF IT COMES OUT THE OTHER WAY, that is a result too. It would mean that
# gradient-subspace overlap measured before training does not predict
# retention, which bounds Steele's law to the settings he tested.
#
#   nohup bash run_placement.sh > run_placement.log 2>&1 &
#
# This is the script as it was run, with paths and Spanish output translated.
# It expects the harness of the companion repository (OmegaS-LLM) and its
# LORA_MODULES hook for explicit placement. The two module lists are also
# shipped separately in results/placements_used.json.
# =============================================================================
set -u
cd "${HARNESS:-/workspace/omega-s}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONPATH="$PWD:$PYTHONPATH"

SUG="model.layers.22.self_attn.o_proj,model.layers.25.self_attn.o_proj,model.layers.1.self_attn.o_proj,model.layers.19.self_attn.o_proj,model.layers.27.self_attn.o_proj,model.layers.17.self_attn.o_proj,model.layers.31.self_attn.q_proj,model.layers.23.self_attn.o_proj,model.layers.20.self_attn.o_proj,model.layers.30.self_attn.o_proj,model.layers.30.self_attn.q_proj,model.layers.24.self_attn.o_proj,model.layers.27.self_attn.k_proj,model.layers.23.self_attn.k_proj,model.layers.23.self_attn.q_proj,model.layers.16.self_attn.o_proj,model.layers.28.self_attn.o_proj,model.layers.19.self_attn.q_proj,model.layers.18.self_attn.o_proj,model.layers.19.self_attn.k_proj,model.layers.29.self_attn.q_proj,model.layers.24.self_attn.q_proj,model.layers.26.self_attn.q_proj,model.layers.21.self_attn.o_proj,model.layers.22.self_attn.q_proj,model.layers.2.self_attn.o_proj,model.layers.28.self_attn.q_proj,model.layers.27.self_attn.q_proj,model.layers.29.self_attn.k_proj,model.layers.0.self_attn.o_proj,model.layers.3.self_attn.o_proj"
CONV="model.layers.0.self_attn.q_proj,model.layers.0.self_attn.v_proj,model.layers.2.self_attn.q_proj,model.layers.2.self_attn.v_proj,model.layers.4.self_attn.q_proj,model.layers.4.self_attn.v_proj,model.layers.6.self_attn.q_proj,model.layers.6.self_attn.v_proj,model.layers.8.self_attn.q_proj,model.layers.8.self_attn.v_proj,model.layers.10.self_attn.q_proj,model.layers.10.self_attn.v_proj,model.layers.12.self_attn.q_proj,model.layers.12.self_attn.v_proj,model.layers.14.self_attn.q_proj,model.layers.14.self_attn.v_proj,model.layers.17.self_attn.q_proj,model.layers.17.self_attn.v_proj,model.layers.19.self_attn.q_proj,model.layers.19.self_attn.v_proj,model.layers.21.self_attn.q_proj,model.layers.21.self_attn.v_proj,model.layers.23.self_attn.q_proj,model.layers.23.self_attn.v_proj,model.layers.25.self_attn.q_proj,model.layers.25.self_attn.v_proj,model.layers.27.self_attn.q_proj,model.layers.27.self_attn.v_proj,model.layers.29.self_attn.q_proj,model.layers.29.self_attn.v_proj,model.layers.31.self_attn.q_proj"

echo "suggested modules:    $(echo $SUG | tr ',' '\n' | wc -l)"
echo "conventional modules: $(echo $CONV | tr ',' '\n' | wc -l)"

echo "[$(date +%H:%M:%S)] waiting for the pod to free up..."
sleep 180
while pgrep -f "rerun_retention\|measure_structure" > /dev/null; do sleep 60; done
sleep 30

launch () {   # launch <arm> <module list> <seeds...>
  local arm="$1"; shift
  local mods="$1"; shift
  local i=$GPU0
  for s in "$@"; do
    CUDA_VISIBLE_DEVICES=$i LORA_MODULES="$mods" \
      nohup python experiments/rerun_retention.py --arm none --seed $s \
      --out pl_${arm}_s$s.json > pl_${arm}_s$s.log 2>&1 &
    i=$((i+1)); sleep 5
  done
}

wait_all () {
  sleep 120
  while pgrep -f rerun_retention > /dev/null; do sleep 60; done
  sleep 30
}

echo "[$(date +%H:%M:%S)] BATCH 1: four seeds of each arm, 8 GPUs"
GPU0=0; launch sug  "$SUG"  42 123 456 789
GPU0=4; launch conv "$CONV" 42 123 456 789
wait_all

echo "[$(date +%H:%M:%S)] BATCH 2: the next four seeds of each arm"
GPU0=0; launch sug  "$SUG"  1011 2022 3033 4044
GPU0=4; launch conv "$CONV" 1011 2022 3033 4044
wait_all

echo "[$(date +%H:%M:%S)] BATCH 3: 5055 and 6066"
GPU0=0; launch sug  "$SUG"  5055 6066
GPU0=2; launch conv "$CONV" 5055 6066
wait_all

echo
echo "=================== DONE ==================="
echo "--- control: number of adapted modules per arm ---"
grep -h "\[lora\]" pl_sug_s42.log pl_conv_s42.log | head -4
echo "  (if the two numbers are not both 31, this is NOT a placement comparison)"
echo
python3 - << "PY"
import glob, re, statistics as st
from math import comb
from itertools import product


def read(pattern):
    """Parse the harness logs. The two Spanish alternatives are deliberate:
    the original run predates the harness being translated, so this keeps
    working on the logs that produced the published numbers."""
    out = {}
    for f in glob.glob(pattern):
        seed = int(re.search(r"_s(\d+)\.log", f).group(1))
        text = open(f).read()
        ret = re.search(r"RETENTION:\s+([\d.]+)", text) or \
              re.search(r"RETENCION:\s+([\d.]+)", text)
        after_a = re.search(r"after task A:\s+([\d.]+)", text) or \
                  re.search(r"tras tarea A:\s+([\d.]+)", text)
        if ret:
            out[seed] = (float(ret.group(1)),
                         float(after_a.group(1)) if after_a else None)
    return out


sug, conv = read("pl_sug_s*.log"), read("pl_conv_s*.log")
common = sorted(set(sug) & set(conv))
print(f"{'seed':>8} {'suggested':>11} {'conventional':>13} {'diff':>9}"
      "   task-A sug/conv")
diffs = []
for s in common:
    d = sug[s][0] - conv[s][0]
    diffs.append(d)
    print(f"  {s:>6} {sug[s][0]:11.4f} {conv[s][0]:13.4f} {d:+9.4f}   "
          f"{sug[s][1] or 0:.3f}/{conv[s][1] or 0:.3f}")
if not diffs:
    print("no results")
    raise SystemExit

# Ties are discarded from the sign test rather than awarded to either arm.
wins = sum(1 for x in diffs if x > 0)
losses = sum(1 for x in diffs if x < 0)
n = wins + losses
p_sign = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n

nz = [x for x in diffs if x != 0]
order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
rank = {i: k + 1 for k, i in enumerate(order)}
w_pos = sum(rank[i] for i in range(len(nz)) if nz[i] > 0)
w_neg = sum(rank[i] for i in range(len(nz)) if nz[i] < 0)
W = min(w_pos, w_neg)
count = sum(1 for sg in product([0, 1], repeat=len(nz))
            if min(sum(rank[i] for i in range(len(nz)) if sg[i]),
                   sum(rank[i] for i in range(len(nz)) if not sg[i])) <= W)
print()
print(f"  suggested wins {wins} of {n}   mean difference {st.mean(diffs):+.4f}")
print(f"  sign test p={p_sign:.4f} one-sided   "
      f"Wilcoxon W={W} p={count / 2 ** len(nz):.4f}")
print(f"  means: suggested {st.mean([sug[s][0] for s in common]):.4f}  "
      f"conventional {st.mean([conv[s][0] for s in common]):.4f}")
print()
print("  Check the task-A column: if one arm learns task A worse, its")
print("  retention is not comparable. That is the ceiling effect, and it is")
print("  why the absolute figures are reported alongside the ratio.")
PY
echo
echo "SHUT THE POD DOWN."
