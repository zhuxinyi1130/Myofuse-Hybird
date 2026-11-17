#!/bin/zsh
set -euo pipefail

if [[ $# -lt 2 || $# -gt 12 ]]; then
  print -u2 "Usage: $0 INPUT.czi OUTPUT_DIR [SCENE] [NUCLEI_CHANNEL] [ACTIN_CHANNEL] [MF20_CHANNEL] [HYBRID_ONLY_PROBABILITY_MIN] [DIRECTION_AWARE] [GRAPH_AWARE] [ADAPTIVE_REVIEW] [ACTIN_STRUCTURE_RESCUE] [STRUCTURE_V2]"
  exit 2
fi

script_dir="${0:A:h}"
python_bin="${PYTHON_BIN:-$script_dir/.venv/bin/python}"

input_file="$1"
output_dir="$2"
scene="${3:-ALL}"
nuclei_channel="${4:-0}"
actin_channel="${5:-1}"
mf20_channel="${6:-2}"
hybrid_only_probability_min="${7:-0}"
direction_aware="${8:-0}"
graph_aware="${9:-0}"
adaptive_review="${10:-0}"
actin_structure_rescue="${11:-0}"
structure_v2="${12:-0}"

if [[ ! -f "$input_file" || "${input_file:l}" != *.czi ]]; then
  print -u2 "Input must be one existing .czi file: $input_file"
  exit 2
fi

if [[ ! -x "$python_bin" ]]; then
  print -u2 "Python environment not found: $python_bin"
  print -u2 "Run ./setup_mac.sh first, or set PYTHON_BIN to a compatible Python executable."
  exit 1
fi

myofuse_dir="$output_dir/myofuse"
hybrid_dir="$output_dir/hybrid"
mkdir -p "$myofuse_dir" "$hybrid_dir"

if [[ "$scene" == "ALL" ]]; then
  scenes=("${(@f)$("$python_bin" "$script_dir/list_czi_scenes.py" "$input_file")}")
else
  scenes=("$scene")
fi

print "Step 1/2: running MyoFuse nucleus segmentation and MF20 classification"
myofuse_command=(
  "$python_bin" "$script_dir/run_myofuse.py"
  --input "$input_file"
  --output "$myofuse_dir"
  --nuclei-channel "$nuclei_channel"
  --myotube-channel "$mf20_channel"
  --save-masks
  --save-overlays
)
for analysis_scene in "${scenes[@]}"; do
  myofuse_command+=(--scene "$analysis_scene")
done
"${myofuse_command[@]}"

scene_number=0
for analysis_scene in "${scenes[@]}"; do
  (( scene_number += 1 ))
  print "Step 2/2: running hybrid analysis for $analysis_scene ($scene_number/${#scenes})"
  hybrid_command=(
    "$python_bin" "$script_dir/run_hybrid_fusion.py"
    --input "$input_file"
    --myofuse-results "$myofuse_dir"
    --output "$hybrid_dir"
    --scene "$analysis_scene"
    --nuclei-channel "$nuclei_channel"
    --actin-channel "$actin_channel"
    --mf20-channel "$mf20_channel"
    --hybrid-only-probability-min "$hybrid_only_probability_min"
  )
  if [[ "$direction_aware" == "1" ]]; then
    hybrid_command+=(--direction-aware)
  fi
  if [[ "$graph_aware" == "1" ]]; then
    hybrid_command+=(--graph-aware)
  fi
  if [[ "$adaptive_review" == "1" ]]; then
    hybrid_command+=(--adaptive-review)
  fi
  if [[ "$actin_structure_rescue" == "1" ]]; then
    hybrid_command+=(--actin-structure-rescue)
  fi
  if [[ "$structure_v2" == "1" ]]; then
    hybrid_command+=(--structure-v2)
  fi
  "${hybrid_command[@]}"
done

print "Completed. Hybrid results: $hybrid_dir"
