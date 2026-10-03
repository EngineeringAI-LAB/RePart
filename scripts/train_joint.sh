#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

run_dirs=()
other_args=()
while (( $# > 0 )); do
  case "$1" in
    --run-dir)
      if (( $# < 2 )) || [[ "$2" == --* ]]; then
        echo "--run-dir requires a directory" >&2
        exit 2
      fi
      run_dirs+=("$2")
      shift 2
      ;;
    *)
      other_args+=("$1")
      shift
      ;;
  esac
done

if (( ${#run_dirs[@]} != 3 )); then
  echo "Usage: bash scripts/train_joint.sh --run-dir LITTLE_RUN --run-dir CONTAINER_RUN --run-dir FURNITURE_RUN [--out-dir OUT_DIR]" >&2
  exit 2
fi

python -m repart.joint_training "${run_dirs[@]}" "${other_args[@]}"
