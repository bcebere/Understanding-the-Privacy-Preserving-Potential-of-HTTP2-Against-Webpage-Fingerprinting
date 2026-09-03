#!/usr/bin/env bash
# Usage:
#   ./prepare_workspace.sh <dataset> <defense|all|d1,d2,...> <arch_workspace> [out_workspace] [--benchmarks]
#
# Examples:
#   ./prepare_workspace.sh 4_udemy front ./archives
#   ./prepare_workspace.sh 4_udemy all ./archives ./workspace --benchmarks
#
# Expected Zenodo archives in <arch_workspace>:
#   overhead_analysis.tar.zst
#   datasets_calibrated_defenses_<dataset>_<kind>.tar.zst   (one per kind)

DATASET="${1:-}"
DEFENSE="${2:-}"
ARCH_WORKSPACE="${3:-}"
WORKSPACE="${4:-workspace}"
BENCHMARKS="${5:-}"

if [[ -z "$DATASET" || -z "$DEFENSE" || -z "$ARCH_WORKSPACE" ]]; then
    echo "Usage: $0 <dataset> <defense|all|d1,d2,...> <arch_workspace> [out_workspace] [--benchmarks]"
    exit 1
fi
# allow --benchmarks in place of out_workspace
if [[ "$WORKSPACE" == "--benchmarks" ]]; then
    BENCHMARKS="--benchmarks"
    WORKSPACE="workspace"
fi
WANT_BENCHMARKS=no
if [[ "$BENCHMARKS" == "--benchmarks" ]]; then
    WANT_BENCHMARKS=yes
fi

# "all" -> take whatever defenses the archive holds; otherwise a comma/space list
DEFENSES=()
if [[ "$DEFENSE" != "all" ]]; then
    IFS=', ' read -r -a DEFENSES <<< "$DEFENSE"
fi

OVERHEAD_ARCHIVE="${ARCH_WORKSPACE}/overhead_analysis.tar.zst"
if [[ ! -f "$OVERHEAD_ARCHIVE" ]]; then
    echo "ERROR: archive not found: $OVERHEAD_ARCHIVE"
    exit 1
fi
if ! compgen -G "${ARCH_WORKSPACE}/datasets_calibrated_defenses_${DATASET}_*.tar.zst" \
        > /dev/null; then
    echo "ERROR: no dataset archives for '$DATASET' in $ARCH_WORKSPACE"
    echo "  expected datasets_calibrated_defenses_${DATASET}_<kind>.tar.zst"
    exit 1
fi
DATASET_OUT="${WORKSPACE}/${DATASET}"
mkdir -p "$DATASET_OUT"

# staging area for the per-defense tarballs, on the same filesystem as the output
STAGE="$(mktemp -d "${WORKSPACE}/.stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

echo "Dataset:    $DATASET"
echo "Defenses:   $DEFENSE"
echo "Archives:   $ARCH_WORKSPACE"
echo "Workspace:  $DATASET_OUT"
echo "Benchmarks: $WANT_BENCHMARKS"
echo

extract_nested()
{
    local kind="$1"
    # Zenodo ships one archive per (dataset, kind)
    local src="${ARCH_WORKSPACE}/datasets_calibrated_defenses_${DATASET}_${kind}.tar.zst"
    local patterns=()
    local d f defense dest found=0

    if [[ ! -f "$src" ]]; then
        echo "WARNING: no archive for $kind: $src"
        echo
        return
    fi

    echo "============================================================"
    echo "Extracting $kind"
    echo "  $src"
    if [[ "$DEFENSE" == "all" ]]; then
        echo "    -> */tcp_repr/*_${kind}.tar.zst"
    else
        for d in "${DEFENSES[@]}"; do
            echo "    -> ${d}/tcp_repr/${d}_${kind}.tar.zst"
            patterns+=("*${d}/tcp_repr/${d}_${kind}.tar.zst")
        done
    fi
    echo "============================================================"

    rm -rf "${STAGE:?}/"*
    # Pull the per-defense tarballs into the staging dir, then unpack each.
    # `|| true` because tar exits non-zero when a pattern matches nothing; the
    # emptiness check below reports that properly.
    if [[ "$DEFENSE" == "all" ]]; then
        tar --zstd -xf "$src" -C "$STAGE" || true
    else
        tar --zstd -xf "$src" -C "$STAGE" --wildcards "${patterns[@]}" || true
    fi

    shopt -s nullglob
    for f in "$STAGE"/*/tcp_repr/*_"${kind}".tar.zst; do
        defense="$(basename "$f" "_${kind}.tar.zst")"
        dest="${DATASET_OUT}/${defense}/${kind}"
        mkdir -p "$dest"
        tar --zstd -xf "$f" -C "$dest"
        rm -f "$f"          # free the staged copy as we go
        echo "  ${defense} -> ${dest}"
        found=1
    done
    shopt -u nullglob

    if [[ "$found" -eq 0 ]]; then
        echo "  WARNING: nothing extracted for $kind"
        if [[ "$DEFENSE" != "all" ]]; then
            echo "  (check the defense names; run with 'all' to see what the archive has)"
        fi
    fi
    rm -rf "${STAGE:?}/"*
    echo
}

extract_overhead()
{
    local dataset_member
    # Find the exact member name, allowing either:
    #   4_udemy.tar.zst
    # or
    #   ./4_udemy.tar.zst
    dataset_member="$(
        tar --zstd -tf "$OVERHEAD_ARCHIVE" \
            | grep -E "^(\./)?${DATASET}\.tar\.zst$" \
            | head -n1 || true
    )"
    if [[ -z "$dataset_member" ]]; then
        echo "ERROR: overhead archive for dataset '$DATASET' not found"
        echo
        echo "Available overhead datasets:"
        tar --zstd -tf "$OVERHEAD_ARCHIVE"
        exit 1
    fi
    echo "============================================================"
    echo "Extracting overhead analysis"
    echo "  $dataset_member"
    echo "============================================================"
    mkdir -p "$DATASET_OUT"
    tar --zstd -xOf "$OVERHEAD_ARCHIVE" "$dataset_member" \
        | tar --zstd -xf - -C "$DATASET_OUT"
    echo "Extracted to: ${DATASET_OUT}/overhead"
    echo
}

# Dataset-specific overhead analysis
extract_overhead
# Evaluation datasets
extract_nested "deepsetraces"
extract_nested "wefdetraces"
# Optional benchmark results
if [[ "$WANT_BENCHMARKS" == "yes" ]]; then
    extract_nested "benchmarks"
fi
echo "============================================================"
echo "DONE"
echo "============================================================"
echo
echo "Workspace:"
echo "  $DATASET_OUT"
echo
echo "Contents:"
find "$DATASET_OUT" -mindepth 1 -maxdepth 2 -type d | sort | while read -r d; do
    printf "  %-56s %6s files\n" "$d" "$(find "$d" -type f | wc -l)"
done
