#!/usr/bin/env bash
# Copyright 2026 Ciru. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime=''
out=''
check=0
extra=()
usage() {
    echo 'Usage: scripts/build-native.sh --runtime PATH --out PATH [--check] [-- HIPCC_ARGUMENT ...]'
    echo 'Build the eight gfx1151 native libraries using the installed release runtime.'
    echo 'HIPCC/ROCM_PATH may select a standalone ROCm SDK. --check performs CPU preflight only.'
}
while (($#)); do
    case "$1" in
        --runtime) runtime=${2:?missing runtime directory}; shift 2 ;;
        --out) out=${2:?missing output directory}; shift 2 ;;
        --check) check=1; shift ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$runtime" && -n "$out" ]] || { usage >&2; exit 2; }
runtime=$(cd -- "$runtime" && pwd)
python="$runtime/venv/bin/python"
[[ -x "$python" ]] || { echo "Missing runtime Python: $python" >&2; exit 2; }
site_packages=$("$python" -c 'import site; print(site.getsitepackages()[0])')
if [[ -z ${ROCM_PATH:-} ]]; then
    for candidate in "$site_packages/_rocm_sdk_devel" "$site_packages/_rocm_sdk_core"; do
        if [[ -x "$candidate/bin/hipcc" ]]; then export ROCM_PATH="$candidate"; break; fi
    done
fi
: "${ROCM_PATH:?Set ROCM_PATH to a ROCm SDK with gfx1151 support}"
hipcc=${HIPCC:-$ROCM_PATH/bin/hipcc}
[[ -x "$hipcc" ]] || { echo "Missing HIP compiler: $hipcc" >&2; exit 2; }
export HIP_PATH="$ROCM_PATH"
export PATH="$runtime/venv/bin:$ROCM_PATH/bin:$PATH"
if [[ -d "$ROCM_PATH/lib/llvm/bin" ]]; then
    export HIP_CLANG_PATH="$ROCM_PATH/lib/llvm/bin"
else
    export HIP_CLANG_PATH="$ROCM_PATH/llvm/bin"
fi
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$site_packages/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
common=(-x hip -std=c++20 -O2 -ffp-contract=off
        -fhip-fp32-correctly-rounded-divide-sqrt -shared -fPIC -fvisibility=hidden
        --offload-arch=gfx1151 -gline-tables-only -DORNITH_PREFILL_OUTPUT32=1)
# On NixOS the HIP compiler cannot discover the host GCC/glibc by itself.
# Derive locations from the selected compiler instead of shipping store hashes.
host_flags=()
host_cxx=${CXX:-g++}
if command -v "$host_cxx" >/dev/null 2>&1; then
    gcc_include=$("$host_cxx" -print-file-name=include)
    if [[ "$gcc_include" == /nix/store/*/lib/gcc/* ]]; then
        gcc_root=${gcc_include%%/lib/gcc/*}
        host_flags+=("--gcc-toolchain=$gcc_root")
        while IFS= read -r include_dir; do
            if [[ "$include_dir" == /nix/store/*glibc*-dev/include ]]; then
                host_flags+=(-idirafter "$include_dir")
            fi
        done < <("$host_cxx" -E -x c++ -v /dev/null 2>&1 | sed -n '/search starts here:/,/End of search list./s/^ //p')
        for object in crti.o libstdc++.so libgcc_s.so; do
            object_path=$("$host_cxx" -print-file-name="$object")
            if [[ "$object_path" == /* && -f "$object_path" ]]; then
                object_dir=$(dirname -- "$object_path")
                host_flags+=("-B$object_dir" "-L$object_dir")
            fi
        done
    fi
fi
# Keep these component directories distinct: several ABI headers share a name
# but belong to different measured implementations.
sources=(dense/dense_g256_consumer.cpp routed/routed_direct_consumer.cpp
         head/head_i8_tile_consumer.cpp routed_n32/routed_n32_consumer.cpp
         routed_storage/routed_storage_n32_consumer.cpp attention/prefill.cpp
         attention/persistent.cpp dense_n32/dense_g256_consumer.cpp)
names=(dense_g256 routed_direct head_i8_tile routed_n32 routed_storage_n32
       attention_iu4 persistent_iu4 dense_g256_n32)
for i in "${!sources[@]}"; do
    [[ -f "$root/kernels/${sources[$i]}" ]] || { echo "Missing source: ${sources[$i]}" >&2; exit 2; }
    [[ ! -e "$out/libornith_${names[$i]}.so" ]] || { echo "Choose a fresh output directory; library already exists: ${names[$i]}" >&2; exit 2; }
done
"$hipcc" --version
if ((check)); then
    echo 'CPU preflight passed: compiler found, eight source targets present, output files clear.'
    echo 'No GPU work, compilation, or model startup was performed.'
    exit 0
fi
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd)
"$hipcc" --version > "$out/compiler-version.txt"
for i in "${!sources[@]}"; do
    source_file="$root/kernels/${sources[$i]}"
    library="$out/libornith_${names[$i]}.so"
    command=("$hipcc" "${common[@]}" "${host_flags[@]}" -I "$(dirname -- "$source_file")"
             "${extra[@]}" "$source_file" -o "$library")
    printf '%q ' "${command[@]}" > "$library.command.txt"
    printf '\n' >> "$library.command.txt"
    echo "Building $(basename -- "$library")"
    "${command[@]}"
done
echo "Built ${#names[@]} native libraries in $out"
