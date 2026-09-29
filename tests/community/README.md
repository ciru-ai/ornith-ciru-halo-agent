# Community kernel qualification helpers

The retained reports describe the exact September 29 qualification. These GPU
scripts compare staged control libraries with a native build using the pinned
ROCm/PyTorch runtime. Set CIRU_SCREEN_ROOT to a prepared staging directory with
source/, native-built/, and ornith-baseline/bundle/native/. Copy these helpers
to that directory before invoking verify_retained_native.py. Native rebuild
instructions are in BUILD.md. The serving protocol and scoped results are in
evaluation/. These checks are not broad model-quality benchmarks.
