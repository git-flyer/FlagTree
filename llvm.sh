export MAX_JOBS=6
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
export LLVM_SYSPATH=/public/home/yangcm/ph/llvm-7d5de303-ubuntu-x64
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
