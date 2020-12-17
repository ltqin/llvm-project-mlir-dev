// RUN: mlir-letao-driver -data_format=int -real_params=10,3,5,7  -lowering-to-llvm 2>&1 | mlir-cpu-runner -e main -entry-point-result=void  -shared-libs=./build/lib/libmlir_c_runner_utils.so | FileCheck %s

// CHECK: 25