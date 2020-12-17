// RUN: mlir-letao-driver -data_format=float -real_params=10,3,5,7.5  -lowering-to-llvm 2>&1 | mlir-cpu-runner -e main -entry-point-result=void  -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

// CHECK: 25.5