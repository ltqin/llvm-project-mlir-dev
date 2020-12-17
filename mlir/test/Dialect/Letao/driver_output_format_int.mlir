// RUN: mlir-letao-driver -data_format=int 2>&1 | FileCheck %s

// CHECK: letao.multiadd
// CHECK: i32