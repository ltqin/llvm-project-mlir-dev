// RUN: mlir-letao-driver -data_format=float 2>&1 | FileCheck %s

// CHECK: letao.multiadd
// CHECK: f32