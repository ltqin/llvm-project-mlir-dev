// RUN: mlir-opt -multiadd-to-adds %s | FileCheck %s

func @test_multiadd() {
  %input1 = constant 1 : i32
  %input2 = constant 2 : i32
  %input3 = constant 3 : i32
  %input4 = constant 4 : i32
  %output = letao.multiadd(%input1, %input2, %input3, %input4) : i32,i32,i32,i32 to i32
  return
}

// CHECK-LABEL: func @test_multiadd
// CHECK: addi
// CHECK-NEXT: addi
// CHECK-NEXT: addi
