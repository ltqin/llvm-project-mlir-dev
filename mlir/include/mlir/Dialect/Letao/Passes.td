//===-- Passes.td - MIOpen pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LETAO_PASSES
#define MLIR_DIALECT_LETAO_PASSES

include "mlir/Pass/PassBase.td"


def MultAddTransToAdds : FunctionPass<"multiadd_to_adds"> {
  let summary = "multiaddop translate to some adds op";
  let constructor = "mlir::letao::createMultiAddTransPass()";
}

def PrintOpLowering : FunctionPass<"printop_lowering"> {
  let summary = "lowering printOp";
  let constructor = "mlir::letao::createPrintOpLoweringPass()";
}

#endif // MLIR_DIALECT_MIOPEN_PASSES
