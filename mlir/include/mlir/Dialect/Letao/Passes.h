//===- Passes.h - Letao Passes ------ --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the passes owned by the quantization dialect. As
// things mature, it is expected that passes specific to certain frontend or
// backend dialects will move to those dialects directly. For now, they are
// incubated here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LETAO_PASSES_H
#define MLIR_DIALECT_LETAO_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace letao {

std::unique_ptr<OperationPass<FuncOp>> createMultiAddTransPass();
std::unique_ptr<OperationPass<FuncOp>> createPrintOpLoweringPass();
} // namespace quant
} // namespace mlir

#endif // MLIR_DIALECT_QUANT_PASSES_H
