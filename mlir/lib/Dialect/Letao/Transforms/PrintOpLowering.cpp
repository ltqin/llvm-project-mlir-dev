//===- ConvertConst.cpp - Quantizes constant ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Letao/LetaoOps.h"
#include "mlir/Dialect/Letao/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::letao;

namespace {
struct PrintOpLoweringPass : public PrintOpLoweringBase<PrintOpLoweringPass> {
  void runOnFunction() override;
};

} // end anonymous namespace

void PrintOpLoweringPass::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](letao::PrintOp op) {
    auto loc = op.getLoc();
    auto operand = op.getOperand();

    OpBuilder builder(op.getOperation());
    mlir::IntegerType dataType = builder.getI32Type();
    auto printi32FuncOp =
        FuncOp::create(loc, "print_i32",
                       builder.getFunctionType({dataType}, {}));
    //module.push_back(printi32FuncOp);

    auto printOp = builder.create<CallOp>(
        loc, printi32FuncOp, ValueRange{operand});
    // must do that, so flow operation can use return value
    //op.output().replaceAllUsesWith(printOp);
    op.erase();
  });

}

std::unique_ptr<OperationPass<FuncOp>> mlir::letao::createPrintOpLoweringPass() {
  return std::make_unique<PrintOpLoweringPass>();
}
