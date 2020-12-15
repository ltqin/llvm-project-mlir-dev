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
struct MultAddTransToAdds : public MultAddTransToAddsBase<MultAddTransToAdds> {
  void runOnFunction() override;
private:
  template <class ADDOP> void MultAddTransToAddsImpl(letao::MultiAddOp &op);
};

} // end anonymous namespace
template <class ADDOP> 
void MultAddTransToAdds::MultAddTransToAddsImpl(letao::MultiAddOp &op) {
  auto loc = op.getLoc();
  auto operands = op.getOperands();
  OpBuilder b(op.getOperation());
  Value add;

  add = b.create<ADDOP>(loc, op.getOperand(0), op.getOperand(1));
  for (unsigned i = 2; i < operands.size(); i++) {
    add = b.create<ADDOP>(loc, add, op.getOperand(i));
  }
  // must do that, so flow operation can use return value
  op.output().replaceAllUsesWith(add);
  op.erase();
}
void MultAddTransToAdds::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](letao::MultiAddOp op) {
    bool bIsInteger = op.getType().isa<IntegerType>();

    if (bIsInteger)
      MultAddTransToAddsImpl<AddIOp>(op);
    else
      MultAddTransToAddsImpl<AddFOp>(op);
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::letao::createMultiAddTransPass() {
  return std::make_unique<MultAddTransToAdds>();
}
