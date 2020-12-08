//===- LetaoOps.cpp - Letao MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Letao/LetaoOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::letao;
//===----------------------------------------------------------------------===//
// LetaoDialect
//===----------------------------------------------------------------------===//
LetaoDialect::LetaoDialect(::mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Letao/LetaoOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DotOp
//===----------------------------------------------------------------------===//
static ParseResult parseDotOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  SmallVector<Type, 2> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      // parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}
static void print(OpAsmPrinter &p, DotOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(DotOp op) { return success(); }
//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

static ParseResult parseConv2DOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, Conv2DOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(Conv2DOp op) { return success(); }

//===----------------------------------------------------------------------===//
// MovePosOp
//===----------------------------------------------------------------------===//

static ParseResult parseMovePosOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  Type srcType;

  auto ret = parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
             parser.parseColonType(srcType) ||
             parser.resolveOperand(ops[0], srcType, result.operands);

  for (unsigned i = 1; i < ops.size(); ++i) {
    ret &= succeeded(parser.resolveOperand(
        ops[i], srcType.cast<MemRefType>().getElementType(), result.operands));
  }
  return failure(ret);
}

static void print(OpAsmPrinter &p, MovePosOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p << " : " << op.getOperands()[0].getType();
}

static LogicalResult verify(MovePosOp op) {
  auto operands = op.getOperands();
  if (operands.size() < 3)
    return success(false);

  return success();
}

//===----------------------------------------------------------------------===//
// MultiAddOp
//===----------------------------------------------------------------------===//

static ParseResult parseMultiAddOp(OpAsmParser &parser,
                                   OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  Type retType;
  auto ret = parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
             parser.parseColonTypeList(types) ||
             parser.resolveOperands(ops, types, parser.getNameLoc(),
                                    result.operands) ||
             parser.parseKeywordType("to", retType) ||
             parser.addTypeToList(retType, result.types);

  return failure(ret);
}

static void print(OpAsmPrinter &p, MultiAddOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p << " : " << op.getOperandTypes();
  p << " to " << op.getType();
}

static LogicalResult verify(MultiAddOp op) {
  auto operands = op.getOperands();
  if (operands.size() < 2) {
    llvm::errs() << "MultiAddOp error:parameters should more than 1."
                 << "\n";
    return success(false);
  }

  for (unsigned i = 0; i < operands.size(); i++) {
    if (!(operands[i].getType() == op.getType())) {
      llvm::errs() << "MultiAddOp error:input type different from output type"
                   << "\n";
      return success(false);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

static ParseResult parsePrintOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType ops;
  Type type;
  auto ret = parser.parseOperand(ops) || parser.parseColonType(type) ||
             parser.resolveOperand(ops, type, result.operands);

  return failure(ret);
}

static void print(OpAsmPrinter &p, PrintOp op) {
  p << op.getOperationName() << " " << op.getOperand() << " ";
  p << " : " << op.getOperand().getType();
}

static LogicalResult verify(PrintOp op) { return success(); }
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Letao/LetaoOps.cpp.inc"
