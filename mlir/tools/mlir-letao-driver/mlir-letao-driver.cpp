//===- mlir-cpu-runner.cpp - MLIR CPU Execution Driver---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by  translating MLIR to LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

//#include "mlir/Dialect/Letao/LetaoOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));
// lowering to llvm.
static cl::opt<bool> loweringToLLVM("lowering_to_llvm",
                                    cl::desc("lower to llvm"),
                                    cl::value_desc("lower to llvm"),
                                    cl::init(false));

// multAddtoAdds
static cl::opt<bool> multAddtoAdds("multadd_to_adds", cl::desc("lower to llvm"),
                                   cl::value_desc("lower to llvm"),
                                   cl::init(false));
static LogicalResult runMLIRPasses(ModuleOp &module,
                                   mlir::PassPipelineCLParser &passPipeline,
                                   StringRef kernelName) {
  PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  // Passes for lowering letao dialect.
  if (multAddtoAdds.getValue()) {
    pm.addPass(mlir::letao::createMultiAddTransPass());
  }
  // pm.addPass(mlir::createConvertLinalgToLLVMPass());
  if (loweringToLLVM.getValue())
    pm.addPass(mlir::createLowerToLLVMPass());

  return pm.run(module);
}

SmallString<128> createSource(ModuleOp &module, OpBuilder &builder) {
  mlir::IntegerType dataType = builder.getI32Type();
  auto funcType = builder.getFunctionType({}, {});
  SmallString<128> kernelName;
  kernelName = "test_multiadd";
  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, funcType);
  module.push_back(func);

  Block *block = func.addEntryBlock();

  auto addConstantI32_1 = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 1, builder.getIntegerType(32));

  auto addConstantI32_2 = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 10, builder.getIntegerType(32));

  auto addConstantI32_3 = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 20, builder.getIntegerType(32));

  auto addConstantI32_4 = builder.create<ConstantIntOp>(
      builder.getUnknownLoc(), 30, builder.getIntegerType(32));

  block->push_back(addConstantI32_1);
  block->push_back(addConstantI32_2);
  block->push_back(addConstantI32_3);
  block->push_back(addConstantI32_4);

  auto multiAddOp = builder.create<letao::MultiAddOp>(
      builder.getUnknownLoc(),dataType,
      ValueRange{addConstantI32_1, addConstantI32_2,
                 addConstantI32_3, addConstantI32_4});

  block->push_back(multiAddOp);

  auto printi32FuncOp = FuncOp::create(
      builder.getUnknownLoc(), "print_i32",
      builder.getFunctionType(
          {dataType}, {}));
  module.push_back(printi32FuncOp);

  auto printOp =
      builder.create<CallOp>(builder.getUnknownLoc(), printi32FuncOp,
                             ValueRange{multiAddOp});
  block->push_back(printOp);

  auto returnOp = builder.create<ReturnOp>(
      builder.getUnknownLoc(),
      ValueRange{});
  block->push_back(returnOp);
  return kernelName;
}
int main(int argc, char **argv) {

  mlir::registerAllDialects();
  mlir::registerAllPasses();

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR letao Dialect driver\n");

  MLIRContext context;
  OpBuilder builder(&context);

  std::string errorMessage;
  SourceMgr sourceMgr;
  OwningModuleRef moduleRef;
  ModuleOp module = ModuleOp::create(builder.getUnknownLoc());

  SmallString<128> kernelName = createSource(module, builder);

  if (failed(runMLIRPasses(module, passPipeline, kernelName))) {
    llvm::errs() << "Lowering failed.\n";
    exit(1);
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // llvm::errs() << "output."<<output->os()<<"\n";

  module.print(output->os());
  output->keep();

  return 0;
}
