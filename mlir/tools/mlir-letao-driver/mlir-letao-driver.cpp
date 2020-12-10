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

// lowering to llvm
static cl::opt<bool> lowering_to_llvm("lowering-to-llvm", cl::desc("lower to llvm"),
                                   cl::value_desc("lowering-to-llvm"),
                                   cl::init(false));

static cl::opt<std::string> data_format("data_format",
                                         cl::desc("int, float, or bf16"),
                                         cl::value_desc("int, float, or bf16"),
                                         cl::init("int"));
  
static LogicalResult runMLIRPasses(ModuleOp &module,
                                   mlir::PassPipelineCLParser &passPipeline,
                                   StringRef kernelName) {
  PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  // Passes for lowering letao dialect.
  if (lowering_to_llvm.getValue()) {
    pm.addPass(mlir::letao::createMultiAddTransPass());
    pm.addPass(mlir::createLowerToLLVMPass());
  }
  // Build the provided pipeline.
  if (failed(passPipeline.addToPipeline(pm)))
    return failure();

  return pm.run(module);
}
template<class T,class ConstantTypeOp,class CastType>
SmallString<128> createSource(ModuleOp &module, OpBuilder &builder,T& dataType) {
 // auto dataType = (1=1?builder.getI32Type():builder.getF32Type());
  
  auto printi32FuncOp =
        FuncOp::create(builder.getUnknownLoc(), "print_i32",
                       builder.getFunctionType({dataType}, {}));
  module.push_back(printi32FuncOp);

  auto printnewlineFuncOp =
        FuncOp::create(builder.getUnknownLoc(), "print_newline",
                       builder.getFunctionType({}, {}));
  module.push_back(printnewlineFuncOp);
  // test_multiaddy
  SmallString<128> kernelName;
  kernelName = "test_multiadd";
  auto funcType = builder.getFunctionType(
      {dataType, dataType, dataType, dataType}, {dataType});

  auto func = FuncOp::create(builder.getUnknownLoc(), kernelName, funcType);
  module.push_back(func);

  Block *block = func.addEntryBlock();

  auto args = block->getArguments();
  auto multiAddOp = builder.create<letao::MultiAddOp>(
      builder.getUnknownLoc(), dataType, ValueRange(args));
  block->push_back(multiAddOp);

  auto returnOp =
      builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{multiAddOp});
  block->push_back(returnOp);

  //main
  {
    auto mainType = builder.getFunctionType({}, {});
    auto main = FuncOp::create(builder.getUnknownLoc(), "main", mainType);
    module.push_back(main);
    Block *mainBlock = main.addEntryBlock();
 
    auto addConstantI32_1 = builder.create<ConstantTypeOp>(
        builder.getUnknownLoc(), CastType((float)1), dataType);

    auto addConstantI32_2 = builder.create<ConstantTypeOp>(
        builder.getUnknownLoc(), CastType((float)10), dataType);

    auto addConstantI32_3 = builder.create<ConstantTypeOp>(
        builder.getUnknownLoc(), CastType((float)20), dataType);

    auto addConstantI32_4 = builder.create<ConstantTypeOp>(
        builder.getUnknownLoc(), CastType((float)30), dataType);

    mainBlock->push_back(addConstantI32_1);
    mainBlock->push_back(addConstantI32_2);
    mainBlock->push_back(addConstantI32_3);
    mainBlock->push_back(addConstantI32_4);

    auto calltestOp =
        builder.create<CallOp>(builder.getUnknownLoc(), func,
                               ValueRange{addConstantI32_1, addConstantI32_2,
                                          addConstantI32_3, addConstantI32_4});
    mainBlock->push_back(calltestOp);

    auto printOp = builder.create<CallOp>(builder.getUnknownLoc(),
                                          printi32FuncOp, ValueRange(calltestOp.getResults()));
    mainBlock->push_back(printOp);
    
    auto printNewLineOp = builder.create<CallOp>(builder.getUnknownLoc(),
                                          printnewlineFuncOp, ValueRange{});
    mainBlock->push_back(printNewLineOp);

    auto mainReturnOp =
        builder.create<ReturnOp>(builder.getUnknownLoc(), ValueRange{});
    mainBlock->push_back(mainReturnOp);

  }
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

  SmallString<128> kernelName;
  if (data_format == "int") {
    auto dataType = builder.getI64Type();
    kernelName = createSource<decltype(dataType), ConstantIntOp, int64_t>(
        module, builder, dataType);
  } else if (data_format == "float") {
    auto dataType = builder.getF32Type();
    kernelName = createSource<decltype(dataType), ConstantFloatOp, APFloat>(
        module, builder, dataType);
  } else {
    llvm::errs() << "data_format error!" << "\n";
    exit(1);
  }

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
