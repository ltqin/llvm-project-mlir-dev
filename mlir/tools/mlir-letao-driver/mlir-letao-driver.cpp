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
#include "mlir/IR/Value.h"
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
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

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
static cl::opt<std::string> real_params("real_params",
                                         cl::desc("the parameters for add,eg: 1,10,20 mean sum(1+20+30)"),
                                         cl::value_desc("1,20,30 mean sum(1+20+30)"),
                                         cl::init("1,20,30"));

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
std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}
template<class DataType,class ConstantTypeOp,class RealType,class CastType>
SmallString<128> createSource(ModuleOp &module, OpBuilder &builder,DataType& dataType) {
  auto printi32FuncOp =
        FuncOp::create(builder.getUnknownLoc(), "print_i32",
                       builder.getFunctionType({dataType}, {}));
  module.push_back(printi32FuncOp);

  auto printf32FuncOp =
        FuncOp::create(builder.getUnknownLoc(), "print_f32",
                       builder.getFunctionType({dataType}, {}));
  module.push_back(printf32FuncOp);

  auto printnewlineFuncOp =
        FuncOp::create(builder.getUnknownLoc(), "print_newline",
                       builder.getFunctionType({}, {}));
  module.push_back(printnewlineFuncOp);

  //split parameters
  std::vector<std::string> parameters = split(real_params, ',');
  if(parameters.size() < 2){
    llvm::errs() << "real_params error!" << "\n";
    exit(1);
  }
  // test_multiaddy
  SmallString<128> kernelName;
  kernelName = "test_multiadd";
  llvm::SmallVector<mlir::Type,4> types;
  for(unsigned i = 0; i < parameters.size(); i++){
    types.push_back(dataType);
  }
  auto funcType = builder.getFunctionType(
      types, {dataType});

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
    llvm::SmallVector<mlir::Value,4> values;
    for(unsigned i = 0; i < parameters.size(); i++){
      auto constantData = builder.create<ConstantTypeOp>(
        builder.getUnknownLoc(), RealType((CastType)std::atof(parameters[i].c_str())), dataType);
      values.push_back(constantData);
      mainBlock->push_back(constantData);
    }
    auto calltestOp =
        builder.create<CallOp>(builder.getUnknownLoc(), func,
                               values);
                                   
    mainBlock->push_back(calltestOp);
    
    //print
    bool bIsInteger = ((mlir::Type)dataType).isa<mlir::IntegerType>();
    if (bIsInteger) {
      auto printOp =
          builder.create<CallOp>(builder.getUnknownLoc(), printi32FuncOp,
                                 ValueRange(calltestOp.getResults()));
      mainBlock->push_back(printOp);
    } else {
      auto printOp =
          builder.create<CallOp>(builder.getUnknownLoc(), printf32FuncOp,
                                 ValueRange(calltestOp.getResults()));
      mainBlock->push_back(printOp);
    }

    auto printNewLineOp = builder.create<CallOp>(builder.getUnknownLoc(),
                                          printnewlineFuncOp, ValueRange{});
    mainBlock->push_back(printNewLineOp);
    //return
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
    auto dataType = builder.getI32Type();
    kernelName = createSource<decltype(dataType), ConstantIntOp, int64_t, int>(
        module, builder, dataType);
  } else if (data_format == "float") {
    auto dataType = builder.getF32Type();
    kernelName = createSource<decltype(dataType), ConstantFloatOp, APFloat, float>(
        module, builder, dataType);
  } else if (data_format == "bf16") {
    auto dataType = builder.getBF16Type();
    kernelName = createSource<decltype(dataType), ConstantFloatOp, APFloat, float>(
        module, builder, dataType);
  } 
  else {
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
