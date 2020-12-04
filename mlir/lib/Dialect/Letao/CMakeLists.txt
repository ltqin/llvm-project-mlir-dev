file(GLOB globbed *.c *.cpp Transforms/*.cpp)
add_mlir_dialect_library(MLIRLetaoOps
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Letao

  DEPENDS
  MLIRLetaoOpsIncGen
  )

target_link_libraries(MLIRLetaoOps
  PUBLIC
  MLIRIR
  MLIRPass
  LLVMSupport
  mlir_c_runner_utils_static
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRLLVMIR
  MLIRParser
  MLIRTargetLLVMIR
  MLIRSupport
  )