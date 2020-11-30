set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  )

add_llvm_tool(mlir-letao-driver
  mlir-letao-driver.cpp
  )
llvm_update_compile_flags(mlir-letao-driver)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(mlir-letao-driver PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMIR
  MLIRParser
  MLIRTargetLLVMIR
  MLIRSupport
  )
