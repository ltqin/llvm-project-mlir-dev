add_mlir_dialect(LetaoOps letao)
add_mlir_doc(LetaoOps -gen-dialect-doc LetaoDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRLetaoPassIncGen)
add_dependencies(mlir-headers MLIRLetaoPassIncGen)

add_mlir_doc(Passes -gen-pass-doc LetaoPasses ./)