// RUN: mlir-rocm-runner %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @vecadd(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xi16>) {
  %cst = constant 1 : index
  %cst2 = dim %arg0, 0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst) {
    %a = load %arg0[%tx] : memref<?xf32>
    %b = load %arg1[%tx] : memref<?xf32>
    %c = addf %a, %b : f32
   
    %_c0 = llvm.mlir.constant(16: i32) :!llvm.i32 
    %d = llvm.mlir.cast %c : f32 to !llvm.float
    %e = llvm.bitcast %d : !llvm.float to !llvm.i32
    %f = llvm.lshr %e, %_c0 : !llvm.i32
    %g = llvm.trunc %f : !llvm.i32 to !llvm.i16
    %h = llvm.mlir.cast %g : !llvm.i16 to i16
    store %h, %arg2[%tx] : memref<?xi16>
    gpu.terminator
  }
  return
}

// CHECK: [2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46]
func @main() {
  // allocate CPU memory.
  %0 = alloc() : memref<16xf32>
  %1 = alloc() : memref<16xf32>
  %2 = alloc() : memref<16xi16>

  %3 = memref_cast %0 : memref<16xf32> to memref<?xf32>
  %4 = memref_cast %1 : memref<16xf32> to memref<?xf32>
  %5 = memref_cast %2 : memref<16xi16> to memref<?xi16>

  // populate initial values.
  %cst = constant 1.6 : f32
  %cst0 = constant 1 : i16
  call @mcpuMemset(%3, %cst) : (memref<?xf32>, f32) -> ()
  call @mcpuMemset(%4, %cst) : (memref<?xf32>, f32) -> ()
  call @mcpuMemsetI16(%5, %cst0) : (memref<?xi16>, i16) -> ()

  // allocate GPU memory.
  %6 = call @mgpuMemAlloc(%3) : (memref<?xf32>) -> (memref<?xf32>)
  %7 = call @mgpuMemAlloc(%4) : (memref<?xf32>) -> (memref<?xf32>)
  %8 = call @mgpuMemAllocI16(%5) : (memref<?xi16>) -> (memref<?xi16>)

  // copy direction constants.
  %cst_h2d = constant 1 : i32
  %cst_d2h = constant 2 : i32

  // transfer data CPU -> GPU.
  call @mgpuMemCopy(%3, %6, %cst_h2d) : (memref<?xf32>, memref<?xf32>, i32) -> ()
  call @mgpuMemCopy(%4, %7, %cst_h2d) : (memref<?xf32>, memref<?xf32>, i32) -> ()

  // launch kernel.
  call @vecadd(%6, %7, %8) : (memref<?xf32>, memref<?xf32>, memref<?xi16>) -> ()

  // transfer data GPU -> CPU.
  call @mgpuMemCopyI16(%8, %5, %cst_d2h) : (memref<?xi16>, memref<?xi16>, i32) -> ()

  // print result.
  %9 = memref_cast %5 : memref<?xi16> to memref<*xi16>
  call @print_memref_bf16(%9) : (memref<*xi16>) -> ()

  // dellocate GPU memory.
  call @mgpuMemDealloc(%6) : (memref<?xf32>) -> ()
  call @mgpuMemDealloc(%7) : (memref<?xf32>) -> ()
  //call @mgpuMemDealloc(%8) : (memref<?xi16>) -> ()

  // deallocate CPU memory.
  dealloc %0 : memref<16xf32>
  dealloc %1 : memref<16xf32>
  dealloc %2 : memref<16xi16>

  return
}

func @mcpuMemset(%ptr : memref<?xf32>, %value: f32) -> ()
func @mcpuMemsetI16(%ptr : memref<?xi16>, %value: i16) -> ()
func @mgpuMemAlloc(%ptr : memref<?xf32>) -> (memref<?xf32>)
func @mgpuMemAllocI16(%ptr : memref<?xi16>) -> (memref<?xi16>)
func @mgpuMemDealloc(%ptr : memref<?xf32>) -> ()
func @mgpuMemCopy(%src : memref<?xf32>, %dst : memref<?xf32>, %dir : i32) -> ()
func @mgpuMemCopyI16(%src : memref<?xi16>, %dst : memref<?xi16>, %dir : i32) -> ()
func @print_memref_f32(%ptr : memref<*xf32>) -> ()
func @print_memref_bf16(%ptr : memref<*xi16>) -> ()
