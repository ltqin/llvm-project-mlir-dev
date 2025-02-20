//===- MIOpenOps.td - MIOpen operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIOpen operations.
//
//===----------------------------------------------------------------------===//

#ifndef MIOPEN_OPS
#define MIOPEN_OPS

include "mlir/IR/OpBase.td"
//include "mlir/Transforms/LoopLikeInterface.td"

def MIOpen_Dialect : Dialect {
  let name = "miopen";
  let cppNamespace = "miopen";
}

// Base class for MIOpen dialect ops.
class MIOpen_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MIOpen_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
  }];
}

def MIOpen_Conv2DOp :
    MIOpen_Op<"conv2d">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [4]>:$filter,
                   MemRefRankOf<[F32, F16, BF16], [4]>:$input,
                   MemRefRankOf<[F32, F16, BF16], [4]>:$output)> {
  let summary = "2D convolution forward";
  let description = [{
    The `miopen.conv2d` op computes 2D convolution forward.
  }];
}

def MIOpen_Conv2DBwdDataOp :
    MIOpen_Op<"conv2d_bwd_data">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [4]>:$filter,
                   MemRefRankOf<[F32, F16, BF16], [4]>:$input,
                   MemRefRankOf<[F32, F16, BF16], [4]>:$output)> {
  let summary = "2D convolution backward data";
  let description = [{
    The `miopen.conv2d` op computes 2D convolution backward data.
  }];
}

def MIOpen_Conv2DBwdWeightOp :
    MIOpen_Op<"conv2d_bwd_weight">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [4]>:$filter,
                   MemRefRankOf<[F32, F16, BF16], [4]>:$input,
                   MemRefRankOf<[F32, F16, BF16], [4]>:$output)> {
  let summary = "2D convolution backward weight";
  let description = [{
    The `miopen.conv2d` op computes 2D convolution backward weight.
  }];
}

def MIOpen_TransformOp :
    MIOpen_Op<"transform">,
    Arguments<(ins AnyMemRef:$input)>,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Tensor transformation";
  let description = [{
    The `miopen.transform` op transforms tensor coordinates.
  }];
}

def MIOpen_GridwiseGemmOp :
    MIOpen_Op<"gridwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [2]>:$filter,
                   MemRefRankOf<[F32, F16, BF16], [2]>:$input,
                   MemRefRankOf<[F32, F16, BF16], [2]>:$output)> {
  let summary = "Gridwise GEMM";
  let description = [{
    The `miopen.gridwise_gemm` op computes gridwise GEMM.
  }];
}

// gridwise_gemm_v2
def MIOpen_GridwiseGemmV2Op :
    MIOpen_Op<"gridwise_gemm_v2">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [2]>:$filter,
                   MemRefRankOf<[F32, F16, BF16], [2]>:$input,
                   MemRefRankOf<[F32, F16, BF16], [2]>:$output)> {
  let summary = "Gridwise GEMM V2";
  let description = [{
    The `miopen.gridwise_gemm` op computes gridwise GEMM with XDLOPS.
  }];
}

// Memory allocation on GPU memory hierachy.
def MIOpen_GpuAllocOp:
    MIOpen_Op<"alloc">,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Memory allocation on GPU";
  let description = [{
    The `miopen.alloc` op allocates memory on GPU.
    - Address space 0 : global.
    - Address space 3 : LDS.
    - Address space 5 : private (VGPR).
    All other values would be considered as allocation on global.
  }];
}

// subview provides a subsection view of a memref.
def MIOpen_SubviewOp:
    MIOpen_Op<"subview">,
    Arguments<(ins AnyMemRef:$input,
                   Index:$offset)>,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Subview of memref on GPU";
  let description = [{
    The `miopen.subview` op creates a subview of a memref on GPU.
    The memref must be on either LDS (address space 3) or private (address
    space 5).
  }];
}

// TBD: eventually replace this with linalg.fill?
def MIOpen_FillOp:
    MIOpen_Op<"fill">,
    Arguments<(ins AnyMemRef:$input,
                   AnyTypeOf<[AnyInteger, AnyFloat]>:$value)> {
  let summary = "Fill memory with constant value on GPU";
  let description = [{
    The `miopen.fill` op fills a memref on GPU with a constant value.
  }];
}

def MIOpen_MovePosOp:
    MIOpen_Op<"move_pos">,
    Arguments<(ins AnyMemRef:$memref,
                   Variadic<AnyTypeOf<[AnyInteger, AnyFloat]>>:$values)> {
    let summary = "Add values to indices within the memref.";
    let description = [{
      The `miopen.fill` op adds values to indices within the memref.
    }];
}

def MIOpen_WorkgroupBarrierOp:
    MIOpen_Op<"workgroup_barrier"> {
  let summary = "Setup an workgroup barrier";
  let description = [{
    The `miopen.workgroup_barrier` op sets up a workgroup-level barrier.
  }];
}

def MIOpen_WorkgroupIdOp:
    MIOpen_Op<"workgroup_id">,
    Results<(outs Index:$id)> {
  let summary = "Get current workgroup ID";
  let description = [{
    The `miopen.workgroup_id` op gets the current workgroup ID.
  }];
}

def MIOpen_WorkitemIdOp:
    MIOpen_Op<"workitem_id">,
    Results<(outs Index:$id)> {
  let summary = "Get current workitem ID";
  let description = [{
    The `miopen.workgroup_id` op gets the current workitem ID.
  }];
}

// blockwise_copy
def MIOpen_BlockwiseCopyOp:
    MIOpen_Op<"blockwise_copy">,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest,
                   AnyMemRef:$sourceCoord,
                   AnyMemRef:$destCoord,
                   Optional<AnyMemRef>:$buffer)> {
  let summary = "Blockwise GPU data movement";
  let description = [{
    The `miopen.blockwise` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to LDS (naive tensor).
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to LDS (naive tensor).
  }];
}

// threadwise_copy
def MIOpen_ThreadwiseCopyOp:
    MIOpen_Op<"threadwise_copy">,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest,
                   Variadic<I32>:$sourceAndDestCoord)> {
  let summary = "Threadwise GPU data movement";
  let description = [{
    The `miopen.threadwise` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to LDS (naive tensor).
    - LDS (naive tensor) to register (naive tensor).
    - Register (naive tensor) to global (generic tensor).
  }];
}

// threadwise_copy_v2
def MIOpen_ThreadwiseCopyV2Op:
    MIOpen_Op<"threadwise_copy_v2">,
    Arguments<(ins VectorOfRankAndType<[1], [F32, F16, BF16]>:$source,
                   AnyMemRef:$dest,
                   I32: $sourceOffset,
                   Variadic<I32>:$sourceAndDestCoord)> {
  let summary = "Threadwise GPU data movement XDLOPS version";
  let description = [{
    The `miopen.threadwise_v2` op moves data on GPU. Following movements are
    allowed:
    - Register (naive tensor) as vector type to global (generic tensor).

    The op is used to store C matrices, stored as vectors, back to global VMEM.
  }];
}

// blockwise_gemm
def MIOpen_BlockwiseGemmOp:
    MIOpen_Op<"blockwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [2]>:$matrixA,
                   MemRefRankOf<[F32, F16, BF16], [2]>:$matrixB,
                   AnyMemRef:$matrixC,
                   Index:$threadOffsetA,
                   Index:$threadOffsetB)> {
  let summary = "Blockwise GEMM non-XDLOPS version";
  let description = [{
    The `miopen.block_gemm` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on LDS (naive tensor).
    - Matrix C shall reside on register (naive tensor).
  }];
}

// blockwise_gemm_v2
def MIOpen_BlockwiseGemmV2Op:
    MIOpen_Op<"blockwise_gemm_v2">,
    Arguments<(ins MemRefOf<[F32, F16, BF16]>:$matrixA,
                   MemRefOf<[F32, F16, BF16]>:$matrixB,
                   Index:$threadOffsetA,
                   Index:$threadOffsetB,
                   MemRefOf<[F32, F16, BF16]>:$bufferA,
                   MemRefOf<[F32, F16, BF16]>:$bufferB,
                   Variadic<VectorOfRankAndType<[1], [F32, F16, BF16]>>:$vectorCs)>,
    Results<(outs Variadic<VectorOfRankAndType<[1], [F32, F16, BF16]>>: $vectorDs)> {
  let summary = "Blockwise GEMM XDLOPS version";
  let description = [{
    The `miopen.block_gemm` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on LDS (naive tensor).
    - Matrix C shall be vectors.
  }];
}

// threadwise_gemm
def MIOpen_ThreadwiseGemmOp:
    MIOpen_Op<"threadwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [2]>:$matrixA,
                   MemRefRankOf<[F32, F16, BF16], [2]>:$matrixB,
                   MemRefRankOf<[F32, F16, BF16], [2]>:$matrixC)> {
  let summary = "Threadwise GEMM non-XDLOPS version";
  let description = [{
    The `miopen.threadwise_gemm` op does GEMM at thread level.
    - Matrix A and Matrix B shall reside on register (naive tensor).
    - Matrix C shall reside on LDS (naive tensor).
  }];
}

// mfma_v2
def MIOpen_MFMAV2Op:
    MIOpen_Op<"mfma_v2">,
    Arguments<(ins AnyTypeOf<[F32, VectorOfLengthAndType<[4], [F16]>, VectorOfLengthAndType<[2], [BF16]>]>: $sourceA,
                   AnyTypeOf<[F32, VectorOfLengthAndType<[4], [F16]>, VectorOfLengthAndType<[2], [BF16]>]>: $sourceB,
                   VectorOfRankAndType<[1], [F32, F16, BF16]>: $destC)>,
    Results<(outs VectorOfRankAndType<[1], [F32, F16, BF16]>: $destD)> {
  let summary = "XDLOPS MFMA V2";
  let description = [{
    The `miopen.mfma_v2` op is an abstraction of XDLOPS.
  }];
}

// xdlops_gemm_V2
def MIOpen_XdlopsGemmV2Op:
    MIOpen_Op<"xdlops_gemm_v2">,
    Arguments<(ins MemRefOf<[F32, F16, BF16]>:$matrixA,
                   MemRefOf<[F32, F16, BF16]>:$matrixB,
                   Index:$threadOffsetA,
                   Index:$threadOffsetB,
                   MemRefOf<[F32, F16, BF16]>:$bufferA,
                   MemRefOf<[F32, F16, BF16]>:$bufferB,
                   Variadic<VectorOfRankAndType<[1], [F32, F16, BF16]>>:$vectorCs)>,
    Results<(outs Variadic<VectorOfRankAndType<[1], [F32, F16, BF16]>>: $vectorDs)> {
  let summary = "XDLOPS GEMM V2";
  let description = [{
    The `miopen.xdlops_gemm_v2` op is an abstraction of doing GEMM based on XDLOPS.
    It would employ a series of `miopen.mfma` operations.
  }];
}

#endif // MIOPEN_OPS
