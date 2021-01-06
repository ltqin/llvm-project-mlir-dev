
func @bitcast_test() {
  %0 = constant 3.0 : f32
  %2 = llvm.bitcast llvm.float %0 to i32
  %3 = and i32 %2, 2139095040
  %4 = icmp ne i32 %3, 2139095040
  %5 = and i32 %2, 65535
  %6 = icmp eq i32 %5, 0
  %7 = or i1 %4, %6
  %8 = or i32 %2, 65536
  %9 = select i1 %7, i32 %2, i32 %8
  %10 = bitcast i32 %9 to <2 x i16>
  %11 = extractelement <2 x i16> %10, i64 1
  ret i16 %11

 // %1 = vector.broadcast %0 : f32 to vector<16xf32>
 // %3 = vector.bitcast %2 : vector<16xf32> to vector<32xi16>
 // return
}
