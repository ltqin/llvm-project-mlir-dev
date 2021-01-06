
func @bitcast_test() {
  %0 = constant 3.0 : f32

  %1 = vector.broadcast %0 : f32 to vector<16xf32>
 // %3 = vector.bitcast %2 : vector<16xf32> to vector<32xi16>
  return
}
