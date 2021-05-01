
#map = affine_map<(d0, d1) -> (d0  + d1 - 2)>
module {
  func @print_i32(i32)
  func @print_newline()
  func @main() {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = affine.apply #map (%c0, %c1)
    %1 = index_cast %0 : index to i32
    call @print_i32(%1) : (i32) -> ()
    call @print_newline() : () -> ()
    return
  }
}

