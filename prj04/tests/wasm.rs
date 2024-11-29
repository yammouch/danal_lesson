use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;
use std::arch::wasm32::*;
use prj02::add;
use prj02::log;

//#[wasm_bindgen]
//extern "C" {
//  #[wasm_bindgen(js_namespace = console)]
//  fn log(s: &str);
//}

#[wasm_bindgen_test]
fn it_works() {
  let result = add(2, 2);
  assert_eq!(result, 4);
}

#[wasm_bindgen_test]
fn pass() {
  log("hello");
  assert_eq!(1, 1);
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn simd_test_body() -> [f32; 4] {
  use std::mem;
  let u: v128 = f32x4(1., 2., 3., 4.);
  let v: v128 = f32x4(2., 3., 4., 5.);
  let w: v128 = f32x4_add(u, v);
  let sum: [f32; 4] = { mem::transmute(w) };
  sum
}

#[wasm_bindgen_test]
fn simd_test() {
  unsafe {
    let r = simd_test_body();
    log(&format!("{} {} {} {}", r[0], r[1], r[2], r[3]))
  }
}

//#[wasm_bindgen_test]
//fn fail() {
//    assert_eq!(1, 2);
//}
