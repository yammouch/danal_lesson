use wasm_bindgen::prelude::*;
use std::arch::wasm32::*;

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}

//#[wasm_bindgen]
#[no_mangle]
pub fn add(left: u64, right: u64) -> u64 {
    log("Hello");
    left + right
}

//#[wasm_bindgen]
//extern "C" {
//  #[wasm_bindgen(js_namespace = console)]
//  fn log(s: &str);
//}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[no_mangle]
pub unsafe fn simd_test_body() -> [f32; 4] {
  use std::mem;
  let u: v128 = f32x4(1., 2., 3., 4.);
  let v: v128 = f32x4(2., 3., 4., 5.);
  let w: v128 = f32x4_add(u, v);
  let sum: [f32; 4] = { mem::transmute(w) };
  sum
}
