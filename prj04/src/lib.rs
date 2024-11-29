use wasm_bindgen::prelude::*;

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
