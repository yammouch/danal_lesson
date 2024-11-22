use wasm_bindgen_test::*;
use prj02::add;

#[wasm_bindgen_test]
fn it_works() {
    let result = add(2, 2);
    assert_eq!(result, 4);
}

#[wasm_bindgen_test]
fn pass() {
    assert_eq!(1, 1);
}

//#[wasm_bindgen_test]
//fn fail() {
//    assert_eq!(1, 2);
//}
