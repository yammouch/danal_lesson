use wasm_bindgen_test::*;
use reson::Fir;
use reson::log;

#[wasm_bindgen_test]
fn fir_test() {
  let mut fir = Fir::new(vec![1.0]);
  let test_case : Vec<f64> = vec![-0.5, -0.25, 0., 0.25, 0.5];
  test_case.into_iter().for_each( |x| assert_eq!(fir.next(x), x) );
  log(&format!("{:?}", fir));
}

#[wasm_bindgen_test]
fn order_to_f_test() {
  let test_cases
  = vec![ (0, vec![0./1.]),
          (1, vec![0./2., 1./2.]),
          (2, vec![0./3., 1./3.]),
          (3, vec![0./4., 1./4., 2./4.]),
          (4, vec![0./5., 1./5., 2./5.]),
          (5, vec![0./6., 1./6., 2./6., 3./6.]) ];
  test_cases.iter().for_each( |&(i, ref expc)| {
    let ret = reson::order_to_f(i);
    ret.iter().zip(expc.iter()).for_each( |(&r, &e)| {
      assert_eq!(r, e);
    })
  })
}
