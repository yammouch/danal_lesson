use wasm_bindgen_test::*;
use reson::Fir;
use reson::log;

#[wasm_bindgen_test]
fn fir_test() {
  let mut fir = Fir::new(vec![1.0], 0);
  let test_case : Vec<f64> = vec![-0.5, -0.25, 0., 0.25, 0.5];
  test_case.into_iter().for_each( |x| {
    fir.tick(x);
    assert_eq!(fir.out, x) } );
  log(&format!("{:?}", fir));
}

#[wasm_bindgen_test]
fn resonator_test() {
  let mut re = reson::Resonator::new(
   1./4.,
   vec![1.],
   1./2.,
   1./4. );

  let out : Vec<f64> = re.by_ref().take(3).collect();
  let exp = [0., 0., 0.];
  for i in 0..out.len() {
    assert!((out[i] - exp[i]).abs() < 1e-6);
  }

  re.on();
  let out : Vec<f64> = re.by_ref().take(4).collect();
  let exp = [0., -1./4., 0., 1./16.];
  for i in 0..out.len() {
    assert!((out[i] - exp[i]).abs() < 1e-6);
  }

  re.off();
  let out : Vec<f64> = re.by_ref().take(4).collect();
  let exp = [0., -1./256., 0., 1./4096.];
  for i in 0..out.len() {
    assert!((out[i] - exp[i]).abs() < 1e-6);
  }
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

#[wasm_bindgen_test]
fn div_wave_test() {
  let t = std::f64::consts::TAU;

  let expc = vec!
  [vec![(0.*t/3.).cos()/3. , (0.*t/3.).cos()/3. , (0.*t/3.).cos()/3. ],
   vec![(0.*t/3.).cos()/1.5, (1.*t/3.).cos()/1.5, (2.*t/3.).cos()/1.5]];
  assert_eq!(reson::div_wave(2), expc);

  let expc = vec!
  [vec![(0.*t/4.).cos()/4. , (0.*t/4.).cos()/4. ,
        (0.*t/4.).cos()/4. , (0.*t/4.).cos()/4. ],
   vec![(0.*t/4.).cos()/2. , (1.*t/4.).cos()/2. ,
        (2.*t/4.).cos()/2. , (3.*t/4.).cos()/2. ],
   vec![(0.*t/4.).cos()/4. , (2.*t/4.).cos()/4. ,
        (4.*t/4.).cos()/4. , (6.*t/4.).cos()/4. ]];
  assert_eq!(reson::div_wave(3), expc);
}

#[wasm_bindgen_test]
fn vxm_test() {
  let v = vec![1., 2.];
  let m = vec![vec![3., 5.],
               vec![4., 6.]];
  let expc = vec![11., 17.];
  assert_eq!(reson::vxm(&v, &m), expc);
}
