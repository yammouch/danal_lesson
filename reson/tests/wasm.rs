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
   Fir::new(vec![1.], 1),
   vec![1.],
   3./4.,
   1./2. );
  let out : Vec<f64> = re.by_ref().take(3).collect();
  assert_eq!(out, [0., 0., 0.]);
  re.on();
  let out : Vec<f64> = re.by_ref().take(4).collect();
  assert_eq!(out, [0., 1., 0., 3./4.]);
  re.off();
  let out : Vec<f64> = re.by_ref().take(4).collect();
  assert_eq!(out, [0., 3./8., 0., 3./16.]);

  let mut re = reson::Resonator::new(
   Fir::new(vec![1.], 1),
   vec![1., 1./2.],
   1.,
   0. );
  re.on();
  let out : Vec<f64> = re.by_ref().take(5).collect();
  assert_eq!(out, [0., 1./2., 1., 1./2., 1.]);

  let mut re = reson::Resonator::new(
   Fir::new(vec![1., 1./2.], 2),
   vec![1.],
   1.,
   0. );
  re.on();
  let out : Vec<f64> = re.by_ref().take(7).collect();
  assert_eq!(out, [0., 0., 1., 1./2., 0., 1., 1.]);
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

fn freq_resp(f: f64, coef: &[f64], skip: usize) -> (f64, f64) {
  let tau = std::f64::consts::TAU;
  let re : f64 = (skip..).zip(coef).map( |(i, &x)|
                   (-tau*f*(i as f64)).cos()*x
                 ).sum();
  let im : f64 = (skip..).zip(coef).map( |(i, &x)|
                   (-tau*f*(i as f64)).sin()*x
                 ).sum();
  (re, im)
}

#[wasm_bindgen_test]
fn resonator_coef_test() {
  let result = reson::resonator_coef(2, &vec![0., 1./3.]);

  let (re, im) = freq_resp(0./3., &result[0], 2);
  assert!((re - 1.).abs() < 1e-6);
  assert!( im      .abs() < 1e-6);

  let (re, im) = freq_resp(1./3., &result[0], 2);
  assert!( re      .abs() < 1e-6);
  assert!( im      .abs() < 1e-6);

  let (re, im) = freq_resp(0./3., &result[1], 2);
  assert!( re      .abs() < 1e-6);
  assert!( im      .abs() < 1e-6);

  let (re, im) = freq_resp(1./3., &result[1], 2);
  assert!((re - 1.).abs() < 1e-6);
  assert!( im      .abs() < 1e-6);
}

#[wasm_bindgen_test]
fn harms_test() {
  let result = reson::harms(0.12, 0.4);
  let expc = vec![0.0, 0.12, 0.24, 0.36, 0.5];
  result.iter().zip(&expc).for_each( |(&r, &e)| assert!((r-e).abs() < 1e-6));

  let result = reson::harms(0.12, 0.3);
  let expc = vec![0.0, 0.12, 0.24, 0.37, 0.5];
  result.iter().zip(&expc).for_each( |(&r, &e)| assert!((r-e).abs() < 1e-6));

  let result = reson::harms(0.15, 0.4);
  let expc = vec![0.0, 0.15, 0.3, 0.4+0.1/3.];
  result.iter().zip(&expc).for_each( |(&r, &e)| assert!((r-e).abs() < 1e-6));

  let result = reson::harms(0.15, 0.5);
  let expc = vec![0.0, 0.15, 0.3, 0.45];
  result.iter().zip(&expc).for_each( |(&r, &e)| assert!((r-e).abs() < 1e-6));
}
