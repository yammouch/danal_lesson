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

#[wasm_bindgen_test]
fn convolve_test() {
  assert_eq!(reson::convolve(&vec![1., 3.], &vec![1., 6., 9.]),
             vec![1., 9., 27., 27.]);
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

fn zeros_test1(f: f64, coef: &[f64]) {
  let (x, _) = freq_resp(f, coef, 0);
  assert!(x.abs() < 1e-6);
}

#[wasm_bindgen_test]
fn zeros_test() {
  let ret = reson::zeros(&vec![1., -1. ]);
  zeros_test1(1./2., &ret[0]);
  zeros_test1(0./2., &ret[1]);

  let ret = reson::zeros(&vec![1., -0.5]);
  zeros_test1(1./3., &ret[0]);
  zeros_test1(0./3., &ret[1]);

  let ret = reson::zeros(&vec![1.,  0. , -1. ]);
  zeros_test1(1./4., &ret[0]);
  zeros_test1(2./4., &ret[0]);
  zeros_test1(0./4., &ret[1]);
  zeros_test1(2./4., &ret[1]);
  zeros_test1(0./4., &ret[2]);
  zeros_test1(1./4., &ret[2]);

  let ret = reson::zeros(&vec![1.,  0.5, -0.5, -1.]);
  zeros_test1(1./6., &ret[0]);
  zeros_test1(2./6., &ret[0]);
  zeros_test1(3./6., &ret[0]);
  zeros_test1(0./6., &ret[1]);
  zeros_test1(2./6., &ret[1]);
  zeros_test1(3./6., &ret[1]);
  zeros_test1(0./6., &ret[2]);
  zeros_test1(1./6., &ret[2]);
  zeros_test1(3./6., &ret[2]);
  zeros_test1(0./6., &ret[3]);
  zeros_test1(1./6., &ret[3]);
  zeros_test1(2./6., &ret[3]);
}

#[wasm_bindgen_test]
fn polyval_test() {
  let (re, im) = reson::polyval(&[1.], 0.5);
  assert!((re - 1.0).abs() < 1e-6);
  assert!((im - 0.0).abs() < 1e-6);

  let (re, im) = reson::polyval(&[0., 0., 1.], 0.5_f64.sqrt());
  assert!((re -  0.0).abs() < 1e-6);
  assert!((im - -1.0).abs() < 1e-6);

  let (re, im) = reson::polyval(&[-1., 2.], 0.0);
  assert!((re - -1.0).abs() < 1e-6);
  assert!((im - -2.0).abs() < 1e-6);

  let (re, im) = reson::polyval(&[-1., 2.], 0.5);
  assert!((re -   0.0         ).abs() < 1e-6);
  assert!((im - -(3f64.sqrt())).abs() < 1e-6);
}

#[wasm_bindgen_test]
fn linsolve01_test() {
  let (x0, x1) = reson::linsolve01(2., 3., 3., 4.);
  assert!((x0 -  3.0).abs() < 1e-6);
  assert!((x1 - -2.0).abs() < 1e-6);

  let (x0, x1) = reson::linsolve01(4., 3., 3., 2.);
  assert!((x0 -  3.0).abs() < 1e-6);
  assert!((x1 - -4.0).abs() < 1e-6);
}

#[wasm_bindgen_test]
fn normalize_0_test() {
  let result = reson::normalize_0(&vec![-1., -1., -1., -1.]);
  let (re, im) = freq_resp(1./4., &result, 0);
  assert!(re.abs() < 1e-6);
  assert!(im.abs() < 1e-6);
  let (re, im) = freq_resp(0., &result, 0);
  assert!((re - 1.).abs() < 1e-6);
  assert!( im      .abs() < 1e-6);
}

#[wasm_bindgen_test]
fn normalize_nyq_test() {
  let result = reson::normalize_nyq(&vec![-1., 1., -1., 1.], 0);
  let (re, im) = freq_resp(1./4., &result, 0);
  assert!(re.abs() < 1e-6);
  assert!(im.abs() < 1e-6);
  let (re, im) = freq_resp(1./2., &result, 0);
  assert!((re - 1.).abs() < 1e-6);
  assert!( im      .abs() < 1e-6);

  let result = reson::normalize_nyq(&vec![-1., 1., -1., 1.], 1);
  let (re, im) = freq_resp(1./4., &result, 1);
  assert!(re.abs() < 1e-6);
  assert!(im.abs() < 1e-6);
  let (re, im) = freq_resp(1./2., &result, 1);
  assert!((re - 1.).abs() < 1e-6);
  assert!( im      .abs() < 1e-6);
}

#[wasm_bindgen_test]
fn normalize_other_test() {
  let result = reson::normalize_other(&vec![-1., 0., 1.], 0, 1./4.);
  log(&format!("{:?}", result));
  let (re, im) = freq_resp(1./4., &result, 0);
  assert!((re - 1.).abs() < 1e-6);
  assert!( im      .abs() < 1e-6);
}
