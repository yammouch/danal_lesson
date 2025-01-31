use wasm_bindgen::prelude::*;
extern crate nalgebra as na;

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}

#[derive(Debug)]
#[wasm_bindgen]
pub struct Fir {
  pos     : usize,
  skip    : usize,
  buf     : Vec<f64>,
  coeff   : Vec<f64>,
  pub out : f64,
}

impl Fir {
  pub fn new(v: Vec<f64>, skip: usize) -> Self {
    Fir {
      pos  : 0,
      skip : skip,
      buf  : vec![0.0; v.len()+skip],
      coeff: v,
      out  : 0.0,
    }
  }

  pub fn tick(&mut self, din: f64) {
    if self.pos == 0 {
      self.pos = self.buf.len() - 1;
    } else {
      self.pos -= 1;
    }
    self.buf[self.pos] = din;
    self.out = self.buf[self.pos..].iter().chain(&self.buf).skip(self.skip)
               .zip(&self.coeff).map(|(&b, &c)| b*c).sum();
  }
}

#[derive(Debug)]
#[wasm_bindgen]
pub struct Resonator {
  fir      : Fir,
  wav      : Vec<f64>,
  wav_pos  : usize,
  decay_on : f64,
  decay_off: f64,
  decay    : f64,
}

#[wasm_bindgen]
impl Resonator {
  pub fn new(fir: Fir, wav: Vec<f64>, decay_on: f64, decay_off: f64) -> Self {
    Self { fir: fir, wav: wav, wav_pos: 0,
           decay_on: decay_on, decay_off: decay_off,
           decay: decay_off }
  }

  pub fn off(&mut self) {
    self.decay = self.decay_off;
  }

  pub fn on(&mut self) {
    self.decay = self.decay_on;
    self.wav_pos = self.wav.len();
  }

  pub fn tick(&mut self) {
    if self.wav_pos == 0 {
      self.fir.tick(self.fir.out*self.decay);
    } else {
      self.wav_pos -= 1;
      self.fir.tick(self.wav[self.wav_pos]);
    }
  }

  pub fn out(&self) -> f64 {
    self.fir.out
  }

  pub fn ptr(&self) -> *const f64 {
    &self.fir.out
  }

  pub fn coeff(&self) -> Vec<f64> {
    self.fir.coeff.to_vec()
  }

  pub fn reson1(f: f64) -> Self {
    //let f : f64 = 0.016;
    //let f : f64 = 0.013; // diverges
    let h = harms(f, 0.4);
    let dly1st = ((1./f + 0.5) as usize)/2 + 1;
    // 2->2, 3->2, 4->3, 5->3, 6->4, 7->4, 8->5, 9->5
    let c = resonator_coef(dly1st, &h);
    let fir = Fir::new(vxm(&vec![0., 1.], &c), dly1st);
    let mag = (fir.coeff.len() as f64)/2.;
    Resonator::new(fir, vec![mag], 1. - 1e-4, 1. - 1e-1)
  }
}

impl Iterator for Resonator {
  type Item = f64;

  fn next(&mut self) -> Option<Self::Item> {
    self.tick();
    Some(self.fir.out)
  }
}

// 0 -> [0.0]
// 1 -> [0.0, 0.5  ]
// 2 -> [0.0, 0.333]             = [0/3, 1/3]
// 3 -> [0.0. 0.25 , 0.5]        = [0/4, 1/4, 2/4]
// 4 -> [0.0. 0.2  , 0.4]        = [0/5, 1/5, 2/5]
// 5 -> [0.0, 0.166, 0.333, 0.5] = [0/6, 1/6, 2/6, 3/6]
pub fn order_to_f(ord: u32) -> Vec<f64> {
  let denom = f64::from(ord+1);
  (0..=(ord+1)/2).map( |i| f64::from(i)/denom ).collect()
}

pub fn div_wave(ord: u32) -> Vec<Vec<f64>> {
  let tau   = std::f64::consts::TAU;
  let denom = f64::from(ord+1);
  order_to_f(ord).into_iter().map( |f|
    (0..=ord).map( |i| {
      (tau*f*f64::from(i)).cos()
      / if f == 0.0 || f == 0.5 { denom } else { 0.5*denom }
    }).collect()
  ).collect()
}

pub fn vxm(v: &[f64], m: &[Vec<f64>]) -> Vec<f64> {
  v.iter().zip(m)
  .map( |(&v1, row)| row.iter().map( |&x| v1*x ).collect() )
  .reduce( |acc: Vec<f64>, row| acc.into_iter().zip(row).map( |(a, r)| a+r )
                                .collect() ).unwrap()
}

pub fn resonator_coef(dly1st: usize, f: &[f64]) -> Vec<Vec<f64>> {
  let mut f_pm : Vec<f64> = vec![];
  f.iter().for_each( |&x| {
    match x {
      0. | 0.5 => {
        f_pm.push(x);
      },
      x => {
        f_pm.push( x);
        f_pm.push(-x);
      },
    }
  });
  let mut b = na::DMatrix::from_element(f_pm.len(), f.len(),
   na::Complex::new(0., 0.) );
  let mut i : usize = 0;
  f.iter().enumerate().for_each( |(j, &x)| {
    match x {
      0. | 0.5 => {
        b[(i, j)] = na::Complex::new(1., 0.);
        i += 1;
      },
      _ => {
        b[(i  , j)] = na::Complex::new(1., 0.);
        b[(i+1, j)] = na::Complex::new(1., 0.);
        i += 2;
      },
    }
  });
  let f_pm = na::DVector::from_vec(f_pm);
  let n = na::RowDVector::from_vec(
   (dly1st..dly1st+f_pm.len()).map( |i| i as f64).collect::<Vec<_>>());
  let a = na::DMatrix::from_fn(f_pm.len(), n.len(), |i, j| {
    let tau = std::f64::consts::TAU;
    let val = (f_pm[i] - (f_pm[i]*n[j] + 1./2.).floor()/n[j])*n[j];
    let ph = tau*val;
    na::Complex::new(ph.cos(), -ph.sin())
  } );
  let x = a.full_piv_lu().solve(&b).unwrap();
  let v = x.column_iter().map( |c| {
    c.as_slice().iter().map( |&x| x.re ).collect::<Vec<_>>()
  }).collect::<Vec<_>>();
  v
}

pub fn harms(f: f64, flim: f64) -> Vec<f64> {
  use std::iter::successors;
  let point_n = (1.0/f + 0.5) as usize;
  let mut v : Vec<f64> = (0..=(point_n-1)/2).map(|i| f*i as f64)
                         .take_while(|&fi| fi <= flim).collect();
  let sector_n = point_n - 2*(v.len() - 1);
  let sector = 2.*(0.5 - v[v.len()-1])/(sector_n as f64);
  successors(Some(v[v.len()-1]), |x| Some(x + sector)).skip(1)
  .take((sector_n-1)/2).for_each( |f| v.push(f) );
  if point_n % 2 == 0 { v.push(0.5); }
  v
}
