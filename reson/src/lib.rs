use wasm_bindgen::prelude::*;
use std::ops::{Add, AddAssign};

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

#[derive(Debug, Clone, Copy)]
pub struct Cplxpol {
  pub mag  : f64,
  pub angle: f64, // rad
}

impl Cplxpol {
  pub fn from_reim(re: f64, im: f64) -> Self {
    Self {
      mag  : (re*re+im*im).sqrt(),
      angle: im.atan2(re)
    }
  }

  pub fn re(&self) -> f64 {
    self.mag*self.angle.cos()
  }

  pub fn im(&self) -> f64 {
    self.mag*self.angle.sin()
  }

  pub fn rotate(&mut self, a: f64) {
    let pi = std::f64::consts::PI;
    self.angle += a;
    if pi < self.angle {
      self.angle -= 2.*pi;
    }
  }
}

impl Add for Cplxpol {
  type Output = Self;

  fn add(self, other: Self) -> Self {
    let pi = std::f64::consts::PI;
    let diff_angle = other.angle - self.angle;
    let mag = ( self.mag*self.mag + other.mag*other.mag
              + 2.*self.mag*other.mag*diff_angle.cos() )
            .max(0.).sqrt();
    let angle_add = (other.mag*diff_angle.sin()).atan2(
                     other.mag*diff_angle.cos()+self.mag );
    let angle_updt = self.angle + angle_add;
    let angle_regu = angle_updt - ((angle_updt+pi)/(2.*pi)).floor()*(2.*pi);
    Self { mag: mag, angle: angle_regu }
  }
}

impl Add<f64> for Cplxpol {
  type Output = Self;

  fn add(self, other: f64) -> Self {
    self + Cplxpol { mag: other, angle: 0. }
  }
}

impl AddAssign for Cplxpol {
  fn add_assign(&mut self, other: Self) {
    *self = self.clone() + other;
  }
}

impl AddAssign<f64> for Cplxpol {
  fn add_assign(&mut self, other: f64) {
    *self = self.clone() + other;
  }
}

#[derive(Debug)]
#[wasm_bindgen]
pub struct Resonator {
  c        : Cplxpol,
  w        : f64,
  wav      : Vec<f64>,
  wav_pos  : usize,
  decay_on : f64,
  decay_off: f64,
  decay    : f64,
  out      : f64,
}

#[wasm_bindgen]
impl Resonator {
  pub fn new(f: f64, wav: Vec<f64>, decay_on: f64, decay_off: f64) -> Self {
    let tau = std::f64::consts::TAU;
    Self { c: Cplxpol { mag: 0., angle: 0. }, w: tau*f, wav: wav, wav_pos: 0,
           decay_on: decay_on, decay_off: decay_off,
           decay: decay_off, out: 0. }
  }

  pub fn off(&mut self) {
    self.decay = self.decay_off;
  }

  pub fn on(&mut self) {
    self.decay = self.decay_on;
    self.wav_pos = self.wav.len();
  }

  pub fn tick(&mut self) {
    if self.wav_pos != 0 {
      self.wav_pos -= 1;
      self.c += self.wav[self.wav_pos]
    }
    self.c.rotate(self.w);
    self.c.mag *= self.decay;
    self.out = self.c.mag * self.c.angle.cos();
  }

  pub fn out(&self) -> f64 {
    self.out
  }

  pub fn ptr(&self) -> *const f64 {
    &self.out
  }

  pub fn reson1(f: f64) -> Self {
    Resonator::new(f, vec![1.], 1. - 1e-4, 1. - 1e-1)
  }
}

impl Iterator for Resonator {
  type Item = f64;

  fn next(&mut self) -> Option<Self::Item> {
    self.tick();
    Some(self.out)
  }
}

#[derive(Debug)]
#[wasm_bindgen]
pub struct Source {
  r: Vec<Resonator>,
  v: Vec<f32>,
}

#[wasm_bindgen]
impl Source {
  pub fn new(f_master_a: f64) -> Self {
    let mut slf = Self {
      r: vec![],
      v: Vec::with_capacity(128),
    };
    for i in 0..=39 {
      slf.r.push(
       Resonator::new(
        f_master_a * 2f64.powf((i as f64 - 33.)/12.),
        vec![1.], 1. - 1e-4, 1. - 1e-1));
    }
    slf
  }

  pub fn off(&mut self, i: usize) {
    self.r[i].off();
  }

  pub fn on(&mut self, i: usize) {
    self.r[i].on();
  }

  pub fn tick(&mut self, n: usize) {
    self.v.clear();
    for _ in 0..n {
      self.r.iter_mut().for_each( |r| r.tick() );
      self.v.push(self.r.iter().map(Resonator::out).sum::<f64>() as f32);
    }
  }

  pub fn ptr(&self) -> *const f32 {
    self.v.as_ptr()
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

#[cfg(test)]
mod cplxpol_test {
  use wasm_bindgen_test::*;
  use super::Cplxpol;

  #[wasm_bindgen_test(unsupported = test)]
  fn add () {
    use std::iter::once;
    let crd = vec![3f64.sqrt()*0.5, 1.0, 3f64.sqrt()];
    let crd = crd.iter().rev().map(|&x| -x).chain(once(0f64))
              .chain(crd.iter().map(|&x| x)).collect::<Vec<_>>();
    let pi = std::f64::consts::PI;
    let mut points : Vec<(f64, f64)> = vec![];
    for &re in &crd {
      for &im in &crd {
        points.push((re, im));
      }
    }
    for &(are, aim) in &points {
      for &(bre, bim) in &points {
        let a = Cplxpol::from_reim(are, aim);
        let b = Cplxpol::from_reim(bre, bim);
        let sum = a + b;
        assert!((are + bre - sum.re()).abs() < 1e-6,
         "are: {are}, bre: {bre}, result: {}", sum.re());
        assert!((aim + bim - sum.im()).abs() < 1e-6,
         "aim: {aim}, bim: {bim}, result: {}", sum.im());
        assert!(sum.angle.abs() < pi+0.1,
         "angle: {}", sum.angle);
      }
    }
  }
}
