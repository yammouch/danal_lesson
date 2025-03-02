use wasm_bindgen::prelude::*;

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
  pub fn add_re(&mut self, x: f64) {
    let pi = std::f64::consts::PI;
    let mag = (self.mag*self.mag + x*x + 2.*self.mag*x*self.angle.cos()).sqrt();
    let x_angle = if 0. <= mag {
      -self.angle
    } else if 0. <= self.angle {
      pi - self.angle
    } else {
      -pi - self.angle
    };
    let angle_add = (x*x_angle.sin()).atan2(self.mag + x*x_angle.cos());
    self.mag = mag;
    self.angle += angle_add;
  }

  pub fn rotate(&mut self, a: f64) {
    let pi = std::f64::consts::PI;
    self.angle += a;
    if pi < self.angle {
      self.angle -= 2.*pi;
    }
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
      self.c.add_re(self.wav[self.wav_pos]);
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
  r: Resonator,
  v: Vec<f32>,
}

#[wasm_bindgen]
impl Source {
  pub fn reson1(f: f64) -> Self {
    Self {
      r: Resonator::new(f, vec![1.], 1. - 1e-4, 1. - 1e-1),
      v: Vec::with_capacity(128),
    }
  }

  pub fn off(&mut self) {
    self.r.off();
  }

  pub fn on(&mut self) {
    self.r.on();
  }

  pub fn tick(&mut self, n: usize) {
    self.v.clear();
    for _ in 0..n {
      self.r.tick();
      self.v.push(self.r.out() as f32);
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
