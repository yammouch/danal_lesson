use wasm_bindgen::prelude::*;
use std::ops::{Add, AddAssign, Mul, MulAssign};

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

#[derive(Debug, Clone, Copy, PartialEq)]
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
}

pub fn angle_regu(angle: f64) -> f64 {
  let pi = std::f64::consts::PI;
  angle - ((angle+pi)/(2.*pi)).floor()*(2.*pi)
}

impl Add for Cplxpol {
  type Output = Self;

  fn add(self, other: Self) -> Self {
    let diff_angle = other.angle - self.angle;
    let mag = ( self.mag*self.mag + other.mag*other.mag
              + 2.*self.mag*other.mag*diff_angle.cos() )
            .max(0.).sqrt();
    let angle_add = (other.mag*diff_angle.sin()).atan2(
                     other.mag*diff_angle.cos()+self.mag );
    Self { mag: mag, angle: angle_regu(self.angle + angle_add) }
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

impl Mul for Cplxpol {
  type Output = Self;

  fn mul(self, other: Self) -> Self {
    let mag = self.mag*other.mag;
    Self { mag: mag, angle: angle_regu(self.angle + other.angle) }
  }
}

impl MulAssign for Cplxpol {
  fn mul_assign(&mut self, other: Self) {
    *self = self.clone() * other;
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
    self.c *= Cplxpol { mag: self.decay, angle: self.w };
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

fn k2r<T: AsRef<[usize]>>(nk: usize, cfg: &[T]) -> Vec<Vec<(usize, usize)>> {
  use std::iter::once;
  let mx : Vec<usize> = cfg.iter().map( |r|
    r.as_ref().iter().max().expect("empty array").clone()
  ).collect();
  let cum : Vec<usize> = mx.iter().scan(0, |stt, &x| {
    *stt += nk + x;
    Some(*stt)
  }).collect();
  let mut cnt : Vec<usize> = vec![0; cum[cum.len()-1]];
  let mut ret : Vec<Vec<(usize, usize)>> = vec![vec![]; nk];
  for (&ofs, v) in once(&0).chain(cum.iter()).zip(cfg) {
    for h in v.as_ref().iter() {
      for i in 0..nk {
        ret[i].push((ofs+h+i, cnt[ofs+h+i]));
        cnt[ofs+h+i] += 1;
      }
    }
  }
  ret
}

#[derive(Debug, Clone)]
struct Rsn {
  c  : Vec<Cplxpol>,
  lim: Vec<f64>,
  dcn: Vec<f64>,
  dcf: Vec<f64>,
  k2r: Vec<Vec<(usize, usize)>>,
  prs: Vec<Vec<bool>>,
}

impl Rsn {
  fn tick(&self, dst: &mut [Cplxpol]) {
    for i in 0..dst.len() {
      dst[i] *= self.c[i];
      dst[i].mag = dst[i].mag.min(self.lim[i]);
    }
  }
  fn on(&mut self, i: usize) {
    for &t in self.k2r[i].iter() {
      self.prs[t.0][t.1] = true;
      self.c[t.0].mag = self.dcn[t.0];
    }
  }
  fn off(&mut self, i: usize) {
    for &t in self.k2r[i].iter() {
      self.prs[t.0][t.1] = false;
      if self.prs[t.0].iter().all( |&x| x == false ) {
        self.c[t.0].mag = self.dcf[t.0];
      }
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
struct Exc1 {
  n: Vec<usize>,
  v: Vec<Vec<Cplxpol>>,
}

impl Iterator for Exc1 {
  type Item = Cplxpol;

  fn next(&mut self) -> Option<Self::Item> {
    let mut acc = Cplxpol { mag: 0.0, angle: 0.0 };
    for i in 0..self.n.len() {
      if 0 < self.n[i] {
        self.n[i] -= 1;
        acc += self.v[i][self.n[i]]
      }
    }
    Some(acc)
  }
}

#[derive(Debug, Clone, PartialEq)]
struct Exc {
  a  : Vec<Exc1>,
  exi: Vec<Vec<(usize, usize)>>,
}

impl Exc {
  fn tick(&mut self, dst: &mut [Cplxpol]) {
    for i in 0..dst.len() {
      let c = self.a[i].next().expect("Exc1 no value");
      if c.mag != 0.0 {
        dst[i] += c;
      }
    }
  }
  fn on(&mut self, i: usize) {
    for &t in self.exi[i].iter() {
      self.a[t.0].n[t.1] = self.a[t.0].v[t.1].len();
    }
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

  fn points() -> Vec<(f64, f64)> {
    use std::iter::once;
    let crd = vec![3f64.sqrt()*0.5, 1.0, 3f64.sqrt()];
    let crd = crd.iter().rev().map(|&x| -x).chain(once(0f64))
              .chain(crd.iter().map(|&x| x)).collect::<Vec<_>>();
    let mut points : Vec<(f64, f64)> = vec![];
    for &re in &crd {
      for &im in &crd {
        points.push((re, im));
      }
    }
    points
  }

  #[wasm_bindgen_test(unsupported = test)]
  fn add () {
    let pi = std::f64::consts::PI;
    let points = points();
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

  #[wasm_bindgen_test(unsupported = test)]
  fn mul () {
    let pi = std::f64::consts::PI;
    let points = points();
    for &(are, aim) in &points {
      for &(bre, bim) in &points {
        let a = Cplxpol::from_reim(are, aim);
        let b = Cplxpol::from_reim(bre, bim);
        let prod = a * b;
        assert!((are*bre - aim*bim - prod.re()).abs() < 1e-6,
         "rere: {}, imim: {}, re: {}", are*bre, aim*bim, prod.re());
        assert!((aim*bre + are*bim - prod.im()).abs() < 1e-6,
         "imre: {}, reim: {}, im: {}", aim*bre, are*bim, prod.im());
        assert!(prod.angle.abs() < pi+0.1,
         "angle: {}", prod.angle);
      }
    }
  }
}

#[cfg(test)]
mod test_vecreson {
  use wasm_bindgen_test::*;

  #[wasm_bindgen_test(unsupported = test)]
  fn test_k2r() {
    let cfg = vec![vec![0usize]];
    let rv = super::k2r(13, &cfg);
    assert_eq!(rv,
     vec![vec![(0, 0)], vec![(1, 0)], vec![(2, 0)], vec![(3, 0)], vec![(4, 0)],
          vec![(5, 0)], vec![(6, 0)], vec![(7, 0)], vec![(8, 0)], vec![(9, 0)],
          vec![(10, 0)], vec![(11, 0)], vec![(12, 0)]]);

    let cfg = vec![vec![0usize, 12], vec![0, 7]];
    let rv = super::k2r(13, &cfg);
    assert_eq!(rv,
     vec![[(0, 0), (12, 1), (25, 0), (32, 1)],
          [(1, 0), (13, 0), (26, 0), (33, 1)],
          [(2, 0), (14, 0), (27, 0), (34, 1)],
          [(3, 0), (15, 0), (28, 0), (35, 1)],
          [(4, 0), (16, 0), (29, 0), (36, 1)],
          [(5, 0), (17, 0), (30, 0), (37, 1)],
          [(6, 0), (18, 0), (31, 0), (38, 0)],
          [(7, 0), (19, 0), (32, 0), (39, 0)],
          [(8, 0), (20, 0), (33, 0), (40, 0)],
          [(9, 0), (21, 0), (34, 0), (41, 0)],
          [(10, 0), (22, 0), (35, 0), (42, 0)],
          [(11, 0), (23, 0), (36, 0), (43, 0)],
          [(12, 0), (24, 0), (37, 0), (44, 0)]]);
  }

  #[wasm_bindgen_test(unsupported = test)]
  fn test_exc1() {
    use super::Cplxpol;
    use super::Exc1;

    let mut exc1 = Exc1 {
      n: vec![1usize, 2],
      v: vec![vec![ Cplxpol { mag: 1.0, angle: 1.0 } ],
              vec![ Cplxpol { mag: 1.0, angle: 0.0 },
                    Cplxpol { mag: 0.5, angle: 0.0 } ]],
    };

    let e = exc1.next().unwrap();
    assert_eq!(e, Cplxpol { mag: 1.0, angle: 1.0 } +
                  Cplxpol { mag: 0.5, angle: 0.0 } );
    assert_eq!(exc1, Exc1 {
      n: vec![0usize, 1],
      v: vec![vec![ Cplxpol { mag: 1.0, angle: 1.0 } ],
              vec![ Cplxpol { mag: 1.0, angle: 0.0 },
                    Cplxpol { mag: 0.5, angle: 0.0 } ]],
    });

    let e = exc1.next().unwrap();
    assert_eq!(e, Cplxpol { mag: 1.0, angle: 0.0 });
    assert_eq!(exc1, Exc1 {
      n: vec![0usize, 0],
      v: vec![vec![ Cplxpol { mag: 1.0, angle: 1.0 } ],
              vec![ Cplxpol { mag: 1.0, angle: 0.0 },
                    Cplxpol { mag: 0.5, angle: 0.0 } ]],
    });
  }

  #[wasm_bindgen_test(unsupported = test)]
  fn test_exc() {
    use super::Cplxpol;
    use super::Exc1;
    use super::Exc;
    let mut exc = Exc {
      a  : vec![
        Exc1 {
          n: vec![0],
          v: vec![vec![ Cplxpol { mag: 1.0, angle: 0.0 } ]],
        },
        Exc1 {
          n: vec![0, 0],
          v: vec![vec![ Cplxpol { mag: 1.0, angle: 0.0 } ],
                  vec![ Cplxpol { mag: 0.5, angle: 0.0 } ] ],
        },
      ],
      exi: vec![vec![(0, 0), (1, 1)],
                vec![(1, 0)]],
    };
    exc.on(0);
    assert_eq!(exc, Exc {
      a  : vec![
        Exc1 {
          n: vec![1],
          v: vec![vec![ Cplxpol { mag: 1.0, angle: 0.0 } ]],
        },
        Exc1 {
          n: vec![0, 1],
          v: vec![vec![ Cplxpol { mag: 1.0, angle: 0.0 } ],
                  vec![ Cplxpol { mag: 0.5, angle: 0.0 } ] ],
        },
      ],
      exi: vec![vec![(0, 0), (1, 1)],
                vec![(1, 0)]],
    });
    let mut o = vec![Cplxpol { mag: 0.0, angle: 0.0 },
                     Cplxpol { mag: 1.0, angle: 0.0 }];
    exc.tick(&mut o);
    assert_eq!(o,
     vec![Cplxpol { mag: 1.0, angle: 0.0 },
          Cplxpol { mag: 1.5, angle: 0.0 }] );
    assert_eq!(exc, Exc {
      a  : vec![
        Exc1 {
          n: vec![0],
          v: vec![vec![ Cplxpol { mag: 1.0, angle: 0.0 } ]],
        },
        Exc1 {
          n: vec![0, 0],
          v: vec![vec![ Cplxpol { mag: 1.0, angle: 0.0 } ],
                  vec![ Cplxpol { mag: 0.5, angle: 0.0 } ] ],
        },
      ],
      exi: vec![vec![(0, 0), (1, 1)],
                vec![(1, 0)]],
    });
  }
}
