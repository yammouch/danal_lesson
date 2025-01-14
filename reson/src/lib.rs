use std::ops::{Add, Mul};
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
    Resonator::new(fir, vec![8.], 1. - 1e-4, 1. - 1e-1)
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

pub fn convolve<T>(u: &[T], v: &[T]) -> Vec<T>
where
  T: Add<Output=T> + Mul<Output=T> + Copy
{
  let mut ret : Vec<_> = v.iter().map( |&v| u[0]*v ).collect();
  (1..u.len()).for_each( |i| {
    (0..v.len()-1).for_each ( |j| {
      ret[i+j] = ret[i+j] + u[i] * v[j];
    });
    ret.push(u[i] * v[v.len()-1]);
  });
  ret
}

pub fn cumconvolve<T>(polys: &[Vec<T>]) -> Vec<Vec<T>>
where
  T: Add<Output=T> + Mul<Output=T> + Copy
{
  if polys.is_empty() {
    vec![]
  } else {
    let mut ret = vec![polys[0].clone()];
    (1..polys.len()).for_each( |i| {
      ret.push(convolve(&ret[ret.len()-1], &polys[i]));
    });
    ret
  }
}

pub fn diagless(polys: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
  //                          [b30, b31, b32, b33, b34] bwd[3] = ret1
  //                               [b20, b21, b22, b23] bwd[2] = ret[0]
  // fwd[0] [f00, f01]           *      [b10, b11, b12] bwd[1] = ret[1]
  // fwd[1] [f10, f11, f12]      *           [b00, b01] bwd[0] = ret[2]
  // fwd[2] [f20, f21, f22, f23]                               = ret[3]
  let mut fwd = if polys.is_empty() {
    vec![]
  } else {
    cumconvolve(&polys[..polys.len()-1])
  };
  let mut polys_rev = polys.to_vec().clone();
  polys_rev.reverse();
  let mut bwd = cumconvolve(&polys_rev);

  let ret1 = match bwd.pop() {
    Some(v) => v,
    None    => vec![1.],
  };

  let mut ret  = vec![];
  if let Some(v) = bwd.pop() {
    ret.push(v);
  }
  if !fwd.is_empty() {
    (0..fwd.len()-1).for_each( |i| {
      ret.push(convolve(&fwd[i], &bwd[fwd.len()-2-i]));
    });
  }
  if let Some(v) = fwd.pop() {
    ret.push(v);
  }
  if polys.len() == 1 {
    ret.push(vec![1.]);
  }

  (ret, ret1)
}

pub fn zeros(f: &[f64]) -> Vec<Vec<f64>> {
  let tau = std::f64::consts::TAU;
  let mut midds : Vec<Vec<f64>> = vec![];
  let mut mid_i : Vec<usize>    = vec![];
  let mut edges : Vec<Vec<f64>> = vec![];
  let mut edg_i : Vec<usize>    = vec![];
  f.iter().enumerate().for_each( |(i, &f)| {
    match f {
      0. => {
        edges.push(vec![1., -1.]);
        edg_i.push(i);
      }, // for DC
      0.5 => {
        edges.push(vec![1.,  1.]);
        edg_i.push(i);
      }, // for Nyquist
      f => {
        midds.push(vec![1., -2.*(tau*f).cos(), 1.]);
        mid_i.push(i);
      },
    }
  });

  let (edg_diag, edg_prod) = diagless(&edges);
  let (mid_diag, mid_prod) = diagless(&midds);

  let mut polys : Vec<Vec<f64>> = vec![];
  let mut m = 0;
  let mut e = 0;
  loop {
    if midds.len() <= m {
      if edges.len() <= e {
        break;
      } else {
        polys.push(convolve(&edg_diag[e], &mid_prod));
        e += 1;
      }
    } else if edges.len() <= e {
      polys.push(convolve(&mid_diag[m], &edg_prod));
      m += 1;
    } else if mid_i[m] < edg_i[e] {
      polys.push(convolve(&mid_diag[m], &edg_prod));
      m += 1;
    } else {
      polys.push(convolve(&edg_diag[e], &mid_prod));
      e += 1;
    }
  }

  polys
}

pub fn polyval(coef: &[f64], cosine: f64) -> (f64, f64) {
  let sine = -(1.0-cosine*cosine).sqrt();
  coef.iter().rfold((0f64, 0f64), |(re, im), &k|
    (k+cosine*re-sine*im, cosine*im+sine*re) )
}

pub fn linsolve01(a00: f64, a01: f64, a10: f64, a11: f64) -> (f64, f64) {
  if a00.abs() < a01.abs() {
    let r = -a00/a01;
    let sol = 1.0/(a10 + a11*r);
    (sol, sol*r)
  } else {
    let r = -a01/a00;
    let sol = 1.0/(a10*r + a11);
    (sol*r, sol)
  }
}

pub fn normalize_0(coef: &[f64]) -> Vec<f64> {
  let denom = 1f64/coef.iter().sum::<f64>();
  coef.iter().map( |&x| x * denom ).collect()
}

pub fn normalize_nyq(coef: &[f64], dly1st: usize) -> Vec<f64> {
  let even_sum = coef     .iter().step_by(2).sum::<f64>();
  let odd_sum  = coef[1..].iter().step_by(2).sum::<f64>();
  let denom = 1f64 /
   if dly1st % 2 == 0 { even_sum - odd_sum } else { odd_sum - even_sum };
  coef.iter().map( |&x| x * denom ).collect()
}

pub fn normalize_other(coef: &[f64], dly1st: usize, f: f64)
 -> Vec<f64> {
  let w = std::f64::consts::TAU * f;
  let dly1st = dly1st as f64;
  let (z0re, z0im) = polyval(coef, w.cos());
  let (a1re, a1im) = ((w* dly1st    ).cos(), -(w* dly1st    ).sin());
  let (a2re, a2im) = ((w*(dly1st+1.)).cos(), -(w*(dly1st+1.)).sin());
  let (z1re, z1im) = (z0re*a1re - z0im*a1im, z0re*a1im + z0im*a1re);
  let (z2re, z2im) = (z0re*a2re - z0im*a2im, z0re*a2im + z0im*a2re);
  let k = linsolve01(z1im, z2im, z1re, z2re);
  convolve(&vec![k.0, k.1], coef)
}

pub fn normalize_bunch(
 dly1st : usize,
 f      : &[f64],
 coeffs : &[Vec<f64>]) -> Vec<Vec<f64>> {
  f.iter().zip(coeffs).map( |(&f, coef)|
    match f {
      0.0 => normalize_0    (coef),
      0.5 => normalize_nyq  (coef, dly1st),
      f   => normalize_other(coef, dly1st, f),
    }
  ).collect()
}

pub fn resonator_coef(dly1st: usize, f: &[f64]) -> Vec<Vec<f64>> {
  let z = zeros(f);
  normalize_bunch(dly1st, f, &z)
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
