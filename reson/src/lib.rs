use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}

#[derive(Debug)]
pub struct Fir {
  pos  : usize,
  buf  : Vec<f64>,
  coeff: Vec<f64>,
}

impl Fir {
  pub fn new(v: Vec<f64>) -> Self {
    Fir {
      pos  : 0,
      buf  : vec![0.0; v.len()],
      coeff: v,
    }
  }

  pub fn next(&mut self, din: f64) -> f64 {
    if self.pos == 0 {
      self.pos = self.buf.len() - 1;
    } else {
      self.pos -= 1;
    }
    self.buf[self.pos] = din;
    self.buf[self.pos..].iter().zip(&self.coeff)
    .chain(self.buf[0..self.pos].iter()
           .zip(&self.coeff[self.coeff.len()-self.pos..]))
    .map(|(&b, &c)| b*c).sum()
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

pub fn convolve(u: &[f64], v: &[f64]) -> Vec<f64> {
  let mut ret : Vec<_> = vec![0.0; u.len() + v.len() - 1];
  u.iter().enumerate().for_each( |(i, &u1)| {
    (i..).zip(v).for_each( |(j, &v1)| { ret[j] += u1*v1; } )
  });
  ret
}

pub fn cumconvolve<'a, I>(polys: I) -> impl Iterator<Item=Vec<f64>> + 'a
where
  I: Iterator<Item=&'a Vec<f64>> + 'a,
{
  polys.scan(vec![1.], |state, p| {
    *state = convolve(state, p);
    Some(state.clone())
  })
}

pub fn zeros(cosines: &[f64]) -> Vec<Vec<f64>> {
  let polys : Vec<Vec<f64>> = cosines.iter().map( |&c|
    match c {
      1. => vec![1., -1.  ], // for DC
     -1. => vec![1.,  1.  ], // for Nyquist
      _  => vec![1., -2.*c, 1.],
    }
  ).collect();

  let fwd = cumconvolve(polys[..polys.len()-1].iter()).collect::<Vec<_>>();
  let bwd = cumconvolve(polys[1..].iter().rev())
            .collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>();
  let mut mid = fwd.iter().zip(&bwd[1..])
                .map(|(f, b)| convolve(f, b)).collect::<Vec<_>>();
  let mut ret = vec![bwd[0].clone()];
  ret.append(&mut mid);
  ret.push(fwd[fwd.len()-1].clone());
  ret
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
  let cosines = f.iter().map( |&f| f.cos() ).collect::<Vec<_>>();
  let z = zeros(&cosines);
  normalize_bunch(dly1st, f, &z)
}
