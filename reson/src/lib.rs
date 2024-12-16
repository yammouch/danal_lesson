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
    let mut buf : Vec<f64> = Vec::with_capacity(v.len());
    buf.resize(v.len(), 0.0);
    Fir {
      pos  : 0,
      buf  : buf,
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
    .map(|(b, c)| (*b)*(*c)).sum()
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
  .map( |(v1, row)| row.iter().map( |x| (*v1)*(*x) ).collect() )
  .reduce( |acc: Vec<f64>, row| acc.iter().zip(row).map( |(a, r)| (*a)+r )
                                .collect() ).unwrap()
}
