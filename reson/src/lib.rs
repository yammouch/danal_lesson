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

pub fn div_wave(n: u32) -> Vec<Vec<f64>> {
  (0..n).map( |i|
    (0..n).map( |j| {
      let tau          = std::f64::consts::TAU;
      let (iflt, j, n) = (f64::from(i), f64::from(j), f64::from(n));
      (tau*iflt*j/n).cos() / if i == 0 { n } else { 0.5*n }
    }).collect()
  ).collect()
}
