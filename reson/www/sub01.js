export function maxRow(a, i) {
  let maxi = i;
  let maxv = Math.abs(a[i][i]);
  for (let j = i+1; j < a.length; j++) {
    if (maxv < Math.abs(a[j][i])) {
      maxi = j;
      maxv = Math.abs(a[j][i]);
    }
  }
  let tmp = a[i];
  a[i] = a[maxi];
  a[maxi] = tmp;
}

export function divRow(a, i) {
  let inv = 1.0/a[i][i];
  for (let j = 0; j < a[i].length; j++) {
    a[i][j] *= inv;
  }
}

export function sweepRows(a, i) {
  for (let j = 0; j < a.length; j++) {
    if (i == j) {
      continue;
    }

    let x = a[j][i];
    for (let k = 0; k < a[j].length; k++) {
      a[j][k] -= a[i][k] * x;
    }
  }
}

export function linSolve(a) {
  for (let i = 0; i < a.length; i++) {
    maxRow(a, i);
    divRow(a, i);
    sweepRows(a, i);
  }
}

export function calcCoeff(divRatio, nHarm) {
  let divN = [];
  for (let i = -nHarm; i < nHarm; i++) {
    divN.push(i + Math.ceil(divRatio));
  }
  let system = [];
  for (let i = 1; i <= nHarm; i++) {
    let row = divN.map( (x) => Math.cos(2*Math.PI*x/divRatio*i) );
    row.push(1.0);
    system.push(row);
    row = divN.map( (x) => Math.sin(2*Math.PI*x/divRatio*i) );
    row.push(0.0);
    system.push(row);
  }
  linSolve(system);
  return system.map( (a) => a[2*nHarm] );
}

export class StringModel {

  constructor(divRatio) {
    let nHarm = 4;
    this.inow = 0;
    this.on = false;
    let bufsize = Math.ceil(divRatio);
    this.coeff = calcCoeff(divRatio, nHarm);
    this.wave = new Float32Array(bufsize);
    this.buf = new Float32Array(bufsize + nHarm);
    for (let i = 0; i < bufsize; i++) {
      if (i < bufsize / 2) {
        this.wave[i] = 1.0;
      } else {
        this.wave[i] = -1.0;
      }
    }
    this.excite = 0;
    this.decay = 1 - 1e-1;
  }

  pop() {
    this.inow--;
    if (this.inow < 0) {
      this.inow += this.buf.length;
    }
    if (0 < this.excite) {
      this.buf[this.inow] = this.wave[--this.excite];
    } else {
      let acc = 0.0;
      for (let i = this.inow-1, j = this.coeff.length-1; 0 <= j; i--, j--) {
        if (i < 0) {
          i += this.buf.length;
        }
        acc += this.coeff[j]*this.buf[i];
      }
      this.buf[this.inow] = this.decay*acc;
    }
  }

}

export class Source {

  static new(f_master_a) {
    const o = Object.create(Source.prototype);
    o.strs = [];
    let note = -31;
    for (let i = 0; i <= 37; i++, note++) {
      o.strs.push(new StringModel(1.0 / (f_master_a * 2**(note/12))));
    }
    return o;
  }

  on(e) {
    this.strs[e.data.note].excite = this.strs[e.data.note].wave.length;
    this.strs[e.data.note].decay = 1 - 1e-2;
  }

  off(e) {
    this.strs[e.data.note].decay = 1 - 1e-1;
  }

  pop() {
    let sum = 0.0;
    for (let j = 0; j < this.strs.length; j++) {
      this.strs[j].pop();
    }
    let to_lo = new Float32Array(this.strs.length);
    for (let i = this.strs.length-1; 0 < i; i--) {
      to_lo[i-1] = to_lo[i] + this.strs[i].buf[this.strs[i].inow];
    }
    for (let i = 0; i < this.strs.length-1; i++) {
      this.strs[i].buf[this.strs[i].inow] += to_lo[i]*1e-2;
    }
    return to_lo[0] + this.strs[0].buf[this.strs[0].inow];
  }

}

