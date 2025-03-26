export function updown2note_build(notekey) {
  let o = {};
  notekey.forEach( (a, i) => {
    a.forEach( (k) => {
      o[k] = i;
    })
  });
  return o;
}

export function note2onoff_build(notekey) {
  return notekey.map( (a) => {
    let o = {};
    a.forEach( (x) => {
      o[x] = false;
    });
    return o;
  });
}

class press2note {

  constructor() {
    this.pressed = {};
    this.pressed[4] = {"KeyZ": false, "Digit2": false}
    this.press2note = {};
    this.press2note["KeyZ"] = "4";
    this.press2note["Digit2"] = "4";
  }

  on(key) {
    const note = this.press2note[key];
    console.log(note);
    if (!note) { return; }
    console.log(this.pressed[note]);
    if (this.pressed[note][key]) { return; }
    this.pressed[note][key] = true;
    return { cmd: "on", note: note };
  }

  off(key) {
    const note = this.press2note[key];
    if (!note) { return; }
    this.pressed[note][key] = false;
    if (Object.entries(this.pressed[note]).some((k) => k[1])) { return; }
    return { cmd: "off", note: note };
  }

}

export const pressed = new press2note();

let notekey = [
  ["KeyQ"],
  ["ShiftLeft", "Digit2"],
  ["KeyA"],
  ["KeyW"],
  ["KeyZ", "Digit3"],
  ["KeyS"],
  ["KeyE"],
  ["KeyX", "Digit4"],
  ["KeyD"],
  ["KeyR"],
  ["KeyC", "Digit5"],
  ["KeyF"],
  ["KeyT"],
  ["KeyV", "Digit6"],
  ["KeyG"],
  ["KeyY"],
  ["KeyB", "Digit7"],
  ["KeyH"],
  ["KeyU"],
  ["KeyN", "Digit8"],
  ["KeyJ"],
  ["KeyI"],
  ["KeyM", "Digit9"],
  ["KeyK"],
  ["KeyO"],
  ["Comma", "Digit0"],
  ["KeyL"],
  ["KeyP"],
  ["Period", "Minus"],
  ["Semicolon"],
  ["BracketLeft"],
  ["Slash", "Equal"],
  ["Quote"],
  ["BracketRight"],
  ["IntlRo", "IntlYen"],
  ["Backslash"],
  ["Enter"],
  ["ShiftRight"]
];
