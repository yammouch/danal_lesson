export function updown2note_build(notekey) {
  let o = {};
  notekey.forEach( (a, i) => {
    a.forEach( (k) => {
      o[k] = i;
    })
  });
  return o;
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

