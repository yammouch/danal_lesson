import { Source } from "./sub01.js";

class SquareProcessor extends AudioWorkletProcessor {

  constructor() {
    super();
    this.port.onmessage = (e) => {
      console.log(e.data);
      if (e.data.cmd == "on") {
        console.log(e.data.note);
        this.src.on(e);
      } else if (e.data.cmd == "off") {
        this.src.off(e);
      } else if (e.data.cmd == "init") {
        this.src = new Source(e);
      }
    };
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0];
    const channel = output[0];
    for (let i = 0; i < channel.length; i++) {
      let sum = 0.0;
      sum += this.src.pop();
      channel[i] = 0.2*sum;
    }
    return true;
  }

}

registerProcessor("sq-pr", SquareProcessor);
