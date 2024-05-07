import { Tensor } from "./tensor";

class Layer {
    w: Tensor;
    b: Tensor;

    constructor(w: Tensor, b: Tensor) {
        this.w = w;
        this.b = b;
    }

    forward(x: Tensor): Tensor {
        let xw = x.mul(this.w);
        let xwb = xw.add(this.b);
        return xwb.leakyReLU();
    }
}

export class Model {
    layers: Array<Layer>;

    get params(): Array<Tensor> {
        let params = [];
        for (let layer of this.layers) {
            params.push(layer.w);
            params.push(layer.b);
        }
        return params;
    }

    constructor(shape: Array<number>) {
        this.layers = [];
        for (let i = 0; i < shape.length - 1; i++) {
            let w = new Tensor(shape[i] * shape[i + 1], [shape[i], shape[i + 1]], `w${i}`);
            w.randomize();
            let b = new Tensor(shape[i + 1], [1, shape[i + 1]], `b${i}`);
            b.randomize();
            this.layers.push(new Layer(w, b));
        }
    }

    forward(x: Tensor): Tensor {
        let y = x;
        for (let layer of this.layers) {
            y = layer.forward(y);
        }
        return y;
    }

    learn(lr: number) {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].w = this.layers[i].w.sub(this.layers[i].w.grad.mul(lr));
            this.layers[i].w.label = `w${i}`;
            this.layers[i].b = this.layers[i].b.sub(this.layers[i].b.grad.mul(lr));
            this.layers[i].b.label = `b${i}`;
        }
    }

    zeroGrad() {
        for (let layer of this.layers) {
            layer.w.zeroGrad();
            layer.b.zeroGrad();
        }
    }
}
