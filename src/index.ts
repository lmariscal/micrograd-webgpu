import { Tensor } from "./tensor";

let xor_train = [
    { input: [0, 0], output: [0] }, // sample
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
];

let and_train = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [0] },
    { input: [1, 0], output: [0] },
    { input: [1, 1], output: [1] }
];

function loss(y: Tensor, desired: number): Tensor {
    return y.add(-desired).pow(2);
}

function forward(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor): Tensor {
    let x1w1 = x.mul(w1);
    let x1w1b1 = x1w1.add(b1);
    let y1 = x1w1b1.ReLU();

    let y1w2 = y1.mul(w2);
    let y1w2b2 = y1w2.add(b2);
    let y2 = y1w2b2.ReLU();

    return y2;
}

async function main() {
    xor_train;
    and_train;

    let desired = 13;

    Tensor.defaultDevice = "wgpu";
    let x = new Tensor([5, 6], [1, 2], "x");

    let w1 = new Tensor([1, 1, 1, 1], [2, 2], "w1", true);
    let b1 = new Tensor([1, 1], [1, 2], "b1", true);

    let w2 = new Tensor([1, 1], [2, 1], "w2", true);
    let b2 = new Tensor([1], [1], "b2", true);

    for (let i = 0; i < 10; i++) {
        w1.zeroGrad();
        b1.zeroGrad();
        w2.zeroGrad();
        b2.zeroGrad();

        let y = forward(x, w1, b1, w2, b2);
        let los = loss(y, desired);
        los.backward();

        let lr = 0.001;
        w1 = w1.add(w1.grad.mul(-lr));
        b1 = b1.add(b1.grad.mul(-lr));
        w2 = w2.add(w2.grad.mul(-lr));
        b2 = b2.add(b2.grad.mul(-lr));

        console.log("i:", i);
    }

    let y = forward(x, w1, b1, w2, b2);
    let los = loss(y, desired);
    los.backward();
    let lossc = los.copy();
    let yc = y.copy();
    if (lossc.device == "wgpu") {
        await lossc.toCPU();
        await yc.toCPU();
    }
    console.log("loss:", lossc.data[0]);
    console.log("y:", yc.data[0]);
}

main();
