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

async function main() {
    xor_train;
    and_train;

    Tensor.defaultDevice = "wgpu";
    let x = new Tensor([5, 6], [1, 2], "x");

    let w1 = new Tensor([1, 0, 0, 1], [2, 2], "w1", true);
    let b1 = new Tensor([1, 1], [1, 2], "b1", true);
    let y1 = x.mul(w1).add(b1).ReLU();

    let w2 = new Tensor([1, 0], [2, 1], "w2", true);
    let b2 = new Tensor([1], [1], "b2", true);
    let y2 = y1.mul(w2).add(b2).ReLU();

    if (y2.device == "wgpu") {
        await y2.toCPU();
    }
    console.log(y2.label, "\n" + y2.pretty());

    let des = w1.descendancy();
    for (let d of des.reverse()) {
        console.log(d);
    }

    let w_grad = w1.grad;
    await w_grad.toCPU();
    console.log(w_grad.label, "\n" + w_grad.pretty());

    let t1 = new Tensor([1, 2, 3, 4], [2, 2], "t1");
    let t2 = new Tensor([5, 6, 7, 8], [2, 2], "t2");
    let r = t1.elemWiseMul(t2);

    if (r.device == "wgpu") {
        await r.toCPU();
    }
    console.log(r.label, "\n" + r.pretty());
}

main();
