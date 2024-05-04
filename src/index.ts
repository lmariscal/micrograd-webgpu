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

    Tensor.defaultDevice = "cpu";
    // let x = new Tensor([5, 6], [1, 2]);
    // let w = new Tensor([1, 0, 0, 1], [2, 2]);
    // let b = new Tensor([1, 1], [1, 2]);
    // let y = x.mul(w).add(b).ReLU();

    // let w2 = new Tensor([1, 0], [2, 1]);
    // let b2 = new Tensor([1], [1]);
    // let y2 = y.mul(w2).add(b2).ReLU();
    let t1 = new Tensor([
        1,  2,  3,  4,
        5,  6,  7,  8,
    ], [2, 4]);
    let t2 = new Tensor([
        1, 0, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1,
        1, 1, 0, 0, 0, 0,
    ], [4, 6]);
    let y2 = t1.mul(t2);

    if (y2.device == "wgpu") {
        await y2.toCPU();
    }
    let d = y2.data;
    console.log(d);
}

main();
