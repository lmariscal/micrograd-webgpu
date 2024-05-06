import { Model } from "./nn";
import { GPUOperations } from "./operations";
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
    Tensor.debugMode = false;

    let model = new Model([1, 1]);
    let desired = new Tensor([13], [1], "desired");
    let x = new Tensor([5], [1], "x");

    let params = model.params;
    for (let p of params) {
        console.log(p.label, p.pretty());
    }

    for (let i = 0; i < 1000; i++) {
        console.log("i:", i);

        GPUOperations.computePassCount = 0;
        let y = model.forward(x);
        let los = y.sub(desired).pow(2);
        los.backward();

        let lr = 0.001;
        model.learn(lr);

        model.zeroGrad();
        console.log("computePassCount:", GPUOperations.computePassCount);
    }

    params = model.params;
    for (let p of params) {
        console.log(p.label, p.pretty());
    }

    let y = model.forward(x);
    let loss = y.sub(desired).pow(2);
    loss.backward();
    if (loss.device == "wgpu") {
        await loss.toCPU();
        await y.toCPU();
    }
    console.log("loss:", loss.pretty());
    console.log("y:", y.pretty());
}

main();
