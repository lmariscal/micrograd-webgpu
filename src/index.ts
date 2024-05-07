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

let continueTraining = false;

async function main() {
    xor_train;
    and_train;
    Tensor.defaultDevice = "cpu";
    Tensor.debugMode = false;

    let model = new Model([2, 2, 1]);

    for (let i = 0; i < 1000; i++) {
        GPUOperations.computePassCount = 0;
        let loss = new Tensor([0], [1], "loss");

        for (let sample of xor_train) {
            let x = new Tensor(sample.input, [1, 2]);
            let desired = new Tensor(sample.output, [1, 1]);

            let y = model.forward(x);
            let sampleLoss = y.sub(desired).pow(2);
            loss = loss.add(sampleLoss);
        }
        loss = loss.mul(0.25);
        loss.backward();

        if (loss.device == "wgpu") {
            await loss.toCPU();
        }

        if (i % 100 == 0) {
            console.group("i:", i);
            console.log("loss:", loss.pretty());
            let params = model.params;
            for (let p of params) {
                // console.log(p.label + "grad", p.grad.pretty());
            }
            console.groupEnd();
        }

        let lr = 0.1;
        model.learn(lr);

        model.zeroGrad();
    }

    for (let sample of xor_train) {
        let x = new Tensor(sample.input, [1, 2]);
        let y = model.forward(x);
        if (y.device == "wgpu") {
            await x.toCPU();
            await y.toCPU();
        }
        console.log("x:", x.pretty(), "y:", y.data[0] > 0.5 ? 1 : 0, "y:", y.pretty());
    }
}

main();
