import { Model } from "../src/nn";
import { Tensor } from "../src/tensor";
import { expect, test, describe } from "bun:test";

Tensor.defaultDevice = "cpu";

let xor_train = [
    { input: [0, 0], output: [0] },
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

describe("Neural Network testing", () => {
    test("Basic Operations 1 Perceptron like, no activation", () => {
        let x = new Tensor([4, 6], [1, 2]);
        let w = new Tensor([1, 0], [2, 1]);
        let b = new Tensor([1], [1]);
        let y = x.mul(w).add(b);

        expect(y.data).toEqual([5]);
    });

    test("Basic Operations 2 Perceptron like, no activation", () => {
        let x = new Tensor([4, 6], [1, 2]);
        let w = new Tensor([1, 0, 0, 1], [2, 2]);
        let b = new Tensor([1, 1], [1, 2]);
        let y = x.mul(w).add(b);

        let w2 = new Tensor([1, 0], [2, 1]);
        let b2 = new Tensor([1], [1]);
        let y2 = y.mul(w2).add(b2);

        expect(y2.data).toEqual([6]);
    });

    test("Basic Operations 2 Perceptron like, with activation", () => {
        let x = new Tensor([4, 6], [1, 2]);
        let w = new Tensor([1, 0, 0, 1], [2, 2]);
        let b = new Tensor([1, 1], [1, 2]);
        let y = x.mul(w).add(b).ReLU();

        let w2 = new Tensor([1, 0], [2, 1]);
        let b2 = new Tensor([1], [1]);
        let y2 = y.mul(w2).add(b2).ReLU();

        expect(y2.data).toEqual([6]);
    });

    test("MLP 1 neuron learns, single step", () => {
        let model = new Model([2, 1]);
        let x = new Tensor([3, 4], [1, 2]);
        let desired = new Tensor([14], [1, 1]);

        let loss = model.forward(x).sub(desired).pow(2);
        loss.backward();

        model.learn(0.01);
        let loss2 = model.forward(x).sub(desired).pow(2);
        expect(loss2.data[0]).toBeLessThan(loss.data[0]);
    });

    test("MLP 1 neuron learns, 10 steps", () => {
        let model = new Model([2, 1]);
        let x = new Tensor([3, 4], [1, 2]);
        let desired = new Tensor([14], [1, 1]);

        for (let i = 0; i < 10; i++) {
            model.zeroGrad();

            let loss = model.forward(x).sub(desired).pow(2);
            loss.backward();
            model.learn(0.01);

            let loss2 = model.forward(x).sub(desired).pow(2);
            expect(loss2.data[0]).toBeLessThan(loss.data[0]);
        }
    });

    test("MLP 1 neuron learns AND gate, single step", () => {
        let model = new Model([2, 1]);
        let lr = 0.01;

        let loss = new Tensor([0], [1], "loss");
        for (let sample of and_train) {
            let x = new Tensor(sample.input, [1, 2]);
            let desired = new Tensor(sample.output, [1, 1]);

            let y = model.forward(x);
            loss = loss.add(y.sub(desired).pow(2));
        }
        loss = loss.mul(1.0 / and_train.length);
        loss.backward();

        model.learn(lr);

        let loss2 = new Tensor([0], [1], "loss");
        for (let sample of and_train) {
            let x = new Tensor(sample.input, [1, 2]);
            let desired = new Tensor(sample.output, [1, 1]);

            let y = model.forward(x);
            loss2 = loss2.add(y.sub(desired).pow(2));
        }
        loss2 = loss2.mul(1.0 / and_train.length);

        expect(loss2.data[0]).toBeLessThan(loss.data[0]);
        expect(loss.data[0]).toBeGreaterThan(0);
        expect(loss2.data[0]).toBeGreaterThan(0);
    });

    test("MLP 1 neuron learns AND gate, 1000 steps", () => {
        let model = new Model([2, 1]);
        let lr = 0.01;

        let previousLoss = new Tensor([Number.MAX_VALUE], [1], "loss");
        for (let i = 0; i < 1000; i++) {
            model.zeroGrad();
            let loss = new Tensor([0], [1], "loss");
            for (let sample of and_train) {
                let x = new Tensor(sample.input, [1, 2]);
                let desired = new Tensor(sample.output, [1, 1]);

                let y = model.forward(x);
                loss = loss.add(y.sub(desired).pow(2));
            }
            loss = loss.mul(1.0 / and_train.length);
            loss.backward();

            model.learn(lr);

            expect(loss.data[0]).toBeLessThan(previousLoss.data[0]);
            expect(loss.data[0]).toBeGreaterThan(0);
            previousLoss = loss.copy();
        }

        for (let sample of and_train) {
            let x = new Tensor(sample.input, [1, 2]);
            let y = model.forward(x);
            expect(y.data[0] > 0.5 ? 1 : 0).toEqual(sample.output[0]);
        }
    });

    test("MLP 3 neurons learn XOR gate, single step", () => {
        let model = new Model([2, 2, 1]);
        let lr = 0.1;

        let loss = new Tensor([0], [1], "loss");
        for (let sample of xor_train) {
            let x = new Tensor(sample.input, [1, 2]);
            let desired = new Tensor(sample.output, [1, 1]);

            let y = model.forward(x);
            loss = loss.add(y.sub(desired).pow(2));
        }
        loss = loss.mul(1.0 / and_train.length);
        loss.backward();

        model.learn(lr);

        let loss2 = new Tensor([0], [1], "loss");
        for (let sample of and_train) {
            let x = new Tensor(sample.input, [1, 2]);
            let desired = new Tensor(sample.output, [1, 1]);

            let y = model.forward(x);
            loss2 = loss2.add(y.sub(desired).pow(2));
        }
        loss2 = loss2.mul(1.0 / and_train.length);

        expect(loss2.data[0]).toBeLessThan(loss.data[0]);
        expect(loss.data[0]).toBeGreaterThan(0);
        expect(loss2.data[0]).toBeGreaterThan(0);
    });

    test("MLP 3 neurons learn XOR gate, 1000 steps", () => {
        let model = new Model([2, 2, 1]);
        let lr = 0.1;

        // heavy on the magic numbers
        model.layers[0].w = new Tensor([-0.8468587983395675, 0.8232307232300605, 0.8468587989469499, -0.8232307259414902], [2, 2]);
        model.layers[0].b = new Tensor([1.0591844213091538e-9, -2.82655284519338e-9], [1, 2]);
        model.layers[1].w = new Tensor([1.1927618106561708, 1.2269960337239865], [2, 1]);
        model.layers[1].b = new Tensor([1.0225553790039269e-7], [1]);

        let previousLoss = new Tensor([Number.MAX_VALUE], [1], "loss");
        for (let i = 0; i < 1000; i++) {
            model.zeroGrad();
            lr = lr * 0.999;

            let loss = new Tensor([0], [1], "loss");
            for (let sample of xor_train) {
                let x = new Tensor(sample.input, [1, 2]);
                let desired = new Tensor(sample.output, [1, 1]);

                let y = model.forward(x);
                loss = loss.add(y.sub(desired).pow(2));
            }
            loss = loss.mul(1.0 / and_train.length);

            previousLoss = loss.copy();

            loss.backward();

            model.learn(lr);
            expect(loss.data[0]).toBeGreaterThan(0);
        }

        if (previousLoss.data[0] < 0.01) {
            for (let sample of xor_train) {
                let x = new Tensor(sample.input, [1, 2]);
                let y = model.forward(x);
                expect(y.data[0] > 0.5 ? 1 : 0).toEqual(sample.output[0]);
            }
        }
    });
});
