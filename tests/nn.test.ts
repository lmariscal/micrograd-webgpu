import { Tensor } from "../src/tensor";
import { expect, test, describe } from "bun:test";

describe("Neural Network testing", () => {
    test.only("Basic Operations 1 Perceptron like, no activation", () => {
        let x = new Tensor([4, 6], [1, 2]);
        let w = new Tensor([1, 0], [2, 1]);
        let b = new Tensor([1], [1]);
        let y = x.mul(w).add(b);

        expect(y.data).toEqual([5]);
    });

    test.only("Basic Operations 2 Perceptron like, no activation", () => {
        let x = new Tensor([4, 6], [1, 2]);
        let w = new Tensor([1, 0, 0, 1], [2, 2]);
        let b = new Tensor([1, 1], [1, 2]);
        let y = x.mul(w).add(b);

        let w2 = new Tensor([1, 0], [2, 1]);
        let b2 = new Tensor([1], [1]);
        let y2 = y.mul(w2).add(b2);

        expect(y2.data).toEqual([6]);
    });

    test.only("Basic Operations 2 Perceptron like, with activation", () => {
        let x = new Tensor([4, 6], [1, 2]);
        let w = new Tensor([1, 0, 0, 1], [2, 2]);
        let b = new Tensor([1, 1], [1, 2]);
        let y = x.mul(w).add(b).ReLU();

        let w2 = new Tensor([1, 0], [2, 1]);
        let b2 = new Tensor([1], [1]);
        let y2 = y.mul(w2).add(b2).ReLU();

        expect(y2.data).toEqual([6]);
    });
});
