import { Tensor } from "../src/tensor";
import { expect, test, describe } from "bun:test";

describe("Tensor General Operations", () => {
    test("Create tensor successfully", () => {
        let t = new Tensor(10, [2, 5]);
        expect(t.at(0, 0)).toBe(0);
    });

    test("Detect invalid shape", () => {
        expect(() => new Tensor(20, [3, 4])).toThrowError();
    });

    test("Tensor 2D > not supported", () => {
        expect(() => new Tensor(60, [3, 4, 5])).toThrowError();
    });

    test("Pre-existing array gets copied successfully", () => {
        let a = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ];
        let t = new Tensor(a, [4, 4]);
        expect(t.data).toBeArray();
        expect(t.data).toEqual(a);
    });
})


describe("CPU TensorXTensor Operations", () => {
    test("Tensor mul (Dot) with a tensor", () => {
        let t = new Tensor([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ], [4, 4]);
        let o = new Tensor([
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
        ], [4, 4]);

        expect(t.mul(o).data).toEqual([
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
        ]);
    });
    test("Tensor add with a tensor", () => {
        let t = new Tensor([2, 3, 4], [3]);
        let o = new Tensor([5, 6, 7], [3]);
        expect(t.add(o).data).toEqual([7, 9, 11]);
    });
})

describe("CPU TensorXScalar Operations", () => {
    test("Tensor mul with a scalar", () => {
        let t = new Tensor([2, 3, 4], [3]);
        expect(t.mul(3).data).toEqual([6, 9, 12]);
    });

    test("Tensor add with a scalar", () => {
        let t = new Tensor([2, 3, 4], [3]);
        expect(t.add(3).data).toEqual([5, 6, 7]);
    });
})

describe("CPU Grad Tensor Operations", () => {
    test("Tensor add Tensor backward", () => {
        let x = new Tensor([1, 2, 3, 4], [2, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = x.add(y);
        z.backward();
        expect(x.grad.data).toEqual([1, 1, 1, 1]);
    });

    test("Tensor add Scalar backward", () => {
        let x = new Tensor([1, 2, 3, 4], [2, 2]);
        let z = x.add(3);
        z.backward();
        expect(x.grad.data).toEqual([1, 1, 1, 1]);
    });

    test("Tensor mul Tensor backward", () => {
        let x = new Tensor([1, 2, 3, 4], [2, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = x.mul(y);
        z.backward();
        expect(x.grad.data).toEqual([11, 15, 11, 15]);
        expect(y.grad.data).toEqual([4, 4, 6, 6]);
    });

    test("Tensor mul Scalar backward", () => {
        let x = new Tensor([1, 2, 3, 4], [2, 2]);
        let z = x.mul(3);
        z.backward();
        expect(x.grad.data).toEqual([3, 3, 3, 3]);
    });

    test("Tensor sub Tensor backward", () => {
        let x = new Tensor([1, 2, 3, 4], [2, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = x.sub(y);
        z.backward();
        expect(x.grad.data).toEqual([1, 1, 1, 1]);
        expect(y.grad.data).toEqual([1, 1, 1, 1]);
    });

    test("Tensor sub Scalar backward", () => {
        let x = new Tensor([1, 2, 3, 4], [2, 2]);
        let z = x.sub(3);
        z.backward();
        expect(x.grad.data).toEqual([1, 1, 1, 1]);
    });

    test("Tensor ReLU Tensor backward", () => {
        let x = new Tensor([1, -2, 3, -4], [2, 2]);
        let z = x.ReLU();
        z.backward();
        expect(x.grad.data).toEqual([1, 0, 1, 0]);
    });

    test("Tensor leakyReLU Tensor backward", () => {
        let x = new Tensor([1, -2, 3, -4], [2, 2]);
        let z = x.leakyReLU(0.1);
        z.backward();
        expect(x.grad.data).toEqual([1, 0.1, 1, 0.1]);
    });

    test("Test power Tensor backward", () => {
        let x = new Tensor([1, 2, 3, 4], [2, 2]);
        let z = x.pow(2);
        z.backward();
        expect(x.grad.data).toEqual([2, 4, 6, 8]);
    });

    test("Test elemWiseMul Tensor backward", () => {
        let x = new Tensor([1, 2, 3, 4], [2, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = x.elemWiseMul(y);
        z.backward();
        expect(x.grad.data).toEqual([5, 6, 7, 8]);
        expect(y.grad.data).toEqual([1, 2, 3, 4]);
    });
})

describe("Tensor deep backward", () => {
    // verified against JAX

    test("Test mul mul backward", () => {
        let x = new Tensor([5, 7], [1, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = new Tensor([1, 2], [2, 1]);
        let w = x.mul(y).mul(z);
        expect(w.data).toEqual([246]);

        w.backward();
        expect(x.grad.data).toEqual([17, 23]);
        expect(y.grad.data).toEqual([5, 10, 7, 14]);
        expect(z.grad.data).toEqual([74, 86]);
    });

    test("Test mul mul sub backward", () => {
        let x = new Tensor([5, 7], [1, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = new Tensor([1, 2], [2, 1]);
        let w = x.mul(y).mul(z).sub(new Tensor([14], [1]));
        expect(w.data).toEqual([232]);

        w.backward();
        expect(x.grad.data).toEqual([17, 23]);
        expect(y.grad.data).toEqual([5, 10, 7, 14]);
        expect(z.grad.data).toEqual([74, 86]);
    });

    test("Test mul mul sub pow backward", () => {
        let x = new Tensor([5, 7], [1, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = new Tensor([1, 2], [2, 1]);
        let w = x.mul(y).mul(z).sub(new Tensor([14], [1])).pow(2);
        expect(w.data).toEqual([53824]);

        w.backward();
        expect(x.grad.data).toEqual([7888, 10672]);
        expect(y.grad.data).toEqual([2320, 4640, 3248, 6496]);
        expect(z.grad.data).toEqual([34336, 39904]);
    });

    test("Test mul mul add sub pow backward", () => {
        let x = new Tensor([5, 7], [1, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = new Tensor([1, 2], [2, 1]);
        let b = new Tensor([4], [1, 1]);
        let w = x.mul(y).mul(z).add(b).sub(new Tensor([14], [1])).pow(2);
        expect(w.data).toEqual([55696]);

        w.backward();
        expect(x.grad.data).toEqual([8024, 10856]);
        expect(y.grad.data).toEqual([2360, 4720, 3304, 6608]);
        expect(z.grad.data).toEqual([34928, 40592]);
    });

    test("Test mul mul add ReLU sub pow backward", () => {
        let x = new Tensor([5, 7], [1, 2]);
        let y = new Tensor([5, 6, 7, 8], [2, 2]);
        let z = new Tensor([1, 2], [2, 1]);
        let b = new Tensor([4], [1, 1]);
        let w = x.mul(y).mul(z).add(b).ReLU().sub(new Tensor([14], [1])).pow(2);
        expect(w.data).toEqual([55696]);

        w.backward();
        expect(x.grad.data).toEqual([8024, 10856]);
        expect(y.grad.data).toEqual([2360, 4720, 3304, 6608]);
        expect(z.grad.data).toEqual([34928, 40592]);
    });

    test("Test learn via backprop, single step", () => {
        let x = new Tensor(2, [1, 2]);
        let y = new Tensor(4, [2, 2]);
        let z = new Tensor(2, [2, 1]);
        let b = new Tensor(1, [1, 1]);
        let desired = new Tensor([14], [1]);
        x.randomize();
        y.randomize();
        z.randomize();
        b.randomize();

        let loss1 = x.mul(y).mul(z).add(b).ReLU().sub(desired).pow(2);
        loss1.backward();

        x = x.sub(x.grad.mul(0.01));
        y = y.sub(y.grad.mul(0.01));
        z = z.sub(z.grad.mul(0.01));
        b = b.sub(b.grad.mul(0.01));
        let loss2 = x.mul(y).mul(z).add(b).ReLU().sub(desired).pow(2);
        expect(loss2.data[0]).toBeLessThan(loss1.data[0]);
    });

    test("Test learn via backprop, 10 steps", () => {
        let x = new Tensor(2, [1, 2]);
        let y = new Tensor(4, [2, 2]);
        let z = new Tensor(2, [2, 1]);
        let b = new Tensor(1, [1, 1]);
        let desired = new Tensor([14], [1]);
        x.randomize();
        y.randomize();
        z.randomize();
        b.randomize();

        for (let i = 0; i < 10; ++i) {
            x.zeroGrad();
            y.zeroGrad();
            z.zeroGrad();
            b.zeroGrad();

            let loss1 = x.mul(y).mul(z).add(b).ReLU().sub(desired).pow(2);
            loss1.backward();

            x = x.sub(x.grad.mul(0.01));
            y = y.sub(y.grad.mul(0.01));
            z = z.sub(z.grad.mul(0.01));
            b = b.sub(b.grad.mul(0.01));
            let loss2 = x.mul(y).mul(z).add(b).ReLU().sub(desired).pow(2);
            expect(loss2.data[0]).toBeLessThan(loss1.data[0]);
        }
    });

});
