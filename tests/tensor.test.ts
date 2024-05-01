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
