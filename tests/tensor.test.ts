import { Tensor } from "../src/tensor";
import { expect, test } from "bun:test";

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

test("Tensor mul with a scalar", () => {
    let t = new Tensor([2, 3, 4], [3]);
    expect(t.mul(3).data).toEqual([6, 9, 12]);
});

test("Tensor add with a scalar", () => {
    let t = new Tensor([2, 3, 4], [3]);
    expect(t.add(3).data).toEqual([5, 6, 7]);
});