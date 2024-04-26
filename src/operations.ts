import { Tensor } from "./tensor";

export class CPUOperations {
    static mulTensor(tensor: Tensor, other: Tensor): Tensor {
        // TODO: Not use a naive approach
        // TODO: Support n-dim Tensors

        if (tensor.shape[1] != other.shape[0]) {
            throw new Error("Tensor multiplication not set for given mismatched Tensors");
        }

        let dst = new Tensor(tensor.shape[0] * other.shape[1], [tensor.shape[0], other.shape[1]]);
        // TOOD
        return dst;
    }

    static mulScalar(tensor: Tensor, scalar: number): Tensor {
        const data = tensor.data.map((val) => val * scalar);
        return new Tensor(data, tensor.shape);
    }

    static addTensor(tensor: Tensor, other: Tensor): Tensor {
        if (tensor.shape != other.shape) {
            throw new Error(`Shapes of Tensors mismatch. ${tensor.label}(${tensor.shape}) vs ${other.label}(${other.shape})`)
        }

        let dst = new Tensor([], tensor.shape);
        // TODO
        return dst;
    }

    static addScalar(tensor: Tensor, scalar: number): Tensor {
        const data = tensor.data.map((val) => val + scalar);
        return new Tensor(data, tensor.shape);
    }
}