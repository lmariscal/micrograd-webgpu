import { Tensor } from "./tensor";

export class CPUOperations {
    static mulTensor(tensor: Tensor, other: Tensor): Tensor {
        // TODO: Not use a naive approach
        // TODO: Support n-dim Tensors

        if (tensor.shape[1] != other.shape[0]) {
            throw new Error(`Tensor multiplication not set for given mismatched Tensors - ${tensor.shape} vs ${other.shape}`);
        }

        let dst = new Tensor(tensor.shape[0] * other.shape[1], [tensor.shape[0], other.shape[1]]);
        for (let i = 0; i < tensor.shape[0]; ++i) {
            for (let j = 0; j < other.shape[1]; ++j) {
                let sum = 0;
                for (let k = 0; k < tensor.shape[1]; ++k) {
                    sum += tensor.data[i * tensor.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
                dst.data[i * other.shape[1] + j] = sum;
            }
        }
        return dst;
    }

    static mulScalar(tensor: Tensor, scalar: number): Tensor {
        const data = tensor.data.map((x) => x * scalar);
        return new Tensor(data, tensor.shape);
    }

    static addTensor(tensor: Tensor, other: Tensor): Tensor {
        if (tensor.shape[0] != other.shape[0] || tensor.shape[1] != other.shape[1]) {
            throw new Error(`Shapes of Tensors mismatch. ${tensor.label}(${tensor.shape}) vs ${other.label}(${other.shape})`)
        }

        let dst = new Tensor(tensor.data.length, tensor.shape);
        for (let i = 0; i < tensor.data.length; ++i) {
            dst.data[i] = tensor.data[i] + other.data[i];
        }
        return dst;
    }

    static addScalar(tensor: Tensor, scalar: number): Tensor {
        const data = tensor.data.map((x) => x + scalar);
        return new Tensor(data, tensor.shape);
    }

    static ReLU(tensor: Tensor): Tensor {
        const data = tensor.data.map((x) => x > 0 ? x : 0);
        return new Tensor(data, tensor.shape);
    }
}
