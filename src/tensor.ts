import { CPUOperations, GPUOperations } from "./operations";
import { WGPUProvider } from "./wgpu_provider";

type Device = "cpu" | "wgpu";

// ~TODO: Actually make this Tensor class a n-dim Tensor rather than just a 2D tensor~ (Not a goal of this project)
export class Tensor {
    public static defaultDevice: Device = "cpu";

    private _label: string;
    private _device: Device;
    private _data: Array<number> | GPUBuffer;
    private _shape: Array<number>; // RxC
    private _length: number;
    private _requiresGrad: boolean;
    private _childen: Array<Tensor>;
    private _grad?: Tensor;

    // add backward()
    // call backward()
    // impl element wise mat mul

    set label(label: string) {
        this._label = label;
    }

    get label(): string {
        return this._label == "" ? "" : this._label;
    }

    get length(): number {
        return this._length;
    }

    get device(): Device {
        return this._device;
    }

    get requiresGrad(): boolean {
        return this._requiresGrad;
    }

    get grad(): Tensor {
        if (this._requiresGrad) {
            return this._grad as Tensor;
        }
        throw new Error("Tensor does not require gradient");
    }

    /**
    * Get the data of the Tensor
    *
    * Only available for CPU Tensors
    */
    get data(): Array<number> {
        if (this._device == "cpu") {
            return this._data as Array<number>;
        }
        throw new Error("Can only get data from CPU Tensors");
    }

    set data(d: Array<number>) {
        if (this._device == "cpu") {
            this._data = d;
        }

        throw new Error("TODO WGPU");
    }

    get shape(): Array<number> {
        return this._shape;
    }

    get gpuBuffer(): GPUBuffer {
        if (this._device == "wgpu") {
            return this._data as GPUBuffer;
        }

        throw new Error("Invalid device");
    }

    /**
    * Create a new Tensor with the given length
    *
    * @param lenght Length in items of the new Tensor
    * @param shape Shape of the Tensor
    * @param label Label for the Tensor
    * @param requiresGrad If the Tensor requires gradient
    * @param device Device where the Tensor will be created
    */
    constructor(length: number, shape: Array<number>, label?: string, requiresGrad?: boolean, device?: Device);
    /**
    * Create a new Tensor with the given Array as value
    *
    * @param items Array containing the items for this Tensor
    * @param shape Shape of the Tensor
    * @param label Label for the Tensor
    * @param requiresGrad If the Tensor requires gradient
    * @param device Device where the Tensor will be created
    */
    constructor(items: ArrayLike<number>, shape: Array<number>, label?: string, requiresGrad?: boolean, device?: Device);
    /**
    * Create a new Tensor with the given Iterable object as value
    *
    * @param items Iterable object containing the items for this Tensor
    * @param shape Shape of the Tensor
    * @param label Label for the Tensor
    * @param requiresGrad If the Tensor requires gradient
    * @param device Device where the Tensor will be created
    */
    constructor(items: Iterable<number>, shape: Array<number>, label?: string, requiresGrad?: boolean, device?: Device);

    constructor(
        e: Iterable<number> | ArrayLike<number> | number,
        shape: Array<number>,
        label: string = "",
        requiresGrad: boolean = false,
        device: Device = Tensor.defaultDevice
    ) {
        if (typeof e == "number") {
            this._data = new Array<number>(e).fill(0);
        } else {
            this._data = Array.from(e);
        }
        this._shape = shape;
        this._label = label;
        this._device = "cpu";
        this._requiresGrad = requiresGrad;
        this._childen = [];

        if (requiresGrad) {
            this._grad = new Tensor(this._data.length, this._shape, this.label + "_grad", false, device);
        }

        if (this._shape.length == 1) {
            this._shape = [1, this._shape[0]];
        }
        if (this._shape.length != 2) {
            throw new Error("Apologies, but at the moment we only support 2D Tensors");
        }

        this._length = this._shape.reduce((acc, value) => acc * value, 1);
        if (this._data.length != this.length) {
            throw new Error(`Invalid shape ${this._shape} for Tensor${this.label} with length ${this._data.length}`);
        }

        if (device == "wgpu") {
            this._data = WGPUProvider.moveTensorDataToGPU(this);
            this._device = "wgpu";
        }
    }

    /**
    * Return the number at the given position RxC
    *
    * @param i Row of the tensor that is being accessed
    * @param j Column of the tensor that is being accessed
    * @returns Number at the given position
    */
    at(i: number, j: number): number {
        return this.data[(i * this._shape[1]) + j];
    }

    /**
    * Set the number of the position given RxC
    *
    * @param i Row of the tensor that is being accessed
    * @param j Column of the tensor that is being accessed
    * @param value Value to set at the given position
    */
    set(i: number, j: number, value: number): void {
        this.data[(i * this._shape[1] + j)] = value;
    }

    /**
    * Multiply a Tensor by a scalar.
    * 
    * @param scalar Scalar to mul to the Tensor elements
    * @returns Tensor with the scalar multiplied to the elements
    */
    mul(scalar: number): Tensor;
    /**
    * Multiply a Tensor by another Tensor
    * 
    * @param tensor Other tensor to multiply against
    * @returns Tensor with the product of the two tensors
    */
    mul(tensor: Tensor): Tensor;

    mul(other: number | Tensor): Tensor {
        let res: Tensor;
        if (this._device == "cpu") {
            switch (typeof other) {
                case "number":
                    res = CPUOperations.mulScalar(this, other);
                    break;
                case "object":
                    res = CPUOperations.mulTensor(this, other);
                    break;
            }
        } else if (this._device == "wgpu") {
            switch (typeof other) {
                case "number":
                    res = GPUOperations.mulScalar(this, other);
                    break;
                case "object":
                    res = GPUOperations.mulTensor(this, other);
                    break;
            }
        } else {
            throw new Error("Invalid device");
        }

        this._childen.push(res);
        if (other instanceof Tensor) {
            other._childen.push(res);
            res.label = `(* ${this.label} ${other.label})`;
        } else {
            res.label = `(* ${this.label} ${other})`;
        }
        return res;
    }

    /**
    * Add a Scalar to a Tensor
    * 
    * @param scalar Scalar to add to the Tensor elements
    * @returns Tensor with the scalar added to the elements
    */
    add(scalar: number): Tensor;
    /**
    * Add a Tensor to a Tensor
    * 
    * @param tensor Other tensor to add
    * @returns Tensor with the sum of the two tensors
    */
    add(tensor: Tensor): Tensor;

    add(other: number | Tensor): Tensor {
        let res: Tensor;
        if (this._device == "cpu") {
            switch (typeof other) {
                case "number":
                    res = CPUOperations.addScalar(this, other);
                    break;
                case "object":
                    res = CPUOperations.addTensor(this, other);
                    break;
            }
        } else if (this._device == "wgpu") {
            switch (typeof other) {
                case "number":
                    res = GPUOperations.addScalar(this, other);
                    break;
                case "object":
                    res = GPUOperations.addTensor(this, other);
                    break;
            }
        } else {
            throw new Error("Invalid device");
        }

        this._childen.push(res);
        if (other instanceof Tensor) {
            other._childen.push(res);
            res.label = `(+ ${this.label} ${other.label})`;
        } else {
            res.label = `(+ ${this.label} ${other})`;
        }
        return res;
    }

    /**
    * Apply the ReLU activation function to the Tensor
    *
    * @returns Tensor with the ReLU applied
    */
    ReLU(): Tensor {
        let res: Tensor;
        if (this.device == "cpu") {
            res = CPUOperations.ReLU(this);
        } else if (this.device == "wgpu") {
            res = GPUOperations.ReLU(this);
        } else {
            throw new Error("Invalid device");
        }

        this._childen.push(res);
        res.label = `(ReLU ${this.label})`;
        return res;
    }

    /**
    * Apply the Leaky ReLU activation function to the Tensor
    *
    * @param alpha Alpha value for the Leaky ReLU function
    * @returns Tensor with the Leaky ReLU applied
    */
    leakyReLU(alpha: number = 0.01): Tensor {
        let res: Tensor;
        if (this.device == "cpu") {
            res = CPUOperations.leakyReLU(this, alpha);
        } else if (this.device == "wgpu") {
            res = GPUOperations.leakyReLU(this, alpha);
        } else {
            throw new Error("Invalid device");
        }

        this._childen.push(res);
        res.label = `(LeakyReLU ${this.label})`;
        return res;
    }

    /**
    * Transpose the Tensor
    *
    * @returns Tensor transposed
    */
    transpose(): Tensor {
        let res: Tensor;
        if (this.device == "cpu") {
            res = CPUOperations.transpose(this);
        } else if (this.device == "wgpu") {
            res = GPUOperations.transpose(this);
        } else {
            throw new Error("Invalid device");
        }

        this._childen.push(res);
        res.label = `(T ${this.label})`;
        return res;
    }

    /**
    * Raise the Tensor to the power of the exponent
    *
    * @param exponent Exponent to raise the Tensor to
    * @returns Tensor with the elements raised to the exponent
    */
    power(exponent: number): Tensor {
        let res: Tensor;
        if (this.device == "cpu") {
            res = CPUOperations.power(this, exponent);
        } else if (this.device == "wgpu") {
            res = GPUOperations.power(this, exponent);
        } else {
            throw new Error("Invalid device");
        }

        this._childen.push(res);
        res.label = `(^ ${this.label} ${exponent})`;
        return res;
    }

    /**
    * Element wise multiplication of two Tensors
    *
    * @param tensor Tensor to multiply against
    * @returns Tensor with the element wise multiplication of the two Tensors
    * */
    elemWiseMul(tensor: Tensor): Tensor {
        let res: Tensor;
        if (this.device == "cpu") {
            res = CPUOperations.elemWiseMul(this, tensor);
        } else if (this.device == "wgpu") {
            res = GPUOperations.elemWiseMul(this, tensor);
        } else {
            throw new Error("Invalid device");
        }

        this._childen.push(res);
        res.label = `(*e ${this.label} ${tensor.label})`;
        return res;
    }

    /**
    * Move the Tensor data to the GPU
    *
    * This function requires the WebGPU provider to be setup.
    */
    toGPU(): void {
        this._device = "wgpu";
        this._data = WGPUProvider.moveTensorDataToGPU(this);
    }

    /**
    * Move the Tensor data to the CPU
    *
    * @returns Promise that resolves when the data is copied to the CPU
    */
    async toCPU(): Promise<void> {
        if (this._device == "wgpu") {
            this._data = Array.from(await WGPUProvider.copyTensorDataToCPU(this));
            this._device = "cpu";
        }
    }

    /**
    * Prettify the Tensor and get it as a string
    *
    * @returns String with the Tensor prettified
    * */
    pretty(): string {
        let str = "";
        for (let i = 0; i < this._shape[0]; i++) {
            str += "[ ";
            for (let j = 0; j < this._shape[1]; j++) {
                str += this.at(i, j) + " ";
            }
            str += "]\n";
        }
        return str;
    }

    /**
    * Get the descendancy of the Tensor, including itself
    *
    * @returns Array with the descendancy of the Tensor
    */
    descendancy(): Array<string> {
        let des = [];
        des.push(this.label);
        this._childen.forEach((child) => {
            des.push(...child.descendancy());
        });
        return des;
    }
}
