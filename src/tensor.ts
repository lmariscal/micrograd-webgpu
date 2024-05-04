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

    set label(label: string) {
        this._label = label;
    }

    get label(): string {
        return this._label == "" ? "" : "(" + this._label + ")" ;
    }

    get length(): number {
        return this._length;
    }

    get device(): Device {
        return this._device;
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
    * @param device Device where the Tensor will be created
    * @param label Label for the Tensor
    */
    constructor(length: number, shape: Array<number>, device?: Device, label?: string,);
    /**
    * Create a new Tensor with the given Array as value
    *
    * @param items Array containing the items for this Tensor
    * @param stride Number of columns that the Tensor contains
    * @param device Device where the Tensor will be created
    * @param label Label for the Tensor
    */
    constructor(items: ArrayLike<number>, shape: Array<number>, device?: Device, label?: string);
    /**
    * Create a new Tensor with the given Iterable object as value
    *
    * @param items Iterable object containing the items for this Tensor
    * @param stride Number of columns that the Tensor contains
    * @param device Device where the Tensor will be created
    * @param label Label for the Tensor
    */
    constructor(items: Iterable<number>, shape: Array<number>, device?: Device, label?: string);

    constructor(
        e: Iterable<number> | ArrayLike<number> | number,
        shape: Array<number>,
        device: Device = Tensor.defaultDevice,
        label: string = ""
    ) {
        if (typeof e == "number") {
            this._data = new Array<number>(e).fill(0);
        } else {
            this._data = Array.from(e);
        }
        this._shape = shape;
        this._label = label;
        this._device = "cpu";

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
        if (this._device == "cpu") {
            switch(typeof other) {
                case "number":
                    return CPUOperations.mulScalar(this, other);
                case "object":
                    return CPUOperations.mulTensor(this, other);
            }
        }
        else if (this._device == "wgpu") {
            switch(typeof other) {
                case "number":
                    return GPUOperations.mulScalar(this, other);
                case "object":
                    return GPUOperations.mulTensor(this, other);
            }
        }

        throw new Error("Invalid device");
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
        if (this._device == "cpu") {
            switch(typeof other) {
                case "number":
                    return CPUOperations.addScalar(this, other);
                case "object":
                    return CPUOperations.addTensor(this, other);
            }
        }
        else if (this._device == "wgpu") {
            switch(typeof other) {
                case "number":
                    return GPUOperations.addScalar(this, other);
                case "object":
                    return GPUOperations.addTensor(this, other);
            }
        }

        throw new Error("Invalid device");
    }

    /**
    * Apply the ReLU activation function to the Tensor
    *
    * @returns Tensor with the ReLU applied
    */
    ReLU(): Tensor {
        if (this.device == "cpu") {
            return CPUOperations.ReLU(this);
        }
        else if (this.device == "wgpu") {
            return GPUOperations.ReLU(this);
        }

        throw new Error("Invalid device");
    }

    /**
    * Apply the Leaky ReLU activation function to the Tensor
    *
    * @param alpha Alpha value for the Leaky ReLU function
    * @returns Tensor with the Leaky ReLU applied
    */
    leakyReLU(alpha: number = 0.01): Tensor {
        if (this.device == "cpu") {
            return CPUOperations.leakyReLU(this, alpha);
        }
        else if (this.device == "wgpu") {
            return GPUOperations.leakyReLU(this, alpha);
        }

        throw new Error("Invalid device");
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
}
