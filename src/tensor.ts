import { CPUOperations } from "./operations";

type Device = "cpu" | "wgpu";

// TODO: Actually make this Tensor class a n-dim Tensor rather than just a 2D tensor
export class Tensor {
    private _data: Array<number>;
    private _shape: Array<number>; // RxC
    private _label: string;
    private _device: Device;

    set label(label: string) {
        this._label = label;
    }

    get label(): string {
        return this._label == "" ? "" : "(" + this._label + ")" ;
    }

    get device(): Device {
        return this._device;
    }

    get data(): Array<number> {
        if (this._device == "cpu") {
            return this._data;
        }

        throw new Error("TODO WGPU");
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

    /**
     * Create a new Tensor with the given length
     * 
     * @param lenght Length in items of the new Tensor
     * @param shape Shape of the Tensor
     */
    constructor(length: number, shape: Array<number>, label?: string, device?: Device);
    /**
     * Create a new Tensor with the given Array as value
     * 
     * @param items Array containing the items for this Tensor
     * @param stride Number of columns that the Tensor contains
     */
    constructor(items: ArrayLike<number>, shape: Array<number>, label?: string, device?: Device);
    /**
     * Create a new Tensor with the given Iterable object as value
     * 
     * @param items Iterable object containing the items for this Tensor
     * @param stride Number of columns that the Tensor contains
     */
    constructor(items: Iterable<number>, shape: Array<number>, label?: string, device?: Device);

    constructor(e: Iterable<number> | ArrayLike<number> | number, shape: Array<number>, label: string = "", device: Device = "cpu") {
        if (typeof e == "number") {
            this._data = new Array<number>(e).fill(0);
        } else {
            this._data = Array.from(e);
        }
        this._shape = shape;
        this._label = label;
        this._device = device;

        if (this._shape.length == 1) {
            this._shape = [1, this._shape[0]];
        }
        if (this._shape.length != 2) {
            throw new Error("Apologies, but at the moment we only support 2D Tensors");
        }

        const product = this._shape.reduce((acc, value) => acc * value, 1);
        if (this._data.length != product) {
            throw new Error(`Invalid shape ${this._shape} for Tensor${this.label} with length ${this._data.length}`);
        }
    }

    at(i: number, j: number): number {
        return this._data[(i * this._shape[1]) + j];
    }

    /**
     * Multiply a Tensor by a scalar.
     * 
     * @param scalar Scalar to mul to the Tensor elements
     */
    mul(scalar: number): Tensor;
    /**
     * Multiply a Tensor by another Tensor
     * 
     * @param tensor Other tensor to multiply against
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

        throw new Error("TODO WGPU");
    }

    /**
     * Add a Scalar to a Tensor
     * 
     * @param scalar Scalar to add to the Tensor elements
     */
    add(scalar: number): Tensor;
    /**
     * Add a Tensor to a Tensor
     * 
     * @param tensor Other tensor to add
     */
    add(tensor: number): Tensor;

    add(other: number | Tensor): Tensor {
        if (this._device == "cpu") {
            switch(typeof other) {
                case "number":
                    return CPUOperations.addScalar(this, other);
                case "object":
                    return CPUOperations.addTensor(this, other);
            }
        }

        throw new Error("TODO WGPU");
    }
}