import { CPUOperations, GPUOperations } from "./operations";
import { WGPUProvider } from "./wgpu_provider";

type Device = "cpu" | "wgpu";

// ~TODO: Actually make this Tensor class a n-dim Tensor rather than just a 2D tensor~ (Not a goal of this project)
export class Tensor {
    public static defaultDevice: Device = "cpu";
    public static debugMode: boolean = false;

    private _label: string;
    private _device: Device;
    private _data: Array<number> | GPUBuffer;
    private _shape: Array<number>; // RxC
    private _length: number;
    private _requiresGrad: boolean;
    private _children: Array<Tensor>;
    private _grad?: Tensor;
    private _backward?: Function;

    // add backward()
    // call backward()

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
        // I don't wanna do data async
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

    get children(): Array<Tensor> {
        return this._children;
    }

    randomize(): void {
        this._data = new Array<number>(this.length).fill(0).map(() => Math.random());
        if (this._device == "wgpu") {
            this._device = "cpu";
            this._data = WGPUProvider.moveTensorDataToGPU(this);
            this._device = "wgpu";

        }
    }

    zeroGrad(): void {
        if (this._requiresGrad) {
            this._grad = new Tensor(this.length, this._shape, this.label + " grad", false, this._device);
            for (let child of this._children) {
                child.zeroGrad();
            }
        }
        this._children = [];
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
        requiresGrad: boolean = true,
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
        this._children = [];

        if (requiresGrad) {
            this._grad = new Tensor(this._data.length, this._shape, this.label + " grad", false, device);
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

        res._children = [];
        res._children.push(this);
        if (other instanceof Tensor) {
            res._children.push(other);
            if (Tensor.debugMode) {
                res.label = `(* ${this.label} ${other.label})`;
            }

            res._backward = () => {
                this._grad = this._grad!.add(res.grad.mul(other.transpose()));
                other._grad = other._grad!.add(this.transpose().mul(res.grad));
            };
        } else {
            if (Tensor.debugMode) {
                res.label = `(* ${this.label} ${other})`;
            }

            res._backward = () => {
                this._grad = this._grad!.add(res.grad.mul(other));
            };
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

        res._children = [];
        res._children.push(this);
        if (other instanceof Tensor) {
            res._children.push(other);
            if (Tensor.debugMode) {
                res.label = `(+ ${this.label} ${other.label})`;
            }

            res._backward = () => {
                this._grad = this._grad!.add(res.grad);
                other._grad = other._grad!.add(res.grad);
            };
        } else {
            if (Tensor.debugMode) {
                res.label = `(+ ${this.label} ${other})`;
            }

            res._backward = () => {
                this._grad = this._grad!.add(res.grad);
            };
        }
        return res;
    }

    /**
    * Subtract a Scalar from a Tensor
    *
    * @param scalar Scalar to subtract from the Tensor elements
    * @returns Tensor with the scalar subtracted from the elements
    */
    sub(scalar: number): Tensor;
    /*
    * Subtract a Tensor from a Tensor
    *
    * @param other Other tensor to subtract
    * @returns Tensor with the difference of the two tensors
    */
    sub(other: Tensor): Tensor;

    sub(other: number | Tensor): Tensor {
        let res: Tensor;
        if (this._device == "cpu") {
            switch (typeof other) {
                case "number":
                    res = CPUOperations.addScalar(this, -other);
                    break;
                case "object":
                    res = CPUOperations.subTensor(this, other);
                    break;
            }
        } else if (this._device == "wgpu") {
            switch (typeof other) {
                case "number":
                    res = GPUOperations.addScalar(this, -other);
                    break;
                case "object":
                    res = GPUOperations.subTensor(this, other);
                    break;
            }
        } else {
            throw new Error("Invalid device");
        }

        res._children = [];
        res._children.push(this);
        if (other instanceof Tensor) {
            res._children.push(other);
            if (Tensor.debugMode) {
                res.label = `(- ${this.label} ${other.label})`;
            }

            res._backward = () => {
                this._grad = this._grad!.add(res.grad);
                other._grad = other._grad!.add(res.grad.mul(-1));
            };
        } else {
            if (Tensor.debugMode) {
                res.label = `(- ${this.label} ${other})`;
            }

            res._backward = () => {
                this._grad = this._grad!.add(res.grad);
            };
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

        res._children = [];
        res._children.push(this);
        if (Tensor.debugMode) {
            res.label = `(ReLU ${this.label})`;
        }

        res._backward = () => {
            if (this._device == "cpu") {
                this._grad = this._grad!.add(CPUOperations.ReLUPrime(this).elemWiseMul(res.grad));
            } else if (this._device == "wgpu") {
                this._grad = this._grad!.add(GPUOperations.ReLUPrime(this).elemWiseMul(res.grad));
            }
        };
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

        res._children = [];
        res._children.push(this);
        if (Tensor.debugMode) {
            res.label = `(LeakyReLU ${this.label})`;
        }

        res._backward = () => {
            if (this._device == "cpu") {
                this._grad = this._grad!.add(CPUOperations.leakyReLUPrime(this, alpha).elemWiseMul(res.grad));
            } else if (this._device == "wgpu") {
                this._grad = this._grad!.add(GPUOperations.leakyReLUPrime(this, alpha).elemWiseMul(res.grad));
            }
        }
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

        res._children = [];
        res._children.push(this);
        if (Tensor.debugMode) {
            res.label = `(T ${this.label})`;
        }
        return res;
    }

    /**
    * Raise the Tensor to the power of the exponent
    *
    * @param exponent Exponent to raise the Tensor to
    * @returns Tensor with the elements raised to the exponent
    */
    pow(exponent: number): Tensor {
        let res: Tensor;
        if (this.device == "cpu") {
            res = CPUOperations.power(this, exponent);
        } else if (this.device == "wgpu") {
            res = GPUOperations.power(this, exponent);
        } else {
            throw new Error("Invalid device");
        }

        res._children = [];
        res._children.push(this);
        if (Tensor.debugMode) {
            res.label = `(^ ${this.label} ${exponent})`;
        }

        res._backward = () => {
            if (this._device == "cpu") {
                this._grad = this._grad!.add(CPUOperations.powerPrime(this, exponent).elemWiseMul(res.grad));
            } else if (this._device == "wgpu") {
                this._grad = this._grad!.add(GPUOperations.powerPrime(this, exponent).elemWiseMul(res.grad));
            }
        };
        return res;
    }

    /**
    * Element wise multiplication of two Tensors
    *
    * @param other Tensor to multiply against
    * @returns Tensor with the element wise multiplication of the two Tensors
    * */
    elemWiseMul(other: Tensor): Tensor {
        let res: Tensor;
        if (this.device == "cpu") {
            res = CPUOperations.elemWiseMul(this, other);
        } else if (this.device == "wgpu") {
            res = GPUOperations.elemWiseMul(this, other);
        } else {
            throw new Error("Invalid device");
        }

        res._children = [];
        res._children.push(this);
        if (Tensor.debugMode) {
            res.label = `(*e ${this.label} ${other.label})`;
        }

        res._backward = () => {
            this._grad = this._grad!.add(res.grad.elemWiseMul(other));
            other._grad = other._grad!.add(res.grad.elemWiseMul(this));
        };
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

        if (this._requiresGrad) {
            this._grad!.toGPU();
        }
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

            if (this._requiresGrad) {
                await this._grad!.toCPU();
            }
        }
    }

    /**
    * Duplicate the Tensor, creating a new Tensor with the same data
    *
    * @returns New Tensor with the same data
    */
    copy(): Tensor {
        let cpy: Tensor;
        if (this._device == "cpu") {
            cpy = new Tensor(this._data as Array<number>, this._shape, this.label, this._requiresGrad, this._device);
        } else {
            cpy = WGPUProvider.duplicateTensor(this);
        }

        if (this._requiresGrad) {
            cpy._grad = this._grad!.copy();
        }
        return cpy;
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
        this._children.forEach((child) => {
            des.push(...child.descendancy());
        });
        return des;
    }

    backward(): void {
        if (!this.requiresGrad) {
            throw new Error("Tensor does not require gradient");
        }
        this._grad = this._grad!.add(1);

        let topo: Array<Tensor> = [];
        let visited = new Set<Tensor>();

        let dfs = (node: Tensor) => {
            visited.add(node);
            node._children.forEach((child) => {
                if (!visited.has(child)) {
                    dfs(child);
                }
            });
            topo.push(node);
        };

        dfs(this);
        topo.reverse();
        if (Tensor.debugMode) {
            console.log("topo length:", topo.length);
        }

        for (let t of topo) {
            if (t._backward) {
                t._backward();
            }
        }
    }
}
