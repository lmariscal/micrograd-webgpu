export class Tensor extends Float32Array {
    shape: Array<number>; // RxC

    /**
     * Create a new Tensor with the given length
     * @param lenght Length in items of the new Tensor
     * @param shape Shape of the Tensor
     */
    constructor(length: number, shape: Array<number>);
    /**
     * Create a new Tensor with the given Array as value
     * @param items Array containing the items for this Tensor
     * @param stride Number of columns that the Tensor contains
     */
    constructor(items: ArrayLike<number>, shape: Array<number>);
    /**
     * Create a new Tensor with the given Iterable object as value
     * @param items Iterable object containing the items for this Tensor
     * @param stride Number of columns that the Tensor contains
     */
    constructor(items: Iterable<number>, shape: Array<number>);

    constructor(e: Iterable<number> | ArrayLike<number> | number, shape: Array<number>) {
        if (typeof e == "number") {
            super(e);
        } else if (Symbol.iterator in Object(e)) {
            super(e as Iterable<number>);
        } else {
            super(e as ArrayLike<number>);
        }

        this.shape = shape;
        const product = this.shape.reduce((acc, value) => acc * value, 1);
        if (this.length != product) {
            throw new Error(`Invalid shape ${this.shape} for Tensor with length ${this.length}`);
        }
    }

    /**
     * Multiply this Tensor by a scalar.
     */
    mul(scalar: number): void;
    /**
     * Multiply this Tensor by another Tensor
     */
    mul(tensor: Tensor): void;

    mul(other: number | Tensor): void {
        // TODO
    }
}