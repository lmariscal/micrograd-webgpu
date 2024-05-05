import { Tensor } from "./tensor";

export class WGPUProvider {
    static device?: GPUDevice;
    static adapter?: GPUAdapter;

    /**
    * Setup the WebGPU provider
    *
    * @returns True if the provider was setup successfully or false otherwise
    */
    static async setup(): Promise<boolean> {
        if (!navigator.gpu) {
            return false;
        }

        const _adapter = await navigator.gpu.requestAdapter({
            powerPreference: "high-performance"
        });
        if (!_adapter) {
            return false;
        }
        this.adapter = _adapter;

        WGPUProvider.device = await this.adapter.requestDevice();
        if (!WGPUProvider.device) {
            return false;
        }

        return true;
    }

    /**
    * Move the Tensor data to the GPU
    *
    * Do not use this function directly. Use the `Tensor.toGPU()` method instead.
    *
    * @param tensor Tensor to move to the GPU
    * @returns GPUBuffer with the data of the Tensor
    * @throws Error if the Tensor is already in the GPU
    */
    static moveTensorDataToGPU(tensor: Tensor): GPUBuffer   {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }
        if (tensor.device == "wgpu") {
            throw new Error("Tensor already in GPU");
        }

        const buffer = WGPUProvider.device.createBuffer({
            label: `tensor${tensor.label} buffer`,
            size: Float32Array.BYTES_PER_ELEMENT * tensor.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        });
        {
            const writeBuffer = new Float32Array(buffer?.getMappedRange() as ArrayBuffer);
            writeBuffer.set(tensor.data);
        }
        buffer.unmap();

        return buffer;
    }

    /**
    * Copy the Tensor data to the CPU
    *
    * Do not use this function directly. Use the `Tensor.toCPU()` method instead.
    *
    * @param tensor Tensor to copy to the CPU
    * @returns Float32Array with the data of the Tensor
    * @throws Error if the Tensor is already in the CPU
    */
    static async copyTensorDataToCPU(tensor: Tensor): Promise<Float32Array> {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }
        if (tensor.device == "cpu") {
            throw new Error("Tensor already in CPU");
        }

        const BUFFER_LENGTH = Float32Array.BYTES_PER_ELEMENT * tensor.length;
        const buffer = WGPUProvider.device.createBuffer({
            label: `tensor${tensor.label} dst buffer`,
            size: BUFFER_LENGTH,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = WGPUProvider.device.createCommandEncoder();
        encoder.copyBufferToBuffer(tensor.gpuBuffer, 0, buffer, 0, BUFFER_LENGTH);
        WGPUProvider.device.queue.submit([encoder.finish()]);

        await buffer.mapAsync(GPUMapMode.READ, 0, BUFFER_LENGTH);
        const copyArray = new Float32Array(buffer.getMappedRange(0, BUFFER_LENGTH));
        const data = copyArray.slice();
        buffer.unmap();

        return data;
    }
}

if (!(await WGPUProvider.setup())) {
    if (typeof Bun === "undefined") {
        console.error("Failed to setup WebGPU provider");
        if (!navigator.gpu) {
            console.error("WebGPU not supported");
        }
    }
} else {
    console.debug("WebGPU provider is ready");
}
