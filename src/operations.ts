import { Tensor } from "./tensor";
import { WGPUProvider } from "./wgpu_provider";
import operationsWGSL from "./operations.comp.wgsl";

export class CPUOperations {
    static mulTensor(tensor: Tensor, other: Tensor): Tensor {
        // TODO: Not use a naive approach
        // ~TODO: Support n-dim Tensors~ (Not a goal of this project)

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

        let dst = new Tensor(tensor.length, tensor.shape);
        for (let i = 0; i < tensor.length; ++i) {
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

    static leakyReLU(tensor: Tensor, alpha: number): Tensor {
        const data = tensor.data.map((x) => x > 0 ? x : x * alpha);
        return new Tensor(data, tensor.shape);
    }
}

export class GPUOperations {
    static shaderModule: GPUShaderModule | undefined;
    static inputShapeBuffer: GPUBuffer | undefined;
    static otherShapeBuffer: GPUBuffer | undefined;
    static bindGroupLayout: GPUBindGroupLayout | undefined;

    static addScalarPipeline: GPUComputePipeline | undefined;
    static addTensorPipeline: GPUComputePipeline | undefined;
    static mulScalarPipeline: GPUComputePipeline | undefined;
    static mulTensorPipeline: GPUComputePipeline | undefined;
    static ReLUPipeline: GPUComputePipeline | undefined;
    static leakyReLUPipeline: GPUComputePipeline | undefined;

    static setup(): Promise<boolean> {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }

        // input shape buffer
        const inputShapeBuffer = WGPUProvider.device.createBuffer({
            label: "input shape buffer",
            size: Uint32Array.BYTES_PER_ELEMENT * 2,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        GPUOperations.inputShapeBuffer = inputShapeBuffer;

        // other shape buffer
        const otherShapeBuffer = WGPUProvider.device.createBuffer({
            label: "other shape buffer",
            size: Uint32Array.BYTES_PER_ELEMENT * 2,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        GPUOperations.otherShapeBuffer = otherShapeBuffer;

        // shader module
        const shaderModule = WGPUProvider.device.createShaderModule({
            label: "operations compute shader",
            code: operationsWGSL
        });
        GPUOperations.shaderModule = shaderModule;

        // pipeline layout
        const bindGroupLayout = WGPUProvider.device.createBindGroupLayout({
            label: "operatioins bind group layout",
            entries: [
                {
                    binding: 0, // inputShape
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 1, // input
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2, // otherShape
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 3, // other
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 4, // result
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                }
            ]
        });
        GPUOperations.bindGroupLayout = bindGroupLayout;

        const pipelineLayout = WGPUProvider.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        // add scalar pipeline
        const addScalarPipeline = WGPUProvider.device.createComputePipeline({
            label: "add scalar compute pipeline",
            compute: {
                module: shaderModule,
                entryPoint: "addScalar"
            },
            layout: pipelineLayout
        });
        GPUOperations.addScalarPipeline = addScalarPipeline;

        // add tensor pipeline
        const addTensorPipeline = WGPUProvider.device.createComputePipeline({
            label: "add tensor compute pipeline",
            compute: {
                module: shaderModule,
                entryPoint: "addTensor"
            },
            layout: pipelineLayout
        });
        GPUOperations.addTensorPipeline = addTensorPipeline;

        // mul scalar pipeline
        const mulScalarPipeline = WGPUProvider.device.createComputePipeline({
            label: "mul scalar compute pipeline",
            compute: {
                module: shaderModule,
                entryPoint: "mulScalar"
            },
            layout: pipelineLayout
        });
        GPUOperations.mulScalarPipeline = mulScalarPipeline;

        // mul tensor pipeline
        const mulTensorPipeline = WGPUProvider.device.createComputePipeline({
            label: "mul tensor compute pipeline",
            compute: {
                module: shaderModule,
                entryPoint: "mulTensor"
            },
            layout: pipelineLayout
        });
        GPUOperations.mulTensorPipeline = mulTensorPipeline;

        // ReLU pipeline
        const ReLUPipeline = WGPUProvider.device.createComputePipeline({
            label: "ReLU compute pipeline",
            compute: {
                module: shaderModule,
                entryPoint: "ReLU"
            },
            layout: pipelineLayout
        });
        GPUOperations.ReLUPipeline = ReLUPipeline;

        // leaky ReLU pipeline
        const leakyReLUPipeline = WGPUProvider.device.createComputePipeline({
            label: "leaky ReLU compute pipeline",
            compute: {
                module: shaderModule,
                entryPoint: "leakyReLU"
            },
            layout: pipelineLayout
        });
        GPUOperations.leakyReLUPipeline = leakyReLUPipeline;

        return Promise.resolve(true);
    }

    private static createBindGroup(label: string, input: Tensor, other: GPUBuffer | null, result: GPUBuffer): GPUBindGroup {
        const bindGroup = WGPUProvider.device!.createBindGroup({
            label: `${label} bind group`,
            layout: GPUOperations.bindGroupLayout!,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: GPUOperations.inputShapeBuffer!
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: input.gpuBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: GPUOperations.otherShapeBuffer!
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: other ? other : input.gpuBuffer
                    }
                },
                {
                    binding: 4,
                    resource: {
                        buffer: result
                    }
                }
            ]
        });

        return bindGroup;
    }

    private static doComputePass(label: string, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroupsSize: Array<number>): void {
        const pass = WGPUProvider.device!.createCommandEncoder();
        const computePass = pass.beginComputePass();
        computePass.label = `${label} compute pass`;
        computePass.setPipeline(pipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(workgroupsSize[0] / 8),
            Math.ceil(workgroupsSize[1] / 8));
        computePass.end();
        WGPUProvider.device!.queue.submit([pass.finish()]);
    }

    private static createScalarBuffer(label: string, scalar: number): GPUBuffer {
        const scalarBuffer = WGPUProvider.device!.createBuffer({
            label: `${label} scalar buffer`,
            size: Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true
        });
        new Float32Array(scalarBuffer.getMappedRange()).set([scalar]);
        scalarBuffer.unmap();
        return scalarBuffer;
    }

    static addScalar(tensor: Tensor, scalar: number): Tensor {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }

        const res = new Tensor(tensor.length, tensor.shape, "wgpu");
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.inputShapeBuffer!, 0, new Uint32Array(tensor.shape));
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.otherShapeBuffer!, 0, new Uint32Array([1, 1]));

        const scalarBuffer = GPUOperations.createScalarBuffer("add", scalar);
        const bindGroup = GPUOperations.createBindGroup("add scalar", tensor, scalarBuffer, res.gpuBuffer);
        GPUOperations.doComputePass("add scalar", GPUOperations.addScalarPipeline!, bindGroup, tensor.shape);
        return res;
    }

    static addTensor(tensor: Tensor, other: Tensor): Tensor {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }

        if (tensor.shape[0] != other.shape[0] || tensor.shape[1] != other.shape[1]) {
            throw new Error(`Shapes of Tensors mismatch. ${tensor.label}(${tensor.shape}) vs ${other.label}(${other.shape})`)
        }
        const res = new Tensor(tensor.length, tensor.shape, "wgpu");
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.inputShapeBuffer!, 0, new Uint32Array(tensor.shape));
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.otherShapeBuffer!, 0, new Uint32Array(other.shape));

        const bindGroup = GPUOperations.createBindGroup("add tensor", tensor, other.gpuBuffer, res.gpuBuffer);
        GPUOperations.doComputePass("add tensor", GPUOperations.addTensorPipeline!, bindGroup, tensor.shape);
        return res;
    }

    static mulScalar(tensor: Tensor, scalar: number): Tensor {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }

        const res = new Tensor(tensor.length, tensor.shape, "wgpu");
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.inputShapeBuffer!, 0, new Uint32Array(tensor.shape));
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.otherShapeBuffer!, 0, new Uint32Array([1, 1]));

        const scalarBuffer = GPUOperations.createScalarBuffer("mul", scalar);
        const bindGroup = GPUOperations.createBindGroup("mul scalar", tensor, scalarBuffer, res.gpuBuffer);
        GPUOperations.doComputePass("mul scalar", GPUOperations.mulScalarPipeline!, bindGroup, tensor.shape);
        return res;
    }

    static mulTensor(tensor: Tensor, other: Tensor): Tensor {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }

        if (tensor.shape[1] != other.shape[0]) {
            throw new Error(`Tensor multiplication not set for given mismatched Tensors - ${tensor.shape} vs ${other.shape}`);
        }

        const res = new Tensor(tensor.shape[0] * other.shape[1], [tensor.shape[0], other.shape[1]], "wgpu");
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.inputShapeBuffer!, 0, new Uint32Array(tensor.shape));
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.otherShapeBuffer!, 0, new Uint32Array(other.shape));

        const bindGroup = GPUOperations.createBindGroup("mul tensor", tensor, other.gpuBuffer, res.gpuBuffer);
        GPUOperations.doComputePass("mul tensor", GPUOperations.mulTensorPipeline!, bindGroup, [tensor.shape[0], other.shape[1]]);
        return res;
    }

    static ReLU(tensor: Tensor): Tensor {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }

        const res = new Tensor(tensor.length, tensor.shape, "wgpu");
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.inputShapeBuffer!, 0, new Uint32Array(tensor.shape));
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.otherShapeBuffer!, 0, new Uint32Array([1, 1]));

        const bindGroup = GPUOperations.createBindGroup("ReLU", tensor, null, res.gpuBuffer);
        GPUOperations.doComputePass("ReLu", GPUOperations.ReLUPipeline!, bindGroup, tensor.shape);
        return res;
    }

    static leakyReLU(tensor: Tensor, alpha: number): Tensor {
        if (!WGPUProvider.device) {
            throw new Error("WebGPU provider not setup");
        }

        const res = new Tensor(tensor.length, tensor.shape, "wgpu");
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.inputShapeBuffer!, 0, new Uint32Array(tensor.shape));
        WGPUProvider.device!.queue.writeBuffer(GPUOperations.otherShapeBuffer!, 0, new Uint32Array([1, 1]));

        const alphaBuffer = GPUOperations.createScalarBuffer("leakyReLU", alpha);
        const bindGroup = GPUOperations.createBindGroup("leakyReLU", tensor, alphaBuffer, res.gpuBuffer);
        GPUOperations.doComputePass("leaky ReLU", GPUOperations.leakyReLUPipeline!, bindGroup, tensor.shape);
        return res;
    }
}

if (!(await GPUOperations.setup())) {
    console.error("Failed to setup GPU Operations module");
} else {
    console.debug("GPU Operations module is ready");
}
