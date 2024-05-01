import { WGPUProvider } from "./wgpu_provider";

async function main() {
    const provider = new WGPUProvider();
    const canvas = document.createElement("canvas");
    document.body.appendChild(canvas);

    if (!provider.setup(canvas)) {
        console.error("Failed to setup WebGPU provider");
        return;
    }

    console.debug("WebGPU provider is ready");
}

main();
