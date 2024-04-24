import { Tensor } from "./tensor";

async function main() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter?.requestDevice();
    
    if (!device) {
        console.error("Your device does not support WebGPU at the moment...");
        return;
    }

    let t = new Tensor([1, 2, 3, 4,
                        5, 6, 7, 8], [2, 4]);
    console.log(t);
    console.info(device)
}

main();