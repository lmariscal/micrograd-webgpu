export class WGPUProvider {
    valid: boolean = false;

    device?: GPUDevice;
    adapter?: GPUAdapter;
    canvas?: HTMLCanvasElement;
    context?: GPUCanvasContext;

    async setup(canvas: HTMLCanvasElement): Promise<boolean> {
        this.canvas = canvas;

        const _adapter = await navigator.gpu.requestAdapter({
            powerPreference: "high-performance"
        });
        if (!_adapter) {
            return false;
        }
        this.adapter = _adapter;

        this.device = await this.adapter.requestDevice();
        if (!this.device) {
            return false;
        }

        const _context = this.canvas.getContext("webgpu");
        if (!_context) {
            return false;
        }
        this.context = _context;

        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: "premultiplied"
        });

        this.valid = true;
        return true;
    }
}
