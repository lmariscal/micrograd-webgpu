# micrograd-webgpu

![Adorable teal fluffy creature sitting in a spring forest](front.jpg)

WebGPU accelerated [micrograd](https://github.com/karpathy/micrograd).\
Autograd engine, implementing backpropagation and a simple neural-network library on top.\
It supports forward and backward passes on both CPU and GPU.

## Usage

It uses [bun](https://bun.sh/) for our js runtime, and bundling.

For managing the build and run commands, it is using just.\
So install [just](https://github.com/casey/just) if not already installed.

To simply serve the project:

```bash
just dev
```

The default port is set to 3000, but you can change it in the justfile.\
So you can access the project at `http://localhost:3000`.\
Chrome has been our main target for development, but it should work on other browsers as well.

It should display its progress in the browser's console.\
And a final forward pass showing the results of the training.

## Examples

The project includes a couple of examples in the test folder.\
Including training some basic neural networks on some logic gates.\
You can find those examples in the [tests folder](tests/).

TODO: Add more examples.\
TODO: Do a comparison between the CPU and GPU implementations.
