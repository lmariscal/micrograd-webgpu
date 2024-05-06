@group(0) @binding(0)
var<storage, read> inputShape: array<u32>;
@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read> otherShape: array<u32>;
@group(0) @binding(3)
var<storage, read> other: array<f32>;

@group(0) @binding(4)
var<storage, read_write> result: array<f32>;

// not creating a Tensor struct since it wouldn't allow for n-dim shapes
// - arrays are only allowed as last element in a struct

@compute @workgroup_size(8, 8)
fn addScalar(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = input[index] + other[0];
}

@compute @workgroup_size(8, 8)
fn addTensor(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = input[index] + other[index];
}

@compute @workgroup_size(8, 8)
fn subScalar(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = input[index] - other[0];
}

@compute @workgroup_size(8, 8)
fn subTensor(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = input[index] - other[index];
}

@compute @workgroup_size(8, 8)
fn mulScalar(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = input[index] * other[0];
}

@compute @workgroup_size(8, 8)
fn mulTensor(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let row = global_id.x;
    let col = global_id.y;
    var sum: f32 = 0;

    for (var k: u32 = 0; k < inputShape[1]; k++) {
        sum += input[(row * inputShape[1]) + k] * other[(k * otherShape[1]) + col];
    }

    result[(row * otherShape[1] + col)] = sum;
}

@compute @workgroup_size(8, 8)
fn ReLU(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = max(0.0, input[index]);
}

@compute @workgroup_size(8, 8)
fn leakyReLU(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = max(other[0] * input[index], input[index]);
}

@compute @workgroup_size(8, 8)
fn transpose(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let row = global_id.x;
    let col = global_id.y;
    result[(col * inputShape[0]) + row] = input[(row * inputShape[1]) + col];
}

@compute @workgroup_size(8, 8)
fn power(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = pow(input[index], other[0]);
}

@compute @workgroup_size(8, 8)
fn elemWiseMul(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = input[index] * other[index];
}

@compute @workgroup_size(8, 8)
fn ReLUPrime(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    if (input[index] > 0.0) {
        result[index] = 1.0;
    } else {
        result[index] = 0.0;
    }
}

@compute @workgroup_size(8, 8)
fn powerPrime(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    result[index] = other[0] * pow(input[index], other[0] - 1.0);
}

@compute @workgroup_size(8, 8)
fn leakyReLUPrime(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if (global_id.x >= inputShape[0] || global_id.y >= inputShape[1]) {
        return;
    }
    let index = global_id.y + (global_id.x * inputShape[1]);
    if (input[index] > 0.0) {
        result[index] = 1.0;
    } else {
        result[index] = other[0];
    }
}
