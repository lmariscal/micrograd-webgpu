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

@compute @workgroup_size(1)
fn addScalar(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let index = global_id.y + (global_id.x * inputShape[1]);
    if (index >= (inputShape[0] * inputShape[1])) {
        return;
    }
    result[index] = input[index] + other[0];
}

@compute @workgroup_size(1)
fn addTensor(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let index = global_id.y + (global_id.x * inputShape[1]);
    if (index >= (inputShape[0] * inputShape[1])) {
        return;
    }
    result[index] = input[index] + other[index];
}

@compute @workgroup_size(1)
fn mulScalar(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let index = global_id.y + (global_id.x * inputShape[1]);
    if (index >= (inputShape[0] * inputShape[1])) {
        return;
    }
    result[index] = input[index] * other[0];
}

@compute @workgroup_size(1)
fn mulTensor(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let row = global_id.x;
    let col = global_id.y;
    var sum: f32 = 0;
    for (var k: u32 = 0; k < inputShape[1]; k++) {
        sum += input[(row * inputShape[1]) + k] * other[(k * otherShape[1]) + col];
    }
    result[(row * otherShape[1] + col)] = sum;
}

@compute @workgroup_size(1)
fn ReLU(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let index = global_id.y + (global_id.x * inputShape[1]);
    if (index >= (inputShape[0] * inputShape[1])) {
        return;
    }
    result[index] = max(0.0, input[index]) + 1;
}

@compute @workgroup_size(1)
fn leakyReLU(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let index = global_id.y + (global_id.x * inputShape[1]);
    if (index >= (inputShape[0] * inputShape[1])) {
        return;
    }
    result[index] = max(0.1 * input[index], input[index]);
}
