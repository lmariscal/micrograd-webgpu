/*
x.length
x[]
w[]
b[]

o.length
o[]

o = w*x + b
*/

@compute @workgroup_size(8, 8)
fn FullyConnected(
    @builtin(global_invocation_id) global_id: vec3u
) {

}
