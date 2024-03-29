@group(0) @binding(0)
var<storage,read_write> data: vec3<i32>;

@compute @workgroup_size(1)
fn main() {
    let data1 = vec3<i32>(1, 2, 3);
    let data2 = vec3<i32>(4, -5, 6);
    let data3 = vec3<i32>(6, 0, 4);
    data = min(max(data1, data2), data3);
}
