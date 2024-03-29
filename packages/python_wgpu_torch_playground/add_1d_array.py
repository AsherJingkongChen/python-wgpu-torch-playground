"""
A simple example to compute the sum of two numpy arrays
and convert the result to a torch tensor.
"""

__all__ = ["main"]


def main():
    from pathlib import Path
    from pprint import pprint as print
    import numpy
    import torch
    import wgpu

    # Define the number of elements, global and local sizes.
    # Change these and see how it affects performance.
    n = 1 << 14
    global_size = [n, 1, 1]

    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")

    # Request a device.
    device = adapter.request_device()
    print("")
    print(device.adapter.request_adapter_info())
    print(device.limits)

    data1 = numpy.arange(0, n, 1, dtype=numpy.int32)
    data2 = (data1 * 2).astype(numpy.int32)

    # Create buffer objects, input buffer is mapped.
    buffer1 = device.create_buffer_with_data(
        data=data1,
        usage=wgpu.BufferUsage.STORAGE,
    )
    buffer2 = device.create_buffer_with_data(
        data=data2,
        usage=wgpu.BufferUsage.STORAGE,
    )
    buffer3 = device.create_buffer(
        size=data1.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    # Setup layout and bindings
    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                },
            },
        ]
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": buffer1,
                    "offset": 0,
                    "size": buffer1.size,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": buffer2,
                    "offset": 0,
                    "size": buffer2.size,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": buffer3,
                    "offset": 0,
                    "size": buffer3.size,
                },
            },
        ],
    )

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        ),
        compute={
            "module": device.create_shader_module(
                code=(
                    Path(__file__).with_name("add_1d_array.wgsl").open().read()
                )
            ),
            "entry_point": "main",
        },
    )

    command_encoder = device.create_command_encoder()
    pass_encoder = command_encoder.begin_compute_pass()

    pass_encoder.set_pipeline(compute_pipeline)
    pass_encoder.set_bind_group(0, bind_group, [], None, None)
    pass_encoder.dispatch_workgroups(*global_size)  # x y z
    pass_encoder.end()

    device.queue.submit([command_encoder.finish()])

    # Read result
    outview = device.queue.read_buffer(buffer3)
    result = torch.frombuffer(outview, dtype=torch.int32)

    # Calculate the result on the CPU for comparison
    result_cpu = torch.from_numpy(data1 + data2)

    # Ensure results are the same
    assert result.equal(result_cpu)

    print(f"{__name__}: Ok")
