"""
A simple example to profile a compute pass using ComputePassTimestampWrites.
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

    # Request a device with the timestamp_query feature, so we can profile our computation
    device = wgpu.gpu.request_adapter(power_preference="low-power").request_device(
        required_features=[wgpu.FeatureName.timestamp_query]
    )
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
        layout=device.create_pipeline_layout(bind_group_layouts=[bind_group_layout]),
        compute={
            "module": device.create_shader_module(
                code=(
                    Path(__file__)
                    .with_name("add_1d_array.wgsl")
                    .open()
                    .read()
                )
            ),
            "entry_point": "main",
        },
    )

    # Create a QuerySet to store the 'beginning_of_pass' and 'end_of_pass' timestamps.
    # Set the 'count' parameter to 2, as this set will contain 2 timestamps.
    query_set = device.create_query_set(type=wgpu.QueryType.timestamp, count=2)
    command_encoder = device.create_command_encoder()

    # Pass our QuerySet and the indices into it, where the timestamps will be written.
    pass_encoder = command_encoder.begin_compute_pass(
        timestamp_writes={
            "query_set": query_set,
            "beginning_of_pass_write_index": 0,
            "end_of_pass_write_index": 1,
        }
    )

    # Create the buffer to store our query results.
    # Each timestamp is 8 bytes. We mark the buffer usage to be QUERY_RESOLVE,
    # as we will use this buffer in a resolve_query_set call later.
    query_buf = device.create_buffer(
        size=8 * query_set.count,
        usage=wgpu.BufferUsage.QUERY_RESOLVE
        | wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST,
    )
    pass_encoder.set_pipeline(compute_pipeline)
    pass_encoder.set_bind_group(0, bind_group, [], None, None)
    pass_encoder.dispatch_workgroups(*global_size)  # x y z
    pass_encoder.end()

    # Resolve our queries, and store the results in the destination buffer we created above.
    command_encoder.resolve_query_set(
        query_set=query_set,
        first_query=0,
        query_count=2,
        destination=query_buf,
        destination_offset=0,
    )

    device.queue.submit([command_encoder.finish()])

    # Read the query buffer to get the timestamps.
    # Index 0: beginning timestamp
    # Index 1: end timestamp
    timestamps = device.queue.read_buffer(query_buf).cast("Q").tolist()
    print(
        f"Multiplying two {n} sized arrays took {(timestamps[1]-timestamps[0]) / 1000} us"
    )

    # Read result
    outview = device.queue.read_buffer(buffer3)
    result = torch.frombuffer(outview, dtype=torch.int32)

    # Calculate the result on the CPU for comparison
    result_cpu = torch.from_numpy(data1 + data2)

    # Ensure results are the same
    assert result.equal(result_cpu)
