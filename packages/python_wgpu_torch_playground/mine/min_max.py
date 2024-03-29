"""
A simple example to compute the minimum and maximum values of two vectors.
"""

__all__ = ["main"]


def main():
    import wgpu
    from pathlib import Path

    adapter = wgpu.gpu.request_adapter(power_preference="low-power")
    device = adapter.request_device()

    buffer = device.create_buffer(
        size=3 * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
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
                    "buffer": buffer,
                    "offset": 0,
                    "size": buffer.size,
                },
            },
        ],
    )

    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        ),
        compute={
            "module": device.create_shader_module(
                code=(Path(__file__).with_name("min_max.wgsl").open().read())
            ),
            "entry_point": "main",
        },
    )

    command_encoder = device.create_command_encoder()
    pass_encoder = command_encoder.begin_compute_pass()

    pass_encoder.set_pipeline(pipeline)
    pass_encoder.set_bind_group(0, bind_group, [], None, None)
    pass_encoder.dispatch_workgroups(1)  # x y z

    pass_encoder.end()
    device.queue.submit([command_encoder.finish()])

    result = device.queue.read_buffer(buffer).cast("i").tolist()
    expected = [4, 0, 4]
    assert result == expected, (result, expected)

    print(f"{__name__}: Ok")
