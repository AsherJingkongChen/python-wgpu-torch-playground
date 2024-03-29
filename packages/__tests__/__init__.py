def test_module_resolving():
    import python_wgpu_torch_playground as plg

    assert plg.add_1d_array
    assert plg.add_1d_array.main
    assert plg.mine
    assert plg.mine.min_max

    from python_wgpu_torch_playground import add_1d_array

    assert add_1d_array.main

    import python_wgpu_torch_playground.mine.min_max as min_max

    assert min_max.main


def test_add_1d_array():
    from python_wgpu_torch_playground import add_1d_array

    add_1d_array.main()


def test_min_max():
    from python_wgpu_torch_playground.mine import min_max

    min_max.main()
