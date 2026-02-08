def language_extend_globals(globals_dict):
    try:
        import acl
        is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
    except Exception as e:
        is_compile_on_910_95 = False
    globals_dict["is_compile_on_910_95"] = is_compile_on_910_95


def language_extend_exports(globals_dict, all_list):
    from triton.language.core import make_tensor_descriptor, load_tensor_descriptor, store_tensor_descriptor, gather
    from triton.language.standard import topk
    globals_dict["make_tensor_descriptor"] = make_tensor_descriptor
    globals_dict["load_tensor_descriptor"] = load_tensor_descriptor
    globals_dict["store_tensor_descriptor"] = store_tensor_descriptor
    globals_dict["gather"] = gather
    globals_dict["topk"] = topk
    all_list.append("make_tensor_descriptor")
    all_list.append("load_tensor_descriptor")
    all_list.append("store_tensor_descriptor")
    all_list.append("gather")
    all_list.append("topk")
