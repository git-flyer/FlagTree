def language_modify_all(all_array):
    try:
        import acl
        is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
    except Exception as e:
        is_compile_on_910_95 = False
