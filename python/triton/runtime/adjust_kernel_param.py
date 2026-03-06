import ast
import inspect
from typing import Dict, Set, Optional, Tuple, List
from triton import knobs


# ========
# Analyzer
# ========


class VariableCollector(ast.NodeVisitor):

    def __init__(self):
        self.variables: Set[str] = set[str]()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.variables.add(node.id)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        self.generic_visit(node)

    def visit_Call(self, node):
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node):
        # For tl.arange, skip module prefix 'tl'; for a.b collect 'a'
        if isinstance(node.value, ast.Name):
            # Skip module prefixes like tl, np, etc.
            if node.value.id not in ('tl', 'triton', 'np', 'torch'):
                self.variables.add(node.value.id)
        self.generic_visit(node)

    @staticmethod
    def collect(node) -> Set[str]:
        collector = VariableCollector()
        collector.visit(node)
        return collector.variables


class KernelDependencyAnalyzer(ast.NodeVisitor):

    def __init__(self):
        self.input_params: Set[str] = set[str]()  # input params
        self.constexpr_params: Set[str] = set[str]()  # constexpr params
        self.var_definitions: Dict[str, ast.AST] = {}  # var -> latest definition node
        # for input-constexpr dependencies analyze
        self.load_addresses: list = []  # tl.load address expressions
        # for make_tensor_descriptor dependencies analyze
        self.tma_args = {}  # tl.make_tensor_descriptor base node -> {strides: [...], block_shape: [...]}
        # for TMA descriptor load dependencies analyze
        self.tma_load_assignments = []  # desc.load {var_name, desc_name, addr_exprs}
        self.transpose_args_nodes = []  # tl.trans args
        # for tl.dot K-dim analyze
        self.dot_calls: list = []  # tl.dot call nodes
        # desc var -> {"shape": [...], "block_shape": [...]} from make_tensor_descriptor or hook
        self.tma_desc_defs: Dict[str, Dict[str, List[str]]] = {}
        # all historical definitions per var (in order); used for arange extraction
        # to avoid losing info when later assignments overwrite earlier ones
        self.var_all_definitions: Dict[str, List[ast.AST]] = {}

    # Collect function parameters and mark constexpr ones
    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            arg_name = arg.arg
            self.input_params.add(arg_name)

            if arg.annotation:
                ann_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else ''
                if not ann_str:
                    try:
                        ann_str = ast.dump(arg.annotation)
                    except Exception:
                        ann_str = ''  # Python 3.8 fallback
                if 'constexpr' in ann_str:
                    self.constexpr_params.add(arg_name)

        self.generic_visit(node)

    # Record variable definitions, capture desc.load and make_tensor_descriptor
    def visit_Assign(self, node):
        targets = node.targets
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            var_name = targets[0].id

            # Capture desc.load([addr, ...]) assignments
            if (isinstance(node.value, ast.Call) and self._is_tma_load(node.value) and node.value.args
                    and isinstance(node.value.args[0], ast.List)):
                desc_name = node.value.func.value.id
                addr_exprs = node.value.args[0].elts
                self.tma_load_assignments.append(
                    {'var_name': var_name, 'desc_name': desc_name, 'addr_exprs': addr_exprs})

            # TMA device: record LHS name + shape/block_shape from make_tensor_descriptor
            if (isinstance(node.value, ast.Call) and self._is_tl_make_tensor_descriptor(node.value)):
                shape_names = []
                block_names = []
                for kw in node.value.keywords:
                    if getattr(kw, 'arg', None) == 'shape' and isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Name):
                                shape_names.append(elt.id)
                    if getattr(kw, 'arg', None) == 'block_shape' and isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Name):
                                block_names.append(elt.id)
                if shape_names or block_names:
                    self.tma_desc_defs[var_name] = {"shape": shape_names, "block_shape": block_names}

            self.var_all_definitions.setdefault(var_name, []).append(node.value)
            self.var_definitions[var_name] = node.value
        self.generic_visit(node)

    # Treat x += expr as x = x <op> expr for dependency tracking
    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # Synthesize an equivalent BinOp node
            binop = ast.BinOp(
                left=ast.Name(id=var_name, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
            self.var_definitions[var_name] = binop
        self.generic_visit(node)

    # Record annotated assignments, mark constexpr if annotated
    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            self.var_definitions[var_name] = node.value

            if node.annotation:
                ann_str = ast.unparse(node.annotation) if hasattr(ast, 'unparse') else ''
                if not ann_str:
                    try:
                        ann_str = ast.dump(node.annotation)
                    except Exception:
                        ann_str = ''
                if 'constexpr' in ann_str:
                    self.constexpr_params.add(var_name)

        self.generic_visit(node)

    # Capture tl.load addresses, tl.trans args, tl.dot, and make_tensor_descriptor args
    def visit_Call(self, node):
        if self._is_tl_load(node) and node.args:
            self.load_addresses.append(node.args[0])
        elif self._is_tl_transpose(node) and node.args:
            self.transpose_args_nodes.append(node.args[0])
        elif self._is_tl_dot(node):
            self.dot_calls.append(node)
        elif self._is_tl_make_tensor_descriptor(node):
            base = None
            # Collect the base node
            for kw in node.keywords:
                if hasattr(kw, 'arg') and kw.arg == 'base':
                    if kw.value not in self.tma_args:
                        base = kw.value
                        self.tma_args[base] = {'strides': [], 'block_shape': []}
            # Collect strides and block_shape element nodes
            for kw in node.keywords:
                if hasattr(kw, 'arg') and (kw.arg in ['strides', 'block_shape']):
                    if hasattr(kw, 'value') and isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            self.tma_args[base][kw.arg].append(elt)
        self.generic_visit(node)

    # Return True if node is tl.load(...)
    def _is_tl_load(self, node) -> bool:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    # Return True if node is desc.load(...)
    def _is_tma_load(self, node) -> bool:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id not in ('tl', 'triton')
        return False

    # Return True if node is tl.make_tensor_descriptor(...)
    def _is_tl_make_tensor_descriptor(self, node) -> bool:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'make_tensor_descriptor':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    # Return True if node is tl.trans(...)
    def _is_tl_transpose(self, node) -> bool:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'trans':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    # Resolve a symbol (e.g. a local TMA desc var) to its underlying tensor input param
    def _resolve_tensor_param(self, symbol: Optional[str]) -> Optional[str]:
        if not symbol:
            return None

        if symbol in self.input_params:
            return symbol

        # Local var from make_tensor_descriptor: use 'base' kwarg directly
        if symbol in self.var_definitions:
            node = self.var_definitions[symbol]
            if isinstance(node, ast.Call) and self._is_tl_make_tensor_descriptor(node):
                for kw in node.keywords:
                    if getattr(kw, "arg", None) == "base" and isinstance(kw.value, ast.Name):
                        base_name = kw.value.id
                        if base_name in self.input_params:
                            return base_name

        # Fallback: find the unique input param via dependency analysis
        input_deps, _ = self.get_dependencies(symbol)
        if len(input_deps) == 1:
            return list(input_deps)[0]

        return None

    def _is_tl_dot(self, node) -> bool:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'dot':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _is_tl_arange(self, node) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'arange':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _extract_arange_bs(self, node: ast.AST) -> Set[str]:
        out: Set[str] = set[str]()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and self._is_tl_arange(child) and len(child.args) >= 2:
                if isinstance(child.args[1], ast.Name):
                    out.add(child.args[1].id)
        return out

    def _extract_arange_bs_recursive(self, var_name: str, visited: Optional[Set[str]] = None) -> Set[str]:
        if visited is None:
            visited = set[str]()
        if var_name in visited:
            return set[str]()
        visited.add(var_name)

        ret = set[str]()
        for def_node in self.var_all_definitions.get(var_name, []):
            ret.update(self._extract_arange_bs(def_node))
            for child_var in VariableCollector.collect(def_node):
                if child_var != var_name and child_var not in self.input_params and child_var not in self.constexpr_params:
                    ret.update(self._extract_arange_bs_recursive(child_var, visited.copy()))
        return ret

    def get_dependencies(self, var_name: str, visited: Optional[Set[str]] = None) -> tuple[Set[str], Set[str]]:
        if visited is None:
            visited = set[str]()
        if var_name in visited:
            return set[str](), set[str]()
        visited.add(var_name)

        input_deps = set[str]()
        constexpr_deps = set[str]()

        # Check if it is a non-constexpr parameter
        if var_name in self.input_params and var_name not in self.constexpr_params:
            input_deps.add(var_name)
            return input_deps, constexpr_deps

        # Check if it is a constexpr parameter
        if var_name in self.constexpr_params:
            constexpr_deps.add(var_name)
            return input_deps, constexpr_deps

        # Recursively analyze the dependencies of the variable definition
        if var_name in self.var_definitions and not var_name.startswith('pid'):
            definition_node = self.var_definitions[var_name]
            # Skip runtime value program_id
            if True:
                used_vars = VariableCollector.collect(definition_node)
                for used_var in used_vars:
                    sub_inputs, sub_constexprs = self.get_dependencies(used_var, visited.copy())
                    input_deps.update(sub_inputs)
                    constexpr_deps.update(sub_constexprs)
        return input_deps, constexpr_deps

    def _get_dependencies_vars(self, var_name: str, visited: Optional[Set[str]] = None) -> Set[str]:
        if visited is None:
            visited = set[str]()
        if var_name in visited:
            return set[str]()
        visited.add(var_name)

        var_deps = set()

        # Check if it is an input or constexpr parameter
        if (var_name in self.input_params) or (var_name in self.constexpr_params):
            return var_deps

        # Recursively analyze the dependencies of the variable definition
        if var_name in self.var_definitions and not var_name.startswith('pid'):
            definition_node = self.var_definitions[var_name]
            # Skip runtime value program_id
            if True:
                used_vars = VariableCollector.collect(definition_node)
                for used_var in used_vars:
                    var_deps.update(self._get_dependencies_vars(used_var, visited.copy()))
        return var_deps

    def analyze_dot_dim(self, desc_block_shapes: Dict[str, List[str]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        # raw_var_block_shapes maps each desc.load target var to (desc_name, block_names)
        raw_var_block_shapes: Dict[str, tuple] = {}
        for tma_info in self.tma_load_assignments:
            target_var = tma_info['var_name']
            desc_name = tma_info['desc_name']
            desc_block_shape = list[str](desc_block_shapes.get(desc_name) or [])
            raw_var_block_shapes[target_var] = (desc_name, desc_block_shape)

        # var_block_shapes: swap last two dims of src's block shape if tl.trans(src)
        var_block_shapes = dict[str, tuple](raw_var_block_shapes)
        for var_name, def_node in self.var_definitions.items():
            if isinstance(def_node, ast.Call) and self._is_tl_transpose(def_node) and def_node.args:
                src_vars = VariableCollector.collect(def_node.args[0])
                for src_var in src_vars:
                    if src_var in raw_var_block_shapes:
                        desc_name, src_block_shapes = raw_var_block_shapes[src_var]
                        transposed = list(src_block_shapes)
                        if len(transposed) >= 2:
                            transposed[-1], transposed[-2] = transposed[-2], transposed[-1]
                        var_block_shapes[var_name] = (desc_name, transposed)
                        break

        # tl.dot(a, b): a (M, K), b (K, N).
        bs_m_map: Dict[str, Set[str]] = {}
        bs_k_map: Dict[str, Set[str]] = {}
        has_inconsistent = False
        for dot_node in self.dot_calls:
            args = dot_node.args
            if len(args) < 2:
                continue

            k_from_a = None
            k_from_b = None
            a_desc_name = None
            b_desc_name = None

            # a: shape (M, K)
            for a_var in VariableCollector.collect(args[0]):
                if a_var in var_block_shapes:
                    a_desc_name, a_block_shapes = var_block_shapes[a_var]
                    if a_block_shapes:
                        m_from_a = a_block_shapes[0]
                        a_tensor_param_for_m = self._resolve_tensor_param(a_desc_name)
                        if m_from_a is not None and a_tensor_param_for_m is not None:
                            if m_from_a not in bs_m_map:
                                bs_m_map[m_from_a] = set[str]()
                            bs_m_map[m_from_a].add(a_tensor_param_for_m)
                        k_from_a = a_block_shapes[-1]
                    break

            # b: shape (K, N)
            for b_var in VariableCollector.collect(args[1]):
                if b_var in var_block_shapes:
                    b_desc_name, b_block_shapes = var_block_shapes[b_var]
                    if b_block_shapes:
                        k_from_b = b_block_shapes[0]
                    break

            # check K dim is consistent
            bs_k_name = k_from_a if k_from_a is not None else k_from_b
            if (k_from_a is not None and k_from_b is not None and k_from_a != k_from_b):
                has_inconsistent = True

            # Resolve desc names to real tensor param names
            a_tensor_param = self._resolve_tensor_param(a_desc_name)
            b_tensor_param = self._resolve_tensor_param(b_desc_name)

            if bs_k_name is not None:
                if a_tensor_param is not None:
                    if bs_k_name not in bs_k_map:
                        bs_k_map[bs_k_name] = set[str]()
                    bs_k_map[bs_k_name].add(a_tensor_param)
                if b_tensor_param is not None:
                    if bs_k_name not in bs_k_map:
                        bs_k_map[bs_k_name] = set[str]()
                    bs_k_map[bs_k_name].add(b_tensor_param)

        if has_inconsistent:
            return {}, {}
        return bs_m_map, bs_k_map

    def analyze_tl_load_bs(self) -> Dict[str, str]:
        load_map: Dict[str, str] = {}
        for addr_expr in self.load_addresses:
            used_vars = VariableCollector.collect(addr_expr)
            for var_name in used_vars:
                if var_name not in self.var_all_definitions or var_name.startswith("pid"):
                    continue
                blocks = self._extract_arange_bs_recursive(var_name)
                input_deps, _ = self.get_dependencies(var_name)
                if len(input_deps) == 1 and len(blocks) == 1:
                    ts_name = list[str](input_deps)[0]
                    bs_name = list[str](blocks)[0]
                    load_map[bs_name] = ts_name
        return load_map

    # Parse nargs['desc'].block_shape = [B1, B2, ...] assignments in a pre_hook
    def _parse_hook_block_shape(self, hook_ast: ast.FunctionDef) -> Dict[str, List[str]]:
        hook_desc_block_shapes_map: Dict[str, List[str]] = {}
        for node in ast.walk(hook_ast):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            t = node.targets[0]
            if not isinstance(t, ast.Attribute) or t.attr != "block_shape":
                continue
            # t.value should be nargs["a_desc"] or similar
            if not isinstance(t.value, ast.Subscript):
                continue
            sub = t.value
            if not isinstance(sub.value, ast.Name):
                continue
            desc_name = None
            if isinstance(sub.slice, ast.Constant):
                desc_name = sub.slice.value
            elif getattr(sub.slice, "value", None) is not None and isinstance(sub.slice.value, ast.Constant):
                desc_name = sub.slice.value.value
            try:
                if desc_name is None:
                    desc_name = ast.literal_eval(sub.slice)
            except Exception:
                pass
            if not isinstance(desc_name, str):
                continue
            if not isinstance(node.value, ast.List):
                continue
            desc_block_shapes = []
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    desc_block_shapes.append(elt.id)
            if desc_block_shapes:
                hook_desc_block_shapes_map[desc_name] = desc_block_shapes
        return hook_desc_block_shapes_map

    def analyze_desc_load_bs(
            self, pre_hook_fn: Optional[object] = None) -> Tuple[Dict[str, Set[Tuple[str, ...]]], Dict[str, List[str]]]:
        # 1) Build desc -> block_shape from make_tensor_descriptor definitions
        desc_block_shapes: Dict[str, List[str]] = {}
        for desc_name, defn in self.tma_desc_defs.items():
            blist = defn.get("block_shape") or []
            if blist:
                desc_block_shapes[desc_name] = list[str](blist)

        # 2) Merge block_shape assignments from pre_hook (for TMA host descriptors)
        if pre_hook_fn is not None and hasattr(pre_hook_fn, "__code__"):
            try:
                try:
                    hook_src = inspect.getsource(pre_hook_fn)
                except Exception:
                    hook_src = None
                if hook_src:
                    hook_ast = ast.parse(hook_src)
                    for n in ast.walk(hook_ast):
                        if not isinstance(n, ast.FunctionDef):
                            continue
                        desc_block_shapes.update(self._parse_hook_block_shape(n))
                        break
            except Exception:
                pass

        # 3) Build tma_map
        transpose_used_vars = set()
        for arg_node in self.transpose_args_nodes:
            for v in VariableCollector.collect(arg_node):
                transpose_used_vars.add(v)
                transpose_used_vars.update(self._get_dependencies_vars(v))

        # Only add the canonical (non-transposed) block shape for each descriptor.
        tma_map: Dict[str, Set[Tuple[str, ...]]] = {}
        for tma_info in self.tma_load_assignments:
            desc_name = tma_info["desc_name"]
            target_var = tma_info["var_name"]
            if target_var in transpose_used_vars:
                continue
            block_shape = list[str](desc_block_shapes.get(desc_name) or [])
            if block_shape:
                tma_map.setdefault(desc_name, set()).add(tuple(block_shape))
        return tma_map, desc_block_shapes


_analysis_cache: Dict[int, Tuple] = {}


# Analyzer
def analyze_kernel_dependencies(jit_fn, pre_hook_fn: Optional[object] = None) -> Tuple:
    cache_key = (id(jit_fn), id(pre_hook_fn) if pre_hook_fn is not None else None)
    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]

    try:
        fn_ast = jit_fn.parse()
        analyzer = KernelDependencyAnalyzer()
        analyzer.visit(fn_ast)

        # Analyzer 1: tl.load - tl.arange
        load_map = analyzer.analyze_tl_load_bs()
        # Analyzer 2: desc.load - desc.block_shape
        tma_map, desc_block_shapes = analyzer.analyze_desc_load_bs(pre_hook_fn)
        # Analyzer 3: tl.dot M/N/K
        bs_m_map, bs_k_map = analyzer.analyze_dot_dim(desc_block_shapes)
        # cache
        _analysis_cache[cache_key] = (load_map, tma_map, bs_m_map, bs_k_map)

        if knobs.autotuning.print:
            if load_map:
                print(
                    f"\n=== FlagTree adjust_kernel_param tl.load (by tl.arange): {getattr(jit_fn, '__name__', 'unknown')} ==="
                )
                for bs_name, ts_name in load_map.items():
                    print(f"  load_map[bs_name, ts_name]: '{bs_name}' -> '{ts_name}'")
            if tma_map:
                print(
                    f"\n=== FlagTree adjust_kernel_param desc.load (by block_shape): {getattr(jit_fn, '__name__', 'unknown')} ==="
                )
                for desc_name, bs_names_set in tma_map.items():
                    print(f"  tma_map[desc_name, set[block_shapes]]: '{desc_name}' -> {bs_names_set}")
            if bs_m_map or bs_k_map:
                print(f"\n=== FlagTree adjust_kernel_param tl.dot: {getattr(jit_fn, '__name__', 'unknown')} ===")
                print(f"  bs_m_map[bs_name, set[param_name]]: {bs_m_map}")
                print(f"  bs_k_map[bs_name, set[param_name]]: {bs_k_map}")
            print("==============================================================\n")

        return (load_map, tma_map, bs_m_map, bs_k_map)

    except Exception as e:
        print(f"Warning: adjust_kernel_param failed: {e}")
        return (None, None, None, None)


def clear_analysis_cache():
    _analysis_cache.clear()


# ========
# Adjuster
# ========


def update_bs(nargs, current, config, bs_name, bs, title, reason):
    if knobs.autotuning.print:
        print(f'[AABS] {title}: adjust {bs_name} {current[bs_name]} => {bs} because {bs_name} {reason}')
    current[bs_name] = bs
    config.kwargs[bs_name] = bs


def adjust_block_size_tl_load(nargs, current, config, bs_name, ts_name):
    if bs_name not in current or ts_name not in nargs:
        return
    bs = current[bs_name]
    ts = nargs[ts_name]
    if not isinstance(bs, int) or not isinstance(ts, int):
        return
    if bs > ts:  # block_size > tensor_size
        from triton import next_power_of_2
        update_bs(nargs, current, config, bs_name, next_power_of_2(ts), "tl.load", f"> {ts}")


def adjust_block_size_tma(nargs, current, config, desc_name, bs_names):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor
    from triton import next_power_of_2
    if desc_name not in nargs:
        return
    if not isinstance(nargs[desc_name], TensorDescriptor):
        return
    desc_base: torch.Tensor = nargs[desc_name].base
    if not isinstance(desc_base, torch.Tensor):
        return
    if len(desc_base.shape) != len(bs_names):
        if knobs.autotuning.print:
            print(
                f"[AABS] Warning: len(desc_base.shape)={len(desc_base.shape)} != {len(bs_names)}=len(bs_names), bs_names={bs_names}"
            )
        return
    for shape_size, bs_name in zip(desc_base.shape, bs_names):
        bs = current[bs_name]
        if not isinstance(shape_size, int) or not isinstance(bs, int):
            continue
        if bs > shape_size:
            update_bs(nargs, current, config, bs_name, next_power_of_2(shape_size), "TMA", f"> {shape_size}")


def adjust_block_size_dot_k_dim(nargs, current, config, bs_k_map, limit):
    for bs_name in bs_k_map.keys():
        if bs_name not in current:
            continue
        bs = current[bs_name]
        if not isinstance(bs, int):
            continue
        if bs < limit:
            update_bs(nargs, current, config, bs_name, limit, "tl.dot", f"< {limit}=limit_k")


def adjust_block_size_dot_m_dim(nargs, current, config, bs_k_map, bs_m_map, limit_bytes):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    bs_k = 1
    for bs_k_name in bs_k_map.keys():
        if bs_k_name in current and isinstance(current[bs_k_name], int):
            bs_k = current[bs_k_name]
            break

    for bs_name, param_names in bs_m_map.items():
        if bs_name not in current:
            continue
        bs = current[bs_name]
        if not isinstance(bs, int):
            continue
        elem_type_size = None
        for pname in param_names:
            if pname not in nargs:
                continue
            narg = nargs[pname]
            if isinstance(narg, TensorDescriptor):
                narg = narg.base
            if isinstance(narg, torch.Tensor):
                elem_type_size = narg.element_size()
                break
        if elem_type_size is None:
            continue
        # SWIZZLE_NONE: bs_k * elem_type_size = 16B, limit = 8
        # SWIZZLE_16B:  bs_k * elem_type_size = 16B, limit = 8
        # SWIZZLE_32B:  bs_k * elem_type_size = 32B, limit = 4
        # SWIZZLE_64B:  bs_k * elem_type_size = 64B, limit = 2
        # SWIZZLE_128B: bs_k * elem_type_size = 128B, limit = 1
        limit = max(int(limit_bytes / bs_k / elem_type_size), 1)
        if bs < limit:
            update_bs(nargs, current, config, bs_name, limit, "tl.dot", f"< {limit}=limit_m")


# AABS
def auto_adjust_block_sizes(nargs, fn, configs, current, config):
    pre_hook_fn = getattr(config, "pre_hook", None) or (configs[0].pre_hook if configs else None)
    load_map, tma_map, bs_m_map, bs_k_map = analyze_kernel_dependencies(fn, pre_hook_fn=pre_hook_fn)

    if load_map:
        if knobs.autotuning.print:
            print("[AABS] 1. adjust bs in tl_load")
        for bs_name, ts_name in load_map.items():
            adjust_block_size_tl_load(nargs, current, config, bs_name, ts_name)

    if tma_map:
        if knobs.autotuning.print:
            print("[AABS] 2. adjust bs in tma")
        for desc_name, bs_names_set in tma_map.items():
            for bs_names in bs_names_set:
                adjust_block_size_tma(nargs, current, config, desc_name, bs_names)
        adjust_block_size_dot_k_dim(nargs, current, config, bs_k_map, 16)
        adjust_block_size_dot_m_dim(nargs, current, config, bs_k_map, bs_m_map, 128)

    if knobs.autotuning.print:
        nargs_str = ''
        if nargs:
            import torch
            from triton.tools.tensor_descriptor import TensorDescriptor
            nargs_parts = []
            for k, v in nargs.items():
                if isinstance(v, (torch.Tensor, TensorDescriptor)):
                    nargs_parts.append(k)
                else:
                    nargs_parts.append(f'{k}={v}')
            nargs_str = ', '.join(nargs_parts)
        base_fn = fn
        while not inspect.isfunction(base_fn):
            base_fn = base_fn.fn
        print(f'[AABS] ==== Finish: {base_fn.__name__}({nargs_str})')
        print(f'[AABS] ====         adjusted_config={config}')
