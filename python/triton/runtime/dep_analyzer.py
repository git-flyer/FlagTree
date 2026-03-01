import ast
from typing import Dict, Set, Optional, Tuple, List
from functools import lru_cache
from triton import knobs


class VariableCollector(ast.NodeVisitor):

    def __init__(self):
        self.variables: Set[str] = set()

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
        # 对于 tl.arange 这样的形式，不收集 tl
        # 对于 a.b 形式，收集 a
        if isinstance(node.value, ast.Name):
            # 跳过模块前缀如 tl, np 等
            if node.value.id not in ('tl', 'triton', 'np', 'torch'):
                self.variables.add(node.value.id)
        self.generic_visit(node)

    @staticmethod
    def collect(node) -> Set[str]:
        """收集 AST 节点中的所有变量"""
        collector = VariableCollector()
        collector.visit(node)
        return collector.variables


class KernelDependencyAnalyzer(ast.NodeVisitor):

    def __init__(self):
        self.input_params: Set[str] = set()      # 输入参数名集合
        self.constexpr_params: Set[str] = set()  # constexpr 参数名集合
        self.var_definitions: Dict[str, ast.AST] = {}  # 变量名 -> AST 定义节点
        # for input-constexpr dependencies analyze
        self.load_addresses: list = []  # 存储 tl.load 的地址表达式
        # for make_tensor_descriptor dependencies analyze
        self.tma_args = {} # 存储 tl.make_tensor_descriptor 的 base 及其对应的 stride 和 block shape
        # for TMA descriptor load dependencies analyze
        self.tma_load_assignments = []  # 存储 desc.load 赋值的目标变量和相关信息
        self.transpose_args_nodes = []  # 存储 tl.trans 参数
        # for tl.dot K-dim analyze
        self.dot_calls: list = []  # 存储 tl.dot 调用节点

    def visit_FunctionDef(self, node):
        """分析函数定义，收集参数信息"""
        # 收集所有参数
        for arg in node.args.args:
            arg_name = arg.arg
            self.input_params.add(arg_name)

            # 检查是否是 constexpr
            if arg.annotation:
                ann_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else ''
                if not ann_str:
                    # Python 3.8 fallback
                    try:
                        ann_str = ast.dump(arg.annotation)
                    except:
                        ann_str = ''
                if 'constexpr' in ann_str:
                    self.constexpr_params.add(arg_name)

        # 继续分析函数体
        self.generic_visit(node)

    def visit_Assign(self, node):
        """分析赋值语句，记录变量定义"""
        targets = node.targets
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            var_name = targets[0].id

            # 检查右侧是否是TMA load调用
            if (isinstance(node.value, ast.Call) and
                self._is_tma_load(node.value) and
                node.value.args and
                isinstance(node.value.args[0], ast.List)):
                # 记录TMA load的赋值目标和相关信息
                tma_desc_name = node.value.func.value.id
                addr_exprs = node.value.args[0].elts
                self.tma_load_assignments.append({
                    'var_name': var_name,
                    'tma_desc_name': tma_desc_name,
                    'addr_exprs': addr_exprs
                })

            self.var_definitions[var_name] = node.value
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        """分析类似 x += expr 的增量赋值，把它视作新的定义以便依赖分析"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # 构造一个等价的 BinOp 表达式：x_new = x_old <op> value
            binop = ast.BinOp(
                left=ast.Name(id=var_name, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
            self.var_definitions[var_name] = binop
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """分析带注解的赋值语句"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            self.var_definitions[var_name] = node.value

            # 检查是否是 constexpr
            if node.annotation:
                ann_str = ast.unparse(node.annotation) if hasattr(ast, 'unparse') else ''
                if not ann_str:
                    try:
                        ann_str = ast.dump(node.annotation)
                    except:
                        ann_str = ''
                if 'constexpr' in ann_str:
                    self.constexpr_params.add(var_name)

        self.generic_visit(node)

    def visit_Call(self, node):
        """分析函数调用，捕获 tl.load"""
        # 检查是否是 tl.load 调用
        if self._is_tl_load(node) and node.args:
            self.load_addresses.append(node.args[0])
        elif self._is_tl_transpose(node) and node.args:
            # 获取 transpose 的参数
            self.transpose_args_nodes.append(node.args[0])
        elif self._is_tl_dot(node):
            self.dot_calls.append(node)
        elif self._is_tl_make_tensor_descriptor(node):
            base = None
            # 收集 make_tensor_descriptor 中的 base 的节点
            for kw in node.keywords:
                if hasattr(kw, 'arg') and kw.arg == 'base':
                    if kw.value not in self.tma_args:
                        base = kw.value
                        self.tma_args[base] = {'strides': [], 'block_shape': []}
            # 收集 make_tensor_descriptor 中的 stride 和 block_shape 元素的节点
            for kw in node.keywords:
                if hasattr(kw, 'arg') and (kw.arg in ['strides', 'block_shape']):
                    if hasattr(kw, 'value') and isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            self.tma_args[base][kw.arg].append(elt)
        self.generic_visit(node)

    def _is_tl_load(self, node) -> bool:
        """检查是否是 tl.load 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _is_tma_load(self, node) -> bool:
        """检查是否是 desc.load 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id not in ('tl', 'triton')
        return False

    def _is_tl_make_tensor_descriptor(self, node) -> bool:
        """检查是否是 tl.make_tensor_descriptor 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'make_tensor_descriptor':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _is_tl_transpose(self, node) -> bool:
        """检查是否是 tl.trans 或 triton.trans 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'trans':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _resolve_tensor_param(self, symbol: Optional[str]) -> Optional[str]:
        """
        给定符号名（例如 TMA 描述符变量名），尝试解析其背后对应的
        真实 tensor 形参名。

        - 对于 host kernel：描述符本身就是形参，例如 a_desc
        - 对于 device kernel：描述符通常由 tl.make_tensor_descriptor(base=...) 定义，
          这里通过依赖分析反推唯一的 input param（如 A）。
        """
        if not symbol:
            return None

        # 如果本身就是输入参数，直接返回
        if symbol in self.input_params:
            return symbol

        # 如果是通过 tl.make_tensor_descriptor 定义的局部变量，优先从 base 关键字里找
        if symbol in self.var_definitions:
            node = self.var_definitions[symbol]
            if isinstance(node, ast.Call) and self._is_tl_make_tensor_descriptor(node):
                for kw in node.keywords:
                    if getattr(kw, "arg", None) == "base" and isinstance(kw.value, ast.Name):
                        base_name = kw.value.id
                        # base 通常就是真实的 tensor 形参（如 A/B/C）
                        if base_name in self.input_params:
                            return base_name

        # 否则通过依赖分析找唯一的 input param
        input_deps, _ = self.get_dependencies(symbol)
        if len(input_deps) == 1:
            return list(input_deps)[0]

        return None

    def _is_tl_dot(self, node) -> bool:
        """检查是否是 tl.dot 或 triton.dot 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'dot':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _is_tl_program_id(self, node) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # program_id
        if isinstance(func, ast.Name):
            return func.id == 'program_id'
        # tl.program_id, language.program_id, triton.program_id
        if isinstance(func, ast.Attribute) and func.attr == 'program_id':
            value = func.value
            if isinstance(value, ast.Name):
                return value.id in ('tl', 'triton', 'language')
            # triton.language.program_id
            if isinstance(value, ast.Attribute):
                return (value.attr == 'language' and
                        isinstance(value.value, ast.Name) and
                        value.value.id == 'triton')
        return False

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
        if var_name in self.var_definitions:
            definition_node = self.var_definitions[var_name]
            # Skip runtime value program_id
            if not self._is_tl_program_id(definition_node):
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
            return set()
        visited.add(var_name)

        var_deps = set()

        # Check if it is an input or constexpr parameter
        if (var_name in self.input_params) or (var_name in self.constexpr_params):
            return var_deps

        # Recursively analyze the dependencies of the variable definition
        if var_name in self.var_definitions:
            definition_node = self.var_definitions[var_name]
            # Skip runtime value program_id
            if not self._is_tl_program_id(definition_node):
                used_vars = VariableCollector.collect(definition_node)
                for used_var in used_vars:
                    var_deps.update(self._get_dependencies_vars(used_var, visited.copy()))
        return var_deps


    def analyze_tma_with_trans_check(self):
        tma_map = {}

        # 收集 transpose 参数直接使用的变量名
        transpose_used_vars = set()
        for arg_node in self.transpose_args_nodes:
            used_vars = VariableCollector.collect(arg_node)
            # 收集地址表达式中使用的变量
            for var_name in used_vars:
                transpose_used_vars.add(var_name)
                transpose_used_vars.update(self._get_dependencies_vars(var_name))

        # 检查每个TMA load后面是否紧跟trans调用
        for tma_info in self.tma_load_assignments:
            target_var = tma_info['var_name']
            tma_desc_name = tma_info['tma_desc_name']
            addr_exprs = tma_info['addr_exprs']

            # 分析TMA load地址表达式中的block names
            bs_names = []
            for addr_expr in addr_exprs:
                used_vars = VariableCollector.collect(addr_expr)
                for var_name in used_vars:
                    _, constexpr_deps = self.get_dependencies(var_name)
                    if len(constexpr_deps) == 1:
                        bs_names.append(list(constexpr_deps)[0])
                        break

            if target_var in transpose_used_vars:
                bs_names[-1], bs_names[-2] = bs_names[-2], bs_names[-1]

            # 添加每个 TMA Descriptor 对应的 block names（可能有多对）
            if tma_desc_name in tma_map.keys():
                tma_map[tma_desc_name].add(tuple(bs_names))
            else:
                tma_map[tma_desc_name] = set()
                tma_map[tma_desc_name].add(tuple(bs_names))

        return tma_map

    def analyze_dot_dim(self) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        分析 tl.dot(a, b) 调用中 M / K 维度对应的 BLOCK constexpr 变量。

        策略：
        - 每个 TMA load 赋值 (var = desc.load([addr0, addr1, ...])) 都能推断出
          该 tile 各维度对应的 BLOCK 变量（通过地址表达式依赖分析）。
        - 若该变量后续被 tl.trans 使用，则其维度顺序对应翻转。
        - 若某变量由 tl.trans(src) 赋值，则其 block shape 为 src 翻转后的结果。
        - 在 tl.dot(a, b) 中：a 的最后一维 = K，b 的第一维 = K。
        - 两者应当一致，一致时返回 k_block 字段。

        Returns:
            (block_m_map, block_k_map):
              block_m_map: {BLOCK_M 名 -> 使用该 M 维的 tensor / descriptor 参数名集合}
              block_k_map: {BLOCK_K 名 -> 使用该 K 维的 tensor / descriptor 参数名集合（通过一致性校验）}
        """
        # Step 1: 为每个 TMA load 目标变量推断原始 block shape（不含 trans 翻转效果）
        raw_var_block_shape: Dict[str, tuple] = {}
        for tma_info in self.tma_load_assignments:
            target_var = tma_info['var_name']
            tma_desc_name = tma_info['tma_desc_name']
            addr_exprs = tma_info['addr_exprs']

            bs_names = []
            for addr_expr in addr_exprs:
                used_vars = VariableCollector.collect(addr_expr)
                matched = None
                for var_name in used_vars:
                    _, constexpr_deps = self.get_dependencies(var_name)
                    if len(constexpr_deps) == 1:
                        matched = list(constexpr_deps)[0]
                        break
                bs_names.append(matched)

            raw_var_block_shape[target_var] = (tma_desc_name, bs_names)

        # Step 2: 处理 tl.trans(src) 赋值，推断 trans 后变量的 block shape
        # e.g. a = tl.trans(a_t) → a 的 block shape 为 a_t 的最后两维交换
        var_block_shape: Dict[str, tuple] = dict(raw_var_block_shape)
        for var_name, def_node in self.var_definitions.items():
            if (isinstance(def_node, ast.Call) and
                    self._is_tl_transpose(def_node) and
                    def_node.args):
                src_vars = VariableCollector.collect(def_node.args[0])
                for src_var in src_vars:
                    if src_var in raw_var_block_shape:
                        tma_desc_name, src_block_names = raw_var_block_shape[src_var]
                        transposed = list(src_block_names)
                        if len(transposed) >= 2:
                            transposed[-1], transposed[-2] = transposed[-2], transposed[-1]
                        var_block_shape[var_name] = (tma_desc_name, transposed)
                        break

        # Step 3: 对每个 tl.dot 调用分析 K 维度
        # 判断依据：tl.dot(a, b) 中 a 的形状为 (M, K)，b 的形状为 (K, N)。
        # K 维度由操作数顺序决定：a 的最后一维必然是 K 维，无需拿 b 的首维做等值比对
        # （等值比对在 M=N 等特殊情况下会误判）。
        block_m_map: Dict[str, Set[str]] = {}
        block_k_map: Dict[str, Set[str]] = {}
        has_inconsistent = False
        for dot_node in self.dot_calls:
            args = dot_node.args
            if len(args) < 2:
                continue

            k_from_a = None
            k_from_b = None
            a_desc_name = None
            b_desc_name = None
            a_block_names = None
            b_block_names = None

            # 第一个操作数 a：形状为 (M, K)，M = 第一维，K = 最后一维
            for a_var in VariableCollector.collect(args[0]):
                if a_var in var_block_shape:
                    a_desc_name, a_block_list = var_block_shape[a_var]
                    a_block_names = tuple(a_block_list)
                    if a_block_list:
                        # M 维 block 名：a 的第一维
                        m_from_a = a_block_list[0]
                        # 映射到真实 tensor 形参名（host: a_desc; device: A）
                        a_tensor_param_for_m = self._resolve_tensor_param(a_desc_name)
                        if m_from_a is not None and a_tensor_param_for_m is not None:
                            if m_from_a not in block_m_map:
                                block_m_map[m_from_a] = set()
                            block_m_map[m_from_a].add(a_tensor_param_for_m)
                        # K 维 block 名：a 的最后一维
                        k_from_a = a_block_list[-1]
                    break

            # 第二个操作数 b：形状为 (K, N)，K = 第一维（仅作辅助信息，不参与主判断）
            for b_var in VariableCollector.collect(args[1]):
                if b_var in var_block_shape:
                    b_desc_name, b_block_list = var_block_shape[b_var]
                    b_block_names = tuple(b_block_list)
                    if b_block_list:
                        k_from_b = b_block_list[0]
                    break

            # K 维度由操作数顺序直接确定：a 的最后一维就是 K，
            # 但仍然需要和 b 的第一维做一致性校验；若二者都可解析且不相等，
            # 说明当前 kernel 的访问模式与矩阵乘约定不一致，此时代码不返回分析结果。
            k_block = k_from_a if k_from_a is not None else k_from_b

            if (k_from_a is not None
                    and k_from_b is not None
                    and k_from_a != k_from_b):
                has_inconsistent = True

            # 将参与 tl.dot 的“描述符名”解析为真实的 tensor 形参名
            a_tensor_param = self._resolve_tensor_param(a_desc_name)
            b_tensor_param = self._resolve_tensor_param(b_desc_name)

            if k_block is not None:
                # K 维同时与 a / b 对应的 tensor 参数相关（如果存在）
                if a_tensor_param is not None:
                    if k_block not in block_k_map:
                        block_k_map[k_block] = set[str]()
                    block_k_map[k_block].add(a_tensor_param)
                if b_tensor_param is not None:
                    if k_block not in block_k_map:
                        block_k_map[k_block] = set[str]()
                    block_k_map[k_block].add(b_tensor_param)

        # 若任意一次 tl.dot 的 K 维推断结果在 a / b 两侧不一致，
        # 则认为整体分析不可靠，返回空集合让上层逻辑放弃自动调整。
        if has_inconsistent:
            return {}, {}

        return block_m_map, block_k_map


    def analyze_load(self) -> Dict[str, str]:
        load_map: Dict[str, str] = {}
        for addr_expr in self.load_addresses:
            # Collect variables used in the address expression
            used_vars = VariableCollector.collect(addr_expr)
            # Analyze dependencies of each variable
            for var_name in used_vars:
                input_deps, constexpr_deps = self.get_dependencies(var_name)
                # If depends on both 1 input param and 1 constexpr param
                if len(input_deps) == 1 and len(constexpr_deps) == 1:
                    bs_name = list(constexpr_deps)[0]
                    ts_name = list(input_deps)[0]
                    load_map[bs_name] = ts_name
        return load_map


_analysis_cache: Dict[int, Tuple] = {}


def analyze_kernel_dependencies(jit_fn) -> Tuple:
    # Check cache
    fn_id = id(jit_fn)
    if fn_id in _analysis_cache:
        return _analysis_cache[fn_id]

    try:
        # Create analyzer and visit ast
        fn_ast = jit_fn.parse()
        analyzer = KernelDependencyAnalyzer()
        analyzer.visit(fn_ast)
        # load_map: Dict[bs_name, ts_name]
        load_map = analyzer.analyze_load()
        # tma_map: Dict[desc_name, Set[bs_names: Tuple[bs, bs]]]
        tma_map = analyzer.analyze_tma_with_trans_check()
        # bs_m_map, bs_k_map: Dict[bs_name, Set[param_name]]
        bs_m_map, bs_k_map = analyzer.analyze_dot_dim()

        # Cache analysis results
        _analysis_cache[fn_id] = (load_map, tma_map, bs_m_map, bs_k_map)

        # Print analysis results
        if knobs.autotuning.print:
            if load_map:
                print(f"\n=== FlagTree dep_analyzer tl.load or tma device: {getattr(jit_fn, '__name__', 'unknown')} ===")
                for bs_name, ts_name in load_map.items():
                    print(f"  TensorSize '{bs_name}' is related to BlockSize '{ts_name}'")
            if tma_map:
                print(f"\n=== FlagTree dep_analyzer tma device/host: {getattr(jit_fn, '__name__', 'unknown')} ===")
                for desc_name, bs_names_set in tma_map.items():
                    print(f"  TMA_desc '{desc_name}' is related to BlockSize '{bs_names_set}'")
            if bs_m_map or bs_k_map:
                print(f"\n=== FlagTree dep_analyzer tma tl.dot: {getattr(jit_fn, '__name__', 'unknown')} ===")
                print(f"  BlockSize_M - set(InputParam) dict: {bs_m_map}")
                print(f"  BlockSize_K - set(InputParam) dict: {bs_k_map}")
            print("================================================\n")

        return (load_map, tma_map, bs_m_map, bs_k_map)

    except Exception as e:
        print(f"Warning: dep_analyzer failed: {e}")
        return {}


def clear_analysis_cache():
    _analysis_cache.clear()