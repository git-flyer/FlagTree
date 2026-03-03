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
        # desc 变量名 -> { "shape": [dim_names], "block_shape": [block_names] }（来自 make_tensor_descriptor 或 hook）
        self.tma_desc_defs: Dict[str, Dict[str, List[str]]] = {}
        # 每个变量的所有历史定义（按出现顺序），用于 arange 提取时不因后续覆盖而丢失信息
        self.var_all_definitions: Dict[str, List[ast.AST]] = {}

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

            # TMA device: 记录 make_tensor_descriptor 的 LHS 及 shape / block_shape（用于按 block_shape 分析 bs）
            if (isinstance(node.value, ast.Call) and
                    self._is_tl_make_tensor_descriptor(node.value)):
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

    def _is_tl_arange(self, node) -> bool:
        """检查是否是 tl.arange 或 triton.arange 调用"""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'arange':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _extract_arange_block_sizes(self, node: ast.AST) -> Set[str]:
        """从表达式中提取所有 tl.arange(?, size) 的 size 变量名（即 block 名）。"""
        out: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and self._is_tl_arange(child) and len(child.args) >= 2:
                if isinstance(child.args[1], ast.Name):
                    out.add(child.args[1].id)
        return out

    def _extract_arange_block_sizes_recursive(self, var_name: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """
        递归追踪变量的所有历史定义，提取其中 tl.arange(?, size) 的 size 名。

        之所以要搜索"所有历史定义"而不只是最后一个，是因为像
          rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 含 arange
          rm = rm.to(tl.int64)                           # 后续覆盖，丢失 arange
        这样的模式会让 var_definitions['rm'] 指向后者，而 arange 在前者里。
        """
        if visited is None:
            visited = set()
        if var_name in visited:
            return set()
        visited.add(var_name)

        out: Set[str] = set()
        for def_node in self.var_all_definitions.get(var_name, []):
            out.update(self._extract_arange_block_sizes(def_node))
            for child_var in VariableCollector.collect(def_node):
                if child_var != var_name and not child_var.startswith('pid'):
                    out.update(self._extract_arange_block_sizes_recursive(child_var, visited.copy()))
        return out

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
            return set()
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
        # Step 1: 为每个 TMA load 目标变量推断 block shape，来源为 desc.block_shape（make_tensor_descriptor 或 pre_hook），而非地址偏移依赖
        desc_block_shapes = getattr(self, "_desc_block_shapes", {})
        raw_var_block_shape: Dict[str, tuple] = {}
        for tma_info in self.tma_load_assignments:
            target_var = tma_info['var_name']
            tma_desc_name = tma_info['tma_desc_name']
            bs_names = list(desc_block_shapes.get(tma_desc_name) or [])
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


    # ---------- 分析函数一：仅针对普通 tl.load，通过地址依赖的 tl.arange 得到 dim -> BLOCK 的对应 ----------
    def analyze_tl_load_dim_to_bs(self) -> Dict[str, str]:
        """
        仅分析普通 tl.load：从地址表达式中找到依赖的 tl.arange(0, BLOCK_X)，
        用 BLOCK_X 作为该维的 bs；维度名 M/N/K 从该索引变量的 input 依赖推断。
        返回: { "M": "BLOCK_M", "N": "BLOCK_N", "K": "BLOCK_K" } 形式的 map。
        """
        load_map: Dict[str, str] = {}
        for addr_expr in self.load_addresses:
            used_vars = VariableCollector.collect(addr_expr)
            for var_name in used_vars:
                if var_name not in self.var_all_definitions or var_name.startswith("pid"):
                    continue
                # 用递归搜索所有历史定义，避免 rm = rm.to(int64) 覆盖含 arange 的早期定义
                blocks = self._extract_arange_block_sizes_recursive(var_name)
                input_deps, _ = self.get_dependencies(var_name)
                if len(input_deps) == 1 and len(blocks) == 1:
                    dim_name = list(input_deps)[0]
                    block_name = list(blocks)[0]
                    load_map[dim_name] = block_name
        return load_map

    # ---------- 分析函数二：仅针对 desc.load，通过 desc.block_shape 得到 M/N/K 与 BLOCK_* 及 a_desc/b_desc ----------
    def _parse_hook_block_shape_assignments(self, hook_ast: ast.FunctionDef) -> Dict[str, List[str]]:
        """解析 pre_hook 函数体中 nargs['desc_name'].block_shape = [B1, B2, ...] 的赋值，返回 desc_key -> [block_names]。"""
        hook_desc_block: Dict[str, List[str]] = {}
        for node in ast.walk(hook_ast):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            t = node.targets[0]
            if not isinstance(t, ast.Attribute) or t.attr != "block_shape":
                continue
            # t.value 应为 nargs["a_desc"] 等形式
            if not isinstance(t.value, ast.Subscript):
                continue
            sub = t.value
            if not isinstance(sub.value, ast.Name):
                continue
            key = None
            if isinstance(sub.slice, ast.Constant):
                key = sub.slice.value
            elif getattr(sub.slice, "value", None) is not None and isinstance(sub.slice.value, ast.Constant):
                key = sub.slice.value.value
            try:
                if key is None:
                    key = ast.literal_eval(sub.slice)
            except Exception:
                pass
            if not isinstance(key, str):
                continue
            if not isinstance(node.value, ast.List):
                continue
            block_names = []
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    block_names.append(elt.id)
            if block_names:
                hook_desc_block[key] = block_names
        return hook_desc_block

    def analyze_desc_load_dim_to_bs(
        self, pre_hook_fn: Optional[object] = None
    ) -> Tuple[Dict[str, Set[Tuple[str, ...]]], Dict[str, List[str]]]:
        """
        仅分析 desc.load（TMA device / host）：
        - TMA device：从 tl.make_tensor_descriptor(..., shape=..., block_shape=...) 读 block_shape，与 shape 对应得到各维的 BLOCK。
        - TMA host：描述符为入参，block_shape 可能在 pre_hook 里赋值为 nargs['a_desc'].block_shape = [BLOCK_M, BLOCK_K]；若传入 pre_hook_fn 则解析其 AST。
        返回:
          tma_map: 与现有 autotuner 兼容，{ desc_name: set of (block_name, ...) }，用于按 desc 调整 block。
          desc_block_shapes: { desc_name: [block_name, ...] }，供 analyze_dot_dim 使用。
        """
        # 1) 从 make_tensor_descriptor 得到 desc -> block_shape（及 shape）
        desc_block_shapes: Dict[str, List[str]] = {}
        for desc_name, defn in self.tma_desc_defs.items():
            blist = defn.get("block_shape") or []
            if blist:
                desc_block_shapes[desc_name] = list(blist)

        # 2) 从 pre_hook 解析 nargs["x"].block_shape = [...]，合并到 desc_block_shapes（覆盖或补充 host 描述符）
        if pre_hook_fn is not None and hasattr(pre_hook_fn, "__code__"):
            try:
                import inspect
                try:
                    hook_src = inspect.getsource(pre_hook_fn)
                except Exception:
                    hook_src = None
                if hook_src:
                    hook_ast = ast.parse(hook_src)
                    for n in ast.walk(hook_ast):
                        if isinstance(n, ast.FunctionDef):
                            for k, v in self._parse_hook_block_shape_assignments(n).items():
                                desc_block_shapes[k] = v
                            break
            except Exception:
                pass

        # 3) 构建 tma_map：每个 desc 的 load 对应一个 block 元组（考虑 trans 时交换最后两维）
        transpose_used_vars = set()
        for arg_node in self.transpose_args_nodes:
            for v in VariableCollector.collect(arg_node):
                transpose_used_vars.add(v)
                transpose_used_vars.update(self._get_dependencies_vars(v))

        tma_map: Dict[str, Set[Tuple[str, ...]]] = {}
        for tma_info in self.tma_load_assignments:
            desc_name = tma_info["tma_desc_name"]
            target_var = tma_info["var_name"]
            block_list = list(desc_block_shapes.get(desc_name) or [])
            if target_var in transpose_used_vars and len(block_list) >= 2:
                block_list[-1], block_list[-2] = block_list[-2], block_list[-1]
            if block_list:
                tma_map.setdefault(desc_name, set()).add(tuple(block_list))
        # 供 analyze_dot_dim 使用
        self._desc_block_shapes: Dict[str, List[str]] = desc_block_shapes
        return tma_map, desc_block_shapes


_analysis_cache: Dict[int, Tuple] = {}


def analyze_kernel_dependencies(jit_fn, pre_hook_fn: Optional[object] = None) -> Tuple:
    """
    分析 kernel 的 block size 依赖，供 autotuner 调整用。
    :param jit_fn: JIT 过的 kernel 函数
    :param pre_hook_fn: 可选，TMA host 时设置 block_shape 的 pre_hook（如 matmul_tma_set_block_size_hook），用于解析 nargs["a_desc"].block_shape = [...]
    :return: (load_map, tma_map, bs_m_map, bs_k_map)
      - load_map: 仅 tl.load，dim -> BLOCK（如 M -> BLOCK_M），由 tl.arange 推断
      - tma_map: 仅 desc.load，desc_name -> set of (block_name, ...)，由 desc.block_shape 推断
      - bs_m_map / bs_k_map: tl.dot 的 M/K 维 BLOCK -> 参与该维的 tensor 参数名集合
    """
    cache_key = (id(jit_fn), id(pre_hook_fn) if pre_hook_fn is not None else None)
    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]

    try:
        fn_ast = jit_fn.parse()
        analyzer = KernelDependencyAnalyzer()
        analyzer.visit(fn_ast)

        # 分析函数一：仅普通 tl.load，通过 tl.arange 得到 dim -> BLOCK
        load_map = analyzer.analyze_tl_load_dim_to_bs()

        # 分析函数二：仅 desc.load，通过 desc.block_shape（make_tensor_descriptor 或 pre_hook）得到 tma_map，并写入 _desc_block_shapes 供 dot 用
        tma_map, _ = analyzer.analyze_desc_load_dim_to_bs(pre_hook_fn=pre_hook_fn)

        # tl.dot 的 M/K 维映射（内部使用 _desc_block_shapes）
        bs_m_map, bs_k_map = analyzer.analyze_dot_dim()

        _analysis_cache[cache_key] = (load_map, tma_map, bs_m_map, bs_k_map)

        if knobs.autotuning.print:
            if load_map:
                print(f"\n=== FlagTree dep_analyzer tl.load (by tl.arange): {getattr(jit_fn, '__name__', 'unknown')} ===")
                for dim_name, block_name in load_map.items():
                    print(f"  dim '{dim_name}' -> block '{block_name}'")
            if tma_map:
                print(f"\n=== FlagTree dep_analyzer desc.load (by block_shape): {getattr(jit_fn, '__name__', 'unknown')} ===")
                for desc_name, bs_names_set in tma_map.items():
                    print(f"  desc '{desc_name}' -> block shapes {bs_names_set}")
            if bs_m_map or bs_k_map:
                print(f"\n=== FlagTree dep_analyzer tl.dot: {getattr(jit_fn, '__name__', 'unknown')} ===")
                print(f"  BLOCK_M -> params: {bs_m_map}")
                print(f"  BLOCK_K -> params: {bs_k_map}")
            print("================================================\n")

        return (load_map, tma_map, bs_m_map, bs_k_map)

    except Exception as e:
        print(f"Warning: dep_analyzer failed: {e}")
        return (None, None, None, None)


def clear_analysis_cache():
    _analysis_cache.clear()