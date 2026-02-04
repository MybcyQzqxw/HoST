
"""
轻量化奖励函数 shape 检查脚本。

这个脚本不会导入 `legged_gym` 或 `isaacgym` 的重型模块（避免触发 IsaacGym C++ 扩展编译），
而是直接解析 `host_ground_prone.py` 中 `LeggedRobot` 类的源码，提取以 `_reward_` 开头的方法，
将方法动态定义并绑定到一个轻量 fake 实例上，然后调用以报告返回值 shape 或错误信息。

用法：
    python test_reward_shapes.py

可通过环境变量 `NUM_ENVS` / `NUM_DOFS` 控制假实例的大小（默认 512 / 12）：
    NUM_ENVS=8 NUM_DOFS=12 python test_reward_shapes.py
"""

import ast
import os
import sys
import textwrap
import traceback
import torch


# 配置：从环境变量读取，默认为 512、12
NUM_ENVS = int(os.environ.get('NUM_ENVS', '512'))
NUM_DOFS = int(os.environ.get('NUM_DOFS', '12'))
NUM_FEET = int(os.environ.get('NUM_FEET', '4'))

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, 'legged_gym', 'legged_gym', 'envs', 'base', 'host_ground_prone.py')


def extract_reward_methods(source_path):
    src = open(source_path, 'r', encoding='utf-8').read()
    mod = ast.parse(src)
    # 找到 LeggedRobot 类
    target_class = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == 'LeggedRobot':
            target_class = node
            break
    if target_class is None:
        raise RuntimeError('未找到 LeggedRobot 类')

    methods = []
    for node in target_class.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('_reward_'):
            # 获取源码片段
            start, end = node.lineno - 1, node.end_lineno
            lines = src.splitlines()[start:end]
            # 去掉类缩进（通常前置 4 个空格）
            dedented = textwrap.dedent('\n'.join(lines))
            methods.append((node.name, dedented, node))
    return methods


def collect_nested_self_attrs(node):
    # 收集 self.X.Y 中的 X->Y 映射
    nested = {}

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, n: ast.Attribute):
            # 处理 self.X.Y 或 self.X.Y.Z
            if isinstance(n.value, ast.Attribute):
                val = n.value
                if isinstance(val.value, ast.Name) and val.value.id == 'self':
                    top = val.attr
                    sub = n.attr
                    nested.setdefault(top, set()).add(sub)
            self.generic_visit(n)

    Visitor().visit(node)
    return nested


def collect_self_attrs(node):
    # 在函数 AST 中收集 self.xxx 的属性名
    attrs = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, n: ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id == 'self':
                if isinstance(n.attr, str):
                    attrs.add(n.attr)
            self.generic_visit(n)

    Visitor().visit(node)
    return attrs


def make_fake_instance(attr_names, num_envs=NUM_ENVS, num_dofs=NUM_DOFS, num_feet=NUM_FEET):
    class Fake:
        pass

    f = Fake()
    # 常见属性映射到合适的 shape 或类型（更全面的启发式映射）
    num_bodies = max(32, num_dofs * 2)
    for name in attr_names:
        if name in ('base_lin_vel', 'base_ang_vel', 'projected_gravity'):
            setattr(f, name, torch.zeros(num_envs, 3))
        elif name in ('num_bodies',):
            setattr(f, name, num_bodies)
        elif name in ('root_states',):
            setattr(f, name, torch.zeros(num_envs, 13))
        elif name in ('dof_pos', 'dof_vel', 'actions', 'torques', 'last_dof_vel', 'last_actions', 'target_dof_pos'):
            setattr(f, name, torch.zeros(num_envs, num_dofs))
        elif name in ('action_rescale', 'force'):
            setattr(f, name, torch.ones(num_envs, 1))
        elif name in ('feet_pos', 'feet_vel'):
            setattr(f, name, torch.zeros(num_envs, num_feet, 3))
        elif name in ('feet_quat',):
            setattr(f, name, torch.zeros(num_envs, num_feet, 4))
        elif name == 'contact_forces':
            setattr(f, name, torch.zeros(num_envs, num_bodies, 3))
        elif name in ('rigid_body_states',):
            setattr(f, name, torch.zeros(num_envs, num_bodies, 13))
        elif name.endswith('_indices'):
            # 常见索引应为 long tensor
            # 处理各种关节和刚体索引
            if 'shoulder' in name or 'elbow' in name or 'hip' in name or 'knee' in name or 'ankle' in name or 'waist' in name:
                # 关节索引通常是小数量的
                setattr(f, name, torch.tensor([0], dtype=torch.long))
            elif 'foot' in name or 'feet' in name:
                setattr(f, name, torch.tensor([0], dtype=torch.long))
            elif 'head' in name or 'base' in name or 'keyframe' in name or 'thigh' in name:
                setattr(f, name, torch.tensor([0], dtype=torch.long))
            elif 'arm' in name or 'leg' in name:
                # arm/leg joint indices 可能包含多个关节
                setattr(f, name, torch.arange(min(4, num_dofs), dtype=torch.long))
            elif 'penalis' in name or 'termination' in name:
                setattr(f, name, torch.arange(min(8, num_bodies), dtype=torch.long))
            else:
                setattr(f, name, torch.tensor([0], dtype=torch.long))
        elif name.endswith('_index'):
            # 单一索引（int）
            setattr(f, name, 0)
        elif name.endswith('_buf') or name.endswith('buf'):
            setattr(f, name, torch.zeros(num_envs))
        elif name in ('dof_pos_limits',):
            setattr(f, name, torch.zeros(num_dofs, 2))
        elif name in ('dof_vel_limits', 'torque_limits'):
            setattr(f, name, torch.zeros(num_dofs))
        elif name in ('default_rigid_body_mass',):
            setattr(f, name, torch.zeros(num_bodies))
        elif name in ('old_headheight', 'max_headheight', 'feet_ori'):
            setattr(f, name, torch.zeros(num_envs, 1))
        elif name in ('rewards', 'reward_scales', 'constraint_scales'):
            setattr(f, name, {})
        elif name == 'cfg':
            # 预置一个简单 cfg，用于访问 cfg.rewards / cfg.constraints 等属性
            from types import SimpleNamespace
            cfg = SimpleNamespace()
            cfg.rewards = SimpleNamespace()
            cfg.constraints = SimpleNamespace()
            cfg.limitation = SimpleNamespace()
            cfg.env = SimpleNamespace()
            # 常用默认值（可根据需要扩展）
            # Phase相关配置
            cfg.rewards.target_base_height_phase1 = 0.2
            cfg.rewards.target_base_height_phase2 = 0.25
            cfg.rewards.target_base_height_phase3 = 0.3
            cfg.rewards.target_base_height = 0.8
            cfg.rewards.base_height_target = 0.3
            cfg.rewards.target_base_margin = 0.05
            # Orientation相关
            cfg.rewards.orientation_sigma = -1.0
            cfg.rewards.orientation_threshold = 0.1
            cfg.rewards.target_orientation_sigma = -1.0
            # Height相关
            cfg.rewards.target_head_height = 0.2
            cfg.rewards.target_head_margin = 0.1
            cfg.rewards.target_base_height_sigma = -1.0
            # DOF相关
            cfg.rewards.target_dof_pos_sigma = -1.0
            cfg.rewards.soft_dof_pos_limit = 0.95
            cfg.rewards.soft_dof_vel_limit = 0.9
            cfg.rewards.soft_torque_limit = 0.9
            # Displacement相关
            cfg.rewards.left_foot_displacement_sigma = -1.0
            cfg.rewards.right_foot_displacement_sigma = -1.0
            cfg.rewards.after2_left_foot_displacement_sigma = -1.0
            cfg.rewards.after2_right_foot_displacement_sigma = -1.0
            # Contact相关
            cfg.rewards.max_contact_force = 100.0
            # Tracking相关
            cfg.rewards.tracking_dof_sigma = 1.0
            # Deviation相关
            cfg.rewards.shoulder_roll_deviation_off_center_threshold = 0.1
            cfg.rewards.shoulder_roll_deviation_inside_threshold = 0.1
            cfg.rewards.shoulder_yaw_deviation_off_center_threshold = 0.1
            cfg.rewards.shoulder_yaw_deviation_inside_threshold = 0.1
            cfg.rewards.waist_deviation_threshold = 0.1
            cfg.rewards.hip_yaw_deviation_off_center_threshold = 0.1
            cfg.rewards.hip_yaw_deviation_inside_threshold = 0.1
            cfg.rewards.hip_roll_deviation_off_center_threshold = 0.1
            cfg.rewards.hip_roll_deviation_inside_threshold = 0.1
            cfg.rewards.ankle_roll_deviation_off_center_threshold = 0.1
            cfg.rewards.ankle_roll_deviation_inside_threshold = 0.1
            # After2相关（phase2之后）
            cfg.rewards.after2_base_ang_vel_xyz_sigma = -1.0
            cfg.rewards.after2_base_lin_vel_xy_sigma = -1.0
            cfg.rewards.after2_upper_body_deviation_sigma = -1.0
            cfg.rewards.after2_lower_body_deviation_sigma = -1.0
            cfg.rewards.after2_arm_pos_sigma = -1.0
            cfg.rewards.after2_leg_ori_threshold = 0.8
            cfg.rewards.after2_feet_distance_threshold = 0.3
            cfg.rewards.after2_feet_height_var_sigma = -1.0
            # Before1相关（phase1之前）
            cfg.rewards.before1_thigh_ori_threshold = 0.5
            cfg.rewards.before1_shank_ori_threshold = 0.5
            # Before2相关（phase2之前）
            cfg.rewards.before2_base_ang_vel_xz_sigma = -1.0
            cfg.rewards.before2_base_lin_vel_y_sigma = -1.0
            # After1 Before2相关（phase1和phase2之间）
            cfg.rewards.after1_before2_shank_ori_threshold = 0.5
            # Hip yaw var
            cfg.rewards.hip_yaw_var_sigma = -1.0
            # Constraints
            cfg.constraints.post_task = False
            cfg.constraints.hip_yaw_var_sigma = -1.0
            cfg.constraints.only_positive_rewards = False
            # Limitation
            cfg.limitation.dof_vel_limit = 1e3
            cfg.limitation.base_vel_limit = 1e3
            cfg.limitation.soft_dof_vel_limit = 0.9
            # Env
            cfg.env.num_dofs = num_dofs
            setattr(f, name, cfg)
        elif name == 'obs_scales':
            # simple object with expected attributes
            class ObsScales:
                ang_vel = 1.0
                dof_pos = 1.0
                dof_vel = 1.0
                lin_vel = 1.0

            setattr(f, name, ObsScales())
        elif name == 'motions':
            # minimal stub
            class MotionsStub:
                def check_timeout(self, *args, **kwargs):
                    return torch.zeros(num_envs, dtype=torch.bool)

                def sample_motions(self, n):
                    return torch.zeros(n, dtype=torch.long)

            setattr(f, name, MotionsStub())
        elif name in ('dof_names', 'body_names', 'head_names'):
            setattr(f, name, [])
        elif name in ('penalised_contact_indices', 'termination_contact_indices'):
            setattr(f, name, torch.arange(min(8, num_bodies), dtype=torch.long))
        elif name in ('feet_air_time', 'last_contacts'):
            setattr(f, name, torch.zeros(num_envs, num_feet))
        elif name in ('last_last_actions',):
            setattr(f, name, torch.zeros(num_envs, num_dofs))
        elif name in ('last_last_dof_pos', 'last_dof_pos'):
            setattr(f, name, torch.zeros(num_envs, num_dofs))
        elif name in ('last_dof_vel', 'last_root_vel'):
            setattr(f, name, torch.zeros(num_envs, num_dofs))
        elif name == 'last_actions':
            setattr(f, name, torch.zeros(num_envs, num_dofs))
        elif name in ('old_baseheight', 'max_baseheight', 'old_headheight', 'max_headheight'):
            setattr(f, name, torch.zeros(num_envs, 1))
        elif name in ('feet_ori',):
            setattr(f, name, torch.zeros(num_envs, 1))
        elif name in ('real_episode_length_buf',):
            setattr(f, name, torch.zeros(num_envs))
        else:
            # 兜底为一维 num_envs 向量
            # 特殊字段 is_gaussian 应为布尔而非张量
            if name == 'is_gaussian':
                setattr(f, name, False)
            else:
                setattr(f, name, torch.zeros(num_envs))

    # 一些小辅助属性可能被访问
    f.num_envs = num_envs
    f.num_real_dofs = num_dofs
    f.device = 'cpu'
    # 常见仿真步长（强制使用标量以避免广播维度问题）
    f.dt = 0.02
    # 确保常见的 buffer 在逻辑上下文中拥有合适的 dtype
    if not hasattr(f, 'reset_buf'):
        f.reset_buf = torch.zeros(num_envs, dtype=torch.bool)
    else:
        if isinstance(getattr(f, 'reset_buf'), torch.Tensor) and getattr(f, 'reset_buf').dtype != torch.bool:
            f.reset_buf = getattr(f, 'reset_buf').to(dtype=torch.bool)
        elif not isinstance(getattr(f, 'reset_buf'), torch.Tensor):
            f.reset_buf = torch.zeros(num_envs, dtype=torch.bool)
    if not hasattr(f, 'time_out_buf'):
        f.time_out_buf = torch.zeros(num_envs, dtype=torch.bool)
    else:
        if isinstance(getattr(f, 'time_out_buf'), torch.Tensor) and getattr(f, 'time_out_buf').dtype != torch.bool:
            f.time_out_buf = getattr(f, 'time_out_buf').to(dtype=torch.bool)
        elif not isinstance(getattr(f, 'time_out_buf'), torch.Tensor):
            f.time_out_buf = torch.zeros(num_envs, dtype=torch.bool)

    return f


def make_stub_helpers(globals_ns):
    # 添加一些可能被调用到的外部函数的 stub
    def quat_rotate_inverse(q, v):
        return v

    def get_euler_xyz_in_tensor(q):
        return torch.zeros(q.shape[0], 3)

    globals_ns['quat_rotate_inverse'] = quat_rotate_inverse
    globals_ns['get_euler_xyz_in_tensor'] = get_euler_xyz_in_tensor
    # 常用全局
    try:
        import numpy as np
        globals_ns['np'] = np
    except Exception:
        globals_ns['np'] = None

    def tolerance(x, bounds, margin, sigma):
        # 简单可调用替代：返回与 x 相同形状的 0/1 张量（这里返回 ones，表示通过）
        try:
            return torch.ones_like(x, dtype=torch.float)
        except Exception:
            return 1.0

    globals_ns['tolerance'] = tolerance


def test_reward_function_shapes():
    methods = extract_reward_methods(SRC)
    print(f'发现 {len(methods)} 个 _reward_ 方法')

    # 合并所有被访问到的 self.attr
    all_attrs = set()
    for name, src_code, node in methods:
        attrs = collect_self_attrs(node)
        all_attrs.update(attrs)

    # 收集嵌套属性 self.X.Y 并准备可调用属性信息
    nested = {}
    callable_attrs = set()
    for name, src_code, node in methods:
        nested.update(collect_nested_self_attrs(node))
        # 查找 self.X(...) 形式，标记 X 为需 callable

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, cn):
                f = cn.func
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'self':
                    callable_attrs.add(f.attr)
                self.generic_visit(cn)
        CallVisitor().visit(node)

    fake = make_fake_instance(all_attrs)

    # 为 nested top-level 名称创建 SimpleNamespace 并注入子属性
    from types import SimpleNamespace
    for top, subs in nested.items():
        if hasattr(fake, top) and not isinstance(getattr(fake, top), SimpleNamespace):
            # 如果已有 tensor 等，替换为 namespace 并 keep a tensor accessible as `.tensor`
            existing = getattr(fake, top)
            ns = SimpleNamespace()
            setattr(ns, 'tensor', existing)
            setattr(fake, top, ns)
        elif not hasattr(fake, top):
            setattr(fake, top, SimpleNamespace())
        ns = getattr(fake, top)
        for sub in subs:
            if not hasattr(ns, sub):
                # 根据 sub 名称预置合理默认
                if sub in ('rewards', 'constraints', 'reward_scales', 'constraint_scales'):
                    # 使用 SimpleNamespace 以支持属性访问
                    sn = SimpleNamespace()
                    # 常见 rewards 字段
                    sn.target_base_height_phase1 = 0.2
                    sn.target_base_height_phase2 = 0.25
                    sn.target_base_height_phase3 = 0.3
                    sn.base_height_target = 0.3
                    sn.target_base_height = 0.8
                    sn.target_base_margin = 0.05
                    sn.orientation_sigma = -1.0
                    sn.orientation_threshold = 0.1
                    sn.target_orientation_sigma = -1.0
                    sn.target_head_height = 0.2
                    sn.target_head_margin = 0.1
                    sn.target_base_height_sigma = -1.0
                    sn.target_dof_pos_sigma = -1.0
                    sn.left_foot_displacement_sigma = -1.0
                    sn.right_foot_displacement_sigma = -1.0
                    sn.soft_dof_pos_limit = 0.95
                    sn.soft_dof_vel_limit = 0.9
                    sn.soft_torque_limit = 0.9
                    sn.max_contact_force = 100.0
                    sn.tracking_dof_sigma = 1.0
                    sn.hip_yaw_var_sigma = -1.0
                    # Deviation thresholds
                    sn.shoulder_roll_deviation_off_center_threshold = 0.1
                    sn.shoulder_roll_deviation_inside_threshold = 0.1
                    sn.shoulder_yaw_deviation_off_center_threshold = 0.1
                    sn.shoulder_yaw_deviation_inside_threshold = 0.1
                    sn.waist_deviation_threshold = 0.1
                    sn.hip_yaw_deviation_off_center_threshold = 0.1
                    sn.hip_yaw_deviation_inside_threshold = 0.1
                    sn.hip_roll_deviation_off_center_threshold = 0.1
                    sn.hip_roll_deviation_inside_threshold = 0.1
                    sn.ankle_roll_deviation_off_center_threshold = 0.1
                    sn.ankle_roll_deviation_inside_threshold = 0.1
                    # After2 configs
                    sn.after2_base_ang_vel_xyz_sigma = -1.0
                    sn.after2_base_lin_vel_xy_sigma = -1.0
                    sn.after2_upper_body_deviation_sigma = -1.0
                    sn.after2_lower_body_deviation_sigma = -1.0
                    sn.after2_arm_pos_sigma = -1.0
                    sn.after2_leg_ori_threshold = 0.8
                    sn.after2_feet_distance_threshold = 0.3
                    sn.after2_feet_height_var_sigma = -1.0
                    sn.after2_left_foot_displacement_sigma = -1.0
                    sn.after2_right_foot_displacement_sigma = -1.0
                    # Before1 configs
                    sn.before1_thigh_ori_threshold = 0.5
                    sn.before1_shank_ori_threshold = 0.5
                    # Before2 configs
                    sn.before2_base_ang_vel_xz_sigma = -1.0
                    sn.before2_base_lin_vel_y_sigma = -1.0
                    # After1 Before2 configs
                    sn.after1_before2_shank_ori_threshold = 0.5
                    # Constraints
                    sn.post_task = False
                    sn.only_positive_rewards = False
                    setattr(ns, sub, sn)
                elif sub.endswith('_indices'):
                    setattr(ns, sub, torch.arange(min(8, NUM_DOFS), dtype=torch.long))
                else:
                    setattr(ns, sub, torch.zeros(NUM_ENVS))

    # 为需 callable 的属性注入简单可调用对象
    for a in callable_attrs:
        if not hasattr(fake, a) or not callable(getattr(fake, a)):
            # 根据函数名返回合理形状的 stub
            def make_stub(name):
                def _stub(*args, **kwargs):
                    # 需要返回与真实函数期望的形状相匹配
                    if 'shoulder' in name:
                        return torch.zeros(NUM_ENVS, 2)
                    if 'base_heights' in name or 'base_height' in name:
                        return torch.zeros(NUM_ENVS)
                    if 'motion' in name or 'motions' in name:
                        return torch.zeros(NUM_ENVS, dtype=torch.bool)
                    # 默认返回 per-env 标量
                    return torch.zeros(NUM_ENVS)
                return _stub

            setattr(fake, a, make_stub(a))

    # 动态构建命名空间并定义方法
    ns = {'torch': torch}
    make_stub_helpers(ns)

    # 先 exec 所有方法的函数体（作为顶级函数），再绑定到 Fake
    for name, src_code, node in methods:
        # src_code 是缩进已去的函数定义块，直接执行会定义函数
        try:
            exec(src_code, ns)
        except Exception:
            print(f'无法解析/定义 {name}，将在调用时跳过。')

    # 绑定到实例
    for name, src_code, node in methods:
        fn = ns.get(name)
        if fn is None:
            print(f'{name:30s} 未定义（跳过）')
            continue
        # 将函数绑定为方法
        import types

        bound = types.MethodType(fn, fake)
        setattr(fake, name, bound)

    # 逐个调用并记录输出 shape / 错误
    results = []
    for name, src_code, node in methods:
        fn = getattr(fake, name, None)
        if fn is None:
            results.append((name, 'SKIPPED', 'not defined'))
            continue
        try:
            out = fn()
            if isinstance(out, torch.Tensor):
                results.append((name, 'OK', tuple(out.shape)))
            else:
                results.append((name, 'OK', type(out).__name__))
        except Exception as e:
            tb = traceback.format_exc()
            results.append((name, 'ERROR', str(e).splitlines()[-1]))

    # 打印汇总
    max_name = max((len(r[0]) for r in results), default=20)
    print('\n' + '=' * 80)
    print('奖励函数输出维度检测结果：')
    print('=' * 80)
    
    # 统计
    total = len(results)
    ok_count = sum(1 for r in results if r[1] == 'OK')
    error_count = sum(1 for r in results if r[1] == 'ERROR')
    skipped_count = sum(1 for r in results if r[1] == 'SKIPPED')
    
    # 按状态分类打印
    print(f'\n✓ 成功 ({ok_count}/{total}):')
    print('-' * 80)
    for name, status, info in results:
        if status == 'OK':
            # 检查输出维度是否正确（应该是 (NUM_ENVS,) 或标量）
            if isinstance(info, tuple):
                if info == (NUM_ENVS,):
                    indicator = '✓'
                else:
                    indicator = '⚠'  # 维度不符合预期
            else:
                indicator = '?'  # 非张量输出
            print(f'  {indicator} {name:<{max_name}}  {info}')
    
    if error_count > 0:
        print(f'\n✗ 错误 ({error_count}/{total}):')
        print('-' * 80)
        for name, status, info in results:
            if status == 'ERROR':
                print(f'  ✗ {name:<{max_name}}  {info}')
    
    if skipped_count > 0:
        print(f'\n⊘ 跳过 ({skipped_count}/{total}):')
        print('-' * 80)
        for name, status, info in results:
            if status == 'SKIPPED':
                print(f'  ⊘ {name:<{max_name}}  {info}')
    
    print('\n' + '=' * 80)
    print(f'总结: {ok_count} 成功, {error_count} 错误, {skipped_count} 跳过 / 共 {total} 个')
    print('=' * 80)
    
    # 检查是否所有成功的函数都返回正确的形状
    wrong_shape = []
    for name, status, info in results:
        if status == 'OK' and isinstance(info, tuple):
            if info != (NUM_ENVS,):
                wrong_shape.append((name, info))
    
    if wrong_shape:
        print(f'\n⚠ 警告: {len(wrong_shape)} 个函数返回了非预期的形状（应为 ({NUM_ENVS},)）:')
        for name, shape in wrong_shape:
            print(f'  {name}: {shape}')
    
    return ok_count == total and len(wrong_shape) == 0


if __name__ == '__main__':
    success = test_reward_function_shapes()
    sys.exit(0 if success else 1)
