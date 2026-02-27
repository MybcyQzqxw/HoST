"""
机器人硬直姿态测试脚本
在 Isaac Gym 中设置机器人关节角度，保持硬直姿态，并定时输出连杆高度
"""

import os
import sys
import time

# 设置环境变量（修复 libpython3.8.so.1.0 找不到的问题）
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if conda_prefix:
    lib_path = os.path.join(conda_prefix, 'lib')
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}"

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'legged_gym'))

from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import torch


# ================================================================================
# ========================== 配置区域 - 请在这里修改配置 ==========================
# ================================================================================

# ==================== 输出配置（最重要的配置在这里！）====================
OUTPUT_LINK_NAME = "torso_link"  # 要输出高度的连杆名称

PRINT_INTERVAL = 1.0       # 打印间隔（秒）

# ==================== 关节角度配置（单位：度）====================
# 所有关节角度都在这里设置，脚本会自动转换为弧度
TARGET_JOINT_ANGLES_DEG = {
    # phase1
    # 腿部关节
    'left_hip_yaw_joint': 0.0,
    'right_hip_yaw_joint': 0.0,
    'left_hip_roll_joint': 0.0,
    'right_hip_roll_joint': 0.0,
    'left_hip_pitch_joint': 90,
    'right_hip_pitch_joint': -90,
    'left_knee_joint': -90,
    'right_knee_joint': 90,
    'left_ankle_pitch_joint': -30,
    'right_ankle_pitch_joint': 30,
    'left_ankle_roll_joint': 0.0,
    'right_ankle_roll_joint': 0.0,
    
    # 手臂关节
    'left_shoulder_pitch_joint': -90,
    'right_shoulder_pitch_joint': 90,
    'left_shoulder_roll_joint': 0.0,
    'right_shoulder_roll_joint': 0.0,
    'left_shoulder_yaw_joint': 0.0,
    'right_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': -40,
    'right_elbow_joint': 40,
    
    # 腰部关节（如果有）
    'waist_yaw_joint': 0.0,
    'waist_pitch_joint': 0.0,
    'waist_roll_joint': 0.0,
    
    # phase2
    # # 腿部关节
    # 'left_hip_yaw_joint': 0.0,
    # 'right_hip_yaw_joint': 0.0,
    # 'left_hip_roll_joint': 0.0,
    # 'right_hip_roll_joint': 0.0,
    # 'left_hip_pitch_joint': 60,
    # 'right_hip_pitch_joint': -60,
    # 'left_knee_joint': -120,
    # 'right_knee_joint': 120,
    # 'left_ankle_pitch_joint': 60,
    # 'right_ankle_pitch_joint': -60,
    # 'left_ankle_roll_joint': 0.0,
    # 'right_ankle_roll_joint': 0.0,
    
    # # 手臂关节
    # 'left_shoulder_pitch_joint': -90,
    # 'right_shoulder_pitch_joint': 90,
    # 'left_shoulder_roll_joint': 0.0,
    # 'right_shoulder_roll_joint': 0.0,
    # 'left_shoulder_yaw_joint': 0.0,
    # 'right_shoulder_yaw_joint': 0.0,
    # 'left_elbow_joint': -60,
    # 'right_elbow_joint': 60,
    
    # # 腰部关节（如果有）
    # 'waist_yaw_joint': 0.0,
    # 'waist_pitch_joint': 0.0,
    # 'waist_roll_joint': 0.0,
    
    # phase3
    # # 腿部关节
    # 'left_hip_yaw_joint': 0.0,
    # 'right_hip_yaw_joint': 0.0,
    # 'left_hip_roll_joint': 0.0,
    # 'right_hip_roll_joint': 0.0,
    # 'left_hip_pitch_joint': 20,
    # 'right_hip_pitch_joint': -20,
    # 'left_knee_joint': -40,
    # 'right_knee_joint': 40,
    # 'left_ankle_pitch_joint': 20,
    # 'right_ankle_pitch_joint': -20,
    # 'left_ankle_roll_joint': 0.0,
    # 'right_ankle_roll_joint': 0.0,
    
    # # 手臂关节
    # 'left_shoulder_pitch_joint': -36,
    # 'right_shoulder_pitch_joint': 36,
    # 'left_shoulder_roll_joint': 0.0,
    # 'right_shoulder_roll_joint': 0.0,
    # 'left_shoulder_yaw_joint': 0.0,
    # 'right_shoulder_yaw_joint': 0.0,
    # 'left_elbow_joint': -60,
    # 'right_elbow_joint': 60,
    
    # # 腰部关节（如果有）
    # 'waist_yaw_joint': 0.0,
    # 'waist_pitch_joint': 0.0,
    # 'waist_roll_joint': 0.0,
}

# ==================== 基座初始位姿配置 ====================
BASE_POSITION = [0.0, 0.0, 1.0]  # x, y, z [m]
# BASE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]  # 四元数 [x, y, z, w] (站立)
BASE_ORIENTATION = [0.0, 1.0, 0.0, 1.0]  # 四元数 [x, y, z, w] (趴下)

# ==================== 机器人模型选择 ====================
ROBOT_NAME = "bhr8fc2"  # 可选: "bhr8fc2", "g1", "h1", "pi_12dof"

# ==================== 刚度配置 ====================
POSITION_STIFFNESS = 10000.0  # 位置刚度（越大越硬直）
VELOCITY_DAMPING = 1000.0     # 速度阻尼

# ================================================================================
# ============================== 配置区域结束 ====================================
# ================================================================================


class RobotStiffPoseTest:
    def __init__(self):
        """
        从全局配置初始化
        """
        # ==================== 从全局配置读取 ====================
        self.target_joint_angles_deg = TARGET_JOINT_ANGLES_DEG
        self.base_position = BASE_POSITION
        self.base_orientation = BASE_ORIENTATION
        self.robot_name = ROBOT_NAME
        self.position_stiffness = POSITION_STIFFNESS
        self.velocity_damping = VELOCITY_DAMPING
        self.print_interval = PRINT_INTERVAL
        self.output_link_name = OUTPUT_LINK_NAME
        
        # ==================== 机器人模型路径 ====================
        self.asset_root = os.path.join(os.path.dirname(__file__), 
                                       f"legged_gym/resources/robots/{self.robot_name}")
        urdf_files = {
            "bhr8fc2": "BHR8FC2.urdf",
            "g1": "g1.urdf",
            "h1": "h1.urdf",
            "pi_12dof": "robot.urdf",
        }
        self.asset_file = urdf_files.get(self.robot_name, "BHR8FC2.urdf")
        
        # ==================== 内部变量 ====================
        self.output_link_index = None  # 要输出的连杆索引（初始化后设置）
        self.last_print_time = 0.0
        
        # 转换角度为弧度
        self.target_joint_angles = {
            k: np.deg2rad(v) for k, v in self.target_joint_angles_deg.items()
        }
        
        # 初始化仿真
        self.gym = gymapi.acquire_gym()
        self._create_sim()
        self._load_robot()
        self._create_env()
        self._create_viewer()
        self._set_joint_angles()
        
    def _create_sim(self):
        """创建仿真环境"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # PhysX 参数
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 2
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.5
        sim_params.physx.max_depenetration_velocity = 1.0
        sim_params.physx.use_gpu = True
        
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("❌ 创建仿真失败")
            sys.exit(1)
            
        # 创建地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)
        
    def _load_robot(self):
        """加载机器人URDF"""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False  # 不固定基座，让机器人自然站立
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 1000.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        
        print(f"\n正在加载机器人: {self.robot_name}")
        print(f"路径: {os.path.join(self.asset_root, self.asset_file)}")
        
        self.robot_asset = self.gym.load_asset(self.sim, self.asset_root, 
                                                self.asset_file, asset_options)
        if self.robot_asset is None:
            print(f"❌ 加载模型失败: {os.path.join(self.asset_root, self.asset_file)}")
            sys.exit(1)
            
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        
        # 获取关节名称
        self.dof_names = [self.gym.get_asset_dof_name(self.robot_asset, i) 
                         for i in range(self.num_dof)]
        
        # 获取连杆名称
        self.body_names = [self.gym.get_asset_rigid_body_name(self.robot_asset, i) 
                          for i in range(self.num_bodies)]
        
        print(f"✓ 成功加载机器人 (DOF: {self.num_dof}, Bodies: {self.num_bodies})")
        
    def _create_env(self):
        """创建环境"""
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.env = self.gym.create_env(self.sim, lower, upper, 1)
        
        # 创建机器人actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.base_position)
        pose.r = gymapi.Quat(*self.base_orientation)
        
        self.robot_handle = self.gym.create_actor(self.env, self.robot_asset, 
                                                   pose, "robot", 0, 1)
        
        # 设置DOF属性（高刚度和阻尼以保持硬直）
        dof_props = self.gym.get_actor_dof_properties(self.env, self.robot_handle)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.position_stiffness
            dof_props['damping'][i] = self.velocity_damping
        self.gym.set_actor_dof_properties(self.env, self.robot_handle, dof_props)
        
    def _create_viewer(self):
        """创建可视化窗口"""
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 75.0
        camera_props.width = 1280
        camera_props.height = 720
        
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        if self.viewer is None:
            print("❌ 创建viewer失败")
            sys.exit(1)
            
        # 设置相机位置
        cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
        
        # 查找要输出的连杆索引
        self._find_output_link_index()
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
    def _set_joint_angles(self):
        """设置目标关节角度"""
        dof_targets = np.zeros(self.num_dof, dtype=np.float32)
        
        print("\n" + "="*60)
        print("关节角度设置:")
        print("="*60)
        
        for i, name in enumerate(self.dof_names):
            if name in self.target_joint_angles:
                dof_targets[i] = self.target_joint_angles[name]
                deg_val = self.target_joint_angles_deg[name]
                rad_val = self.target_joint_angles[name]
                print(f"  [{i:2d}] {name:30s}: {deg_val:7.2f}° ({rad_val:7.4f} rad)")
            else:
                dof_targets[i] = 0.0
                print(f"  [{i:2d}] {name:30s}: 未配置，使用默认值 0.0°")
                
        print("="*60)
        
        # 设置目标位置
        self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, dof_targets)
        
        # 强制设置初始状态
        dof_state = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
        for i in range(self.num_dof):
            dof_state['pos'][i] = dof_targets[i]
            dof_state['vel'][i] = 0.0
        self.gym.set_actor_dof_states(self.env, self.robot_handle, dof_state, gymapi.STATE_ALL)
        
    def _find_output_link_index(self):
        """查找要输出的连杆索引"""
        # 先输出所有可用的连杆名称
        print("\n" + "="*60)
        print(f"机器人所有连杆名称（共 {len(self.body_names)} 个）:")
        print("="*60)
        for i, name in enumerate(self.body_names):
            print(f"  [{i:2d}] {name}")
        print("="*60 + "\n")
        
        print(f"正在查找连杆: '{self.output_link_name}'")
        
        # 尝试精确匹配
        if self.output_link_name in self.body_names:
            self.output_link_index = self.body_names.index(self.output_link_name)
            print(f"✓ 找到连杆 '{self.output_link_name}' (索引: {self.output_link_index})")
            return
        
        # 尝试模糊匹配（不区分大小写）
        lower_name = self.output_link_name.lower()
        for i, name in enumerate(self.body_names):
            if lower_name in name.lower() or name.lower() in lower_name:
                self.output_link_index = i
                print(f"⚠ 未找到精确匹配，使用模糊匹配: '{name}' (索引: {i})")
                return
        
        # 如果都没找到，报错退出
        print(f"❌ 错误: 找不到连杆 '{self.output_link_name}'!")
        print(f"\n请修改脚本顶部的 OUTPUT_LINK_NAME 为上述名称之一。")
        sys.exit(1)
    def _print_link_height(self):
        """打印指定连杆的高度"""
        if self.output_link_index is None:
            return
            
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            # 获取刚体状态
            rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
            rb_states = gymtorch.wrap_tensor(rb_states)
            rb_states = rb_states.view(1, self.num_bodies, 13)  # (num_envs, num_bodies, 13)
            
            # 获取目标连杆的位置
            link_pos = rb_states[0, self.output_link_index, 0:3]  # x, y, z
            link_height = link_pos[2].item()  # z坐标
            link_x = link_pos[0].item()
            link_y = link_pos[1].item()
            
            # 打印信息
            elapsed = current_time - self.last_print_time if self.last_print_time > 0 else 0
            link_name = self.body_names[self.output_link_index]
            print(f"[{time.strftime('%H:%M:%S')}] {link_name:20s} | 高度: {link_height:7.4f} m | 位置: ({link_x:6.3f}, {link_y:6.3f}, {link_height:6.3f})")
            
            self.last_print_time = current_time
        
    def run(self):
        """运行仿真循环"""
        print("\n" + "="*60)
        print("机器人硬直姿态测试")
        print("="*60)
        print(f"机器人: {self.robot_name}")
        print(f"关节刚度: {self.position_stiffness}")
        print(f"关节阻尼: {self.velocity_damping}")
        print(f"输出间隔: {self.print_interval}s")
        print("="*60)
        print("按 ESC 退出")
        print("按 V 切换相机视角")
        print("="*60 + "\n")
        
        # 初始化时间
        self.last_print_time = time.time()
        
        iteration = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            # 步进仿真
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 定时打印连杆高度
            if iteration % 10 == 0:  # 每10帧检查一次（避免频繁调用）
                self._print_link_height()
            
            # 更新viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            
            iteration += 1
            
        # 清理
        print("\n正在退出...")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def main():
    print("="*60)
    print("机器人硬直姿态测试脚本")
    print("="*60)
    
    try:
        tester = RobotStiffPoseTest()
        tester.run()
    except KeyboardInterrupt:
        print("\n\n检测到 Ctrl+C，正在退出...")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
