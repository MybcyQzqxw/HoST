"""
测试角速度方向：给机器人施加y轴角速度，观察旋转方向
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'legged_gym'))

from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import torch


class AngularVelocityTester:
    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self._create_sim()
        self._load_robot()
        self._create_envs()
        self._create_viewer()
        
    def _create_sim(self):
        """创建仿真"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.max_depenetration_velocity = 1.0
        
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        
        # 添加地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
    def _load_robot(self):
        """加载机器人"""
        asset_root = os.path.join(os.path.dirname(__file__), 
                                   'legged_gym/resources/robots/bhr8fc2')
        asset_file = "BHR8FC2.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False  # 不固定base，允许旋转观察
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS  # 位置控制模式
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 1000.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.armature = 0.01
        asset_options.thickness = 0.01
        asset_options.disable_gravity = True  # 关闭重力，悬空
        
        self.robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        
        # 获取关节信息
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        print(f"机器人DOF数量: {self.num_dof}")
        
    def _create_envs(self):
        """创建1个环境"""
        spacing = 3.0
        lower = gymapi.Vec3(-spacing, 0.0, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.envs = []
        self.actors = []
        
        # 只创建1个环境：施加 +0.5 rad/s
        env = self.gym.create_env(self.sim, lower, upper, 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.8)  # 站立高度
        pose.r = gymapi.Quat(0, 0, 0, 1)
        actor = self.gym.create_actor(env, self.robot_asset, pose, "robot", 0, 1)
        self.envs.append(env)
        self.actors.append(actor)
        
        # 设置关节驱动属性（高刚度PD控制保持姿态）
        dof_props = self.gym.get_actor_dof_properties(env, actor)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = 500.0  # 高刚度
            dof_props['damping'][i] = 50.0     # 适当阻尼
        self.gym.set_actor_dof_properties(env, actor, dof_props)
        
        # 设置关节为站立姿态
        dof_targets = np.zeros(self.num_dof, dtype=np.float32)
        self.gym.set_actor_dof_position_targets(env, actor, dof_targets)
        
        # 强制设置关节初始状态
        dof_state = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        for i in range(self.num_dof):
            dof_state['pos'][i] = 0.0
            dof_state['vel'][i] = 0.0
        self.gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)
        
        # 准备仿真
        self.gym.prepare_sim(self.sim)
        
        # 获取root state tensor
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_tensor)
        
        print("="*80)
        print("机器人已创建：悬空无重力，关节保持站立姿态")
        print("="*80)
        
    def _create_viewer(self):
        """创建viewer"""
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        
        # 设置相机位置
        cam_pos = gymapi.Vec3(0, -6, 2)
        cam_target = gymapi.Vec3(0, 0, 0.8)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
    def run(self):
        """运行测试"""
        print("\n开始测试...")
        print("观察机器人的旋转方向（观察头部运动）")
        print("机器人悬空无重力，关节保持站立姿态\n")
        
        step = 0
        
        # 初始化位置
        self.root_states[0, 0:3] = torch.tensor([0.0, 0.0, 0.8])
        self.root_states[0, 3:7] = torch.tensor([0, 0, 0, 1])
        self.root_states[0, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            # 步进仿真前先刷新状态
            if step > 0:
                self.gym.refresh_actor_root_state_tensor(self.sim)
            
            # 持续施加y轴角速度
            if step < 180:  # 前3秒
                # 强制只保留y轴角速度，清零x和z轴
                self.root_states[0, 10] = 0.0   # x轴角速度 = 0
                self.root_states[0, 11] = +0.5  # y轴角速度 = +0.5
                self.root_states[0, 12] = 0.0   # z轴角速度 = 0
                # 同时清零线速度，防止平移
                self.root_states[0, 7] = 0.0    # x轴线速度 = 0
                self.root_states[0, 8] = 0.0    # y轴线速度 = 0
                self.root_states[0, 9] = 0.0    # z轴线速度 = 0
                self.gym.set_actor_root_state_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.root_states))
            
            # 步进仿真
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 每30帧打印一次pitch角度
            if step % 30 == 0 and step > 0:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                
                def get_pitch(quat):
                    """从四元数提取pitch角"""
                    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
                    pitch = np.arcsin(np.clip(2.0 * (w*y - z*x), -1.0, 1.0))
                    return np.degrees(pitch)
                
                pitch = get_pitch(self.root_states[0, 3:7].cpu().numpy())
                ang_vel_x = self.root_states[0, 10].item()
                ang_vel_y = self.root_states[0, 11].item()
                ang_vel_z = self.root_states[0, 12].item()
                
                print(f"Step {step:3d} | pitch={pitch:+6.1f}° | "
                      f"ang_vel=[{ang_vel_x:+.3f}, {ang_vel_y:+.3f}, {ang_vel_z:+.3f}]")
            
            # 仿真步进
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 渲染
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            
            step += 1
            
            # 3秒后总结
            if step == 180:
                print("\n" + "="*80)
                print("【结果分析】")
                print("="*80)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                
                def get_pitch(quat):
                    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
                    pitch = np.arcsin(2.0 * (w*y - z*x))
                    return np.degrees(pitch)
                
                final_pitch = get_pitch(self.root_states[0, 3:7].cpu().numpy())
                
                print(f"\n施加 +0.5 rad/s 的y轴角速度后:")
                print(f"  最终pitch角 = {final_pitch:+.1f}°")
                
                if final_pitch > 5:
                    print("  → pitch角【增大】（头往下倾/前倾）")
                    direction = "增大"
                elif final_pitch < -5:
                    print("  → pitch角【减小】（头往上仰/后仰）")
                    direction = "减小"
                else:
                    print("  → pitch角【几乎不变】")
                    direction = "不变"
                
                print("\n" + "="*80)
                print("【结论】")
                print("="*80)
                print(f"  y轴角速度 = +0.5 rad/s → pitch角{direction}")
                
                print("\n【起身动作需要】")
                print("  从趴着(pitch≈+90°)到站立(pitch≈0°)")
                print("  → pitch需要【减小】")
                
                if direction == "减小":
                    print("\n  ✓ 正角速度让pitch减小")
                    print("  ✓ 起身时需要【正的】y轴角速度")
                    print("\n  推荐代码:")
                    print("    reward = torch.clamp(pitch_ang_vel, 0.0, 1.0)")
                elif direction == "增大":
                    print("\n  ✗ 正角速度让pitch增大(错误方向)")
                    print("  ✓ 起身时需要【负的】y轴角速度")
                    print("\n  推荐代码:")
                    print("    reward = torch.clamp(-pitch_ang_vel, 0.0, 1.0)")
                
                print("\n按ESC退出或继续观察...\n")
        
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == "__main__":
    tester = AngularVelocityTester()
    tester.run()
