from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BHR8FC2Cfg(LeggedRobotCfg):
    """
    Configuration for BHR8FC2 humanoid robot on ground terrain.
    """
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, -1, 0, 1.0]  # x,y,z,w [quat]
        
        target_joint_angles = {
            # 右腿关节
            'rhipYaw': 0.0,
            'rhipRoll': 0.0,
            'rhipPitch': 0.0,
            'rknee': 0.0,
            'rankle1': 0.0,
            'rankle2': 0.0,
            # 左腿关节
            'lhipYaw': 0.0,
            'lhipRoll': 0.0,
            'lhipPitch': 0.0,
            'lknee': 0.0,
            'lankle1': 0.0,
            'lankle2': 0.0,
            # 右臂关节
            'rshoulderPitch': 0.0,
            'rshoulderRoll': 0.0,
            'rshoulderYaw': 0.0,
            'relbow': 0.0,
            # 左臂关节
            'lshoulderPitch': 0.0,
            'lshoulderRoll': 0.0,
            'lshoulderYaw': 0.0,
            'lelbow': 0.0,
        }

        default_joint_angles = {
            # 右腿关节
            'rhipYaw': 0.0,
            'rhipRoll': 0.0,
            'rhipPitch': 0.0,
            'rknee': 0.0,
            'rankle1': 0.0,
            'rankle2': 0.0,
            # 左腿关节
            'lhipYaw': 0.0,
            'lhipRoll': 0.0,
            'lhipPitch': 0.0,
            'lknee': 0.0,
            'lankle1': 0.0,
            'lankle2': 0.0,
            # 右臂关节
            'rshoulderPitch': 0.0,
            'rshoulderRoll': 0.0,
            'rshoulderYaw': 0.0,
            'relbow': 0.0,
            # 左臂关节
            'lshoulderPitch': 0.0,
            'lshoulderRoll': 0.0,
            'lshoulderYaw': 0.0,
            'lelbow': 0.0,
        }

    class env(LeggedRobotCfg.env):
        num_dofs = 20
        num_actions = 20
        # 单步观测维度 3【基座角速度】 + 3【投影重力：机器人坐标系下的重力向量】 + 20【关节位置】 + 20【关节速度】 + 20【上一步动作】 + 1【动作缩放因子】
        num_one_step_observations = 67
        num_actor_history = 6  # 历史观测步数
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 10  # 每个episode的时长（秒）
        unactuated_timesteps = 30  # 环境启动后无动作控制的时间步数（用于稳定初始状态）

    class control(LeggedRobotCfg.control):
        # PD Drive parameters (根据BHR8FC2实际参数调整)
        control_type = 'P'
        stiffness = {
            'hip': 150,
            'knee': 200,
            'ankle': 40,
            'shoulder': 100,
            'elbow': 100,
            'waist': 100,
        }  # [N*m/rad]
        damping = {
            'hip': 4,
            'knee': 6,
            'ankle': 2,
            'shoulder': 4,
            'elbow': 4,
            'waist': 4,
        }  # [N*m*s/rad]
        # action scale: target angle = actionRescale * action + cur_dof_pos
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        # 策略网络控制频率相对于物理仿真频率的降低倍数
        decimation = 4

    class terrain:
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.7
        restitution = 0.3
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1
        num_cols = 20
        terrain_proportions = [1, 0., 0, 0, 0]
        slope_treshold = 0.75

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bhr8fc2/BHR8FC2.urdf'
        name = "bhr8fc2"
        
        left_foot_name = "left_ankle_pitch"
        right_foot_name = "right_ankle_pitch"
        left_knee_name = 'left_knee'
        right_knee_name = 'right_knee'
        foot_name = "ankle_roll"
        
        penalize_contacts_on = ["elbow", 'shoulder', 'waist', 'knee', 'hip']
        terminate_after_contacts_on = []

        left_shoulder_name = "left_shoulder"
        right_shoulder_name = "right_shoulder"

        left_leg_joints = ['lhipYaw', 'lhipRoll', 'lhipPitch', 'lknee', 'lankle1', 'lankle2']
        right_leg_joints = ['rhipYaw', 'rhipRoll', 'rhipPitch', 'rknee', 'rankle1', 'rankle2']
        left_hip_joints = ['lhipYaw']
        right_hip_joints = ['rhipYaw']
        left_hip_roll_joints = ['lhipRoll']
        right_hip_roll_joints = ['rhipRoll']
        left_hip_pitch_joints = ['lhipPitch']
        right_hip_pitch_joints = ['rhipPitch']
        left_shoulder_roll_joints = ['lshoulderRoll']
        right_shoulder_roll_joints = ['rshoulderRoll']
        left_knee_joints = ['lknee']
        right_knee_joints = ['rknee']
        left_arm_joints = ['lshoulderPitch', 'lshoulderRoll', 'lshoulderYaw', 'lelbow']
        right_arm_joints = ['rshoulderPitch', 'rshoulderRoll', 'rshoulderYaw', 'relbow']
        knee_joints = ['lknee', 'rknee']
        ankle_joints = ['lankle1', 'lankle2', 'rankle1', 'rankle2']

        keyframe_name = "keyframe"
        head_name = 'keyframe_head'
        trunk_names = ["pelvis", "torso"]
        base_name = 'torso_link'

        left_upper_body_names = ['left_shoulder_pitch', 'left_elbow']
        right_upper_body_names = ['right_shoulder_pitch', 'right_elbow']
        left_lower_body_names = ['left_hip_pitch', 'left_ankle_roll', 'left_knee']
        right_lower_body_names = ['right_hip_pitch', 'right_ankle_roll', 'right_knee']
        left_ankle_names = ['left_ankle_roll']
        right_ankle_names = ['right_ankle_roll']

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 0.75  # TODO: 根据BHR8FC2实际高度调整
        only_positive_rewards = False
        orientation_sigma = 1
        is_gaussian = True
        target_head_height = 1  # TODO: 根据实际高度调整
        target_head_margin = 1
        target_base_height_phase1 = 0.45
        target_base_height_phase2 = 0.45
        target_base_height_phase3 = 0.65
        orientation_threshold = 0.99
        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        target_dof_pos_sigma = -0.1
        tracking_sigma = 0.25

        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [2.5, 0.1, 1, 1]

        class scales(LeggedRobotCfg.rewards.scales):
            # Task rewards
            alive = 1.0
            base_height = 2.0
            pelvis_orientation = 3.0
            torso_orientation = 3.0
            head_orientation = 5.0
            head_height = 1.5
            
            # Regularization rewards
            action_rate = -0.01
            torques = -1e-5
            dof_acc = -2.5e-7
            collision = -5.0
            dof_pos_limits = -10.0
            dof_vel = -0.0
            
            # Style rewards
            target_joint_pos_upper = 0.55
            target_joint_pos_lower = 0.55
            feet_stumble = -0.0
            left_foot_displacement = 0.3
            right_foot_displacement = 0.3
            feet_displacement = 0.3
            feet_drag = 0.0
            
            # Target rewards
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

    class curriculum(LeggedRobotCfg.curriculum):
        base_height_target = 0.75  # TODO: 根据实际高度调整 (~70% of robot height)
        pull_force = True
        pull_force_value = 200  # TODO: 根据机器人重量调整 (~60% of robot weight)

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 10.


class BHR8FC2CfgPPO(LeggedRobotCfgPPO):
    """
    PPO configuration for BHR8FC2 training.
    """
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'bhr8fc2_ground'
        max_iterations = 15000
        save_interval = 500
