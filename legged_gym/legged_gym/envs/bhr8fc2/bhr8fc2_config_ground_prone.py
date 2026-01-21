from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BHR8FC2Cfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        rot = [0.0, 1.0, 0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {
            'left_hip_yaw_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': 0.2,
            'right_hip_pitch_joint': -0.2,
            'left_knee_joint': -0.4,
            'right_knee_joint': 0.4,
            'left_ankle_pitch_joint': 0.2,
            'right_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_ankle_roll_joint': 0.0,

            'left_shoulder_pitch_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': -0.2,
            'right_elbow_joint': 0.2,
        }
        target_joint_angles = {
            'left_hip_yaw_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': 0.3,
            'right_hip_pitch_joint': -0.3,
            'left_knee_joint': -0.6,
            'right_knee_joint': 0.6,
            'left_ankle_pitch_joint': 0.3,
            'right_ankle_pitch_joint': -0.3,
            'left_ankle_roll_joint': 0.0,
            'right_ankle_roll_joint': 0.0,

            'left_shoulder_pitch_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': -0.8,
            'right_elbow_joint': 0.8,
        }

    class env(LeggedRobotCfg.env):
        num_envs = 512
        num_dofs = 20
        num_actions = 20
        # 单步观测维度 3【基座角速度】 +
        #            3【投影重力：机器人坐标系下的重力向量】 +
        #            20【关节位置】 +
        #            20【关节速度】 +
        #            20【上一步动作】 +
        #            1【动作缩放因子】
        num_one_step_observations = 67
        num_actor_history = 6  # 历史观测步数
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 10  # 每个episode的时长（秒）
        unactuated_timesteps = 30  # 环境启动后无动作控制的时间步数（用于稳定初始状态）

    class control(LeggedRobotCfg.control):
        # PD Drive parameters
        control_type = 'P'
        stiffness = {
            'hip': 150,
            'knee': 200,
            'ankle': 150,
            'shoulder': 150,
            'elbow': 150,
        }  # [N*m/rad]
        damping = {
            'hip': 4,
            'knee': 4,
            'ankle': 4,
            'shoulder': 4,
            'elbow': 4,
        }  # [N*m*s/rad]
        # action scale: target angle = actionRescale * action + cur_dof_pos
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        # 策略网络控制频率相对于物理仿真频率的降低倍数
        decimation = 4

    class terrain:
        # 地形类型：'none', 'plane'(无限平面), 'heightfield'(复杂地形), 'trimesh'
        mesh_type = 'plane'
        static_friction = 0.8   # 静摩擦系数
        dynamic_friction = 0.7  # 动摩擦系数
        restitution = 0.3       # 恢复系数（0=完全非弹性，1=完全弹性碰撞）

        # ========== 以下参数仅在 heightfield/trimesh 模式下生效 ==========

        horizontal_scale = 0.1  # [m] 水平分辨率
        vertical_scale = 0.005  # [m] 垂直分辨率
        border_size = 25  # [m] 地形边界缓冲区
        # 控制地形生成的逻辑：
        # if cfg.curriculum:
        #     self.curiculum()           # 按难度递增排列
        # elif cfg.selected:
        #     self.selected_terrain()    # 使用指定的单一地形类型
        # else:
        #     self.randomized_terrain()  # 随机排列
        # 是否启用地形课程学习
        # True：地形按难度排列，效果是仅允许把机器人初始放置在简单难度地形
        # False：地形随机排列，效果是允许把机器人初始放置在任意难度地形
        curriculum = True
        # 是否使用指定的单一地形类型
        selected = False
        terrain_kwargs = None  # selected 为 True 时使用的地形参数
        max_init_terrain_level = 5  # 课程学习初始难度级别 0-5
        terrain_length = 8.0    # 每个地形块长度[m]
        terrain_width = 8.0     # 每个地形块宽度[m]
        num_rows = 1            # 地形网格行数（难度级别数）
        num_cols = 20           # 地形网格列数（地形类型数）
        terrain_proportions = [1, 0, 0, 0, 0]  # [平滑斜坡, 粗糙斜坡, 台阶, 离散障碍, 随机高度]
        # trimesh专用
        slope_treshold = 0.75  # 斜坡角度阈值，超过此值修正为垂直面

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bhr8fc2/BHR8FC2.urdf'
        name = 'bhr8fc2'

        # 惩罚和终止条件
        penalize_contacts_on = ['head', 'shoulder', 'hip']
        terminate_after_contacts_on = []

        left_leg_joints = ['left_hip_yaw_joint',
                           'left_hip_roll_joint',
                           'left_hip_pitch_joint',
                           'left_knee_joint',
                           'left_ankle_pitch_joint',
                           'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_yaw_joint',
                            'right_hip_roll_joint',
                            'right_hip_pitch_joint',
                            'right_knee_joint',
                            'right_ankle_pitch_joint',
                            'right_ankle_roll_joint']
        left_arm_joints = ['left_shoulder_pitch_joint',
                           'left_shoulder_roll_joint',
                           'left_shoulder_yaw_joint',
                           'left_elbow_joint']
        right_arm_joints = ['right_shoulder_pitch_joint',
                            'right_shoulder_roll_joint',
                            'right_shoulder_yaw_joint',
                            'right_elbow_joint']
        left_hip_joints = ['left_hip_yaw_joint']
        right_hip_joints = ['right_hip_yaw_joint']
        left_hip_roll_joints = ['left_hip_roll_joint']
        right_hip_roll_joints = ['right_hip_roll_joint']
        left_hip_pitch_joints = ['left_hip_pitch_joint']
        right_hip_pitch_joints = ['right_hip_pitch_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        left_knee_joints = ['left_knee_joint']
        right_knee_joints = ['right_knee_joint']
        ankle_joints = ['left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
        left_shoulder_roll_joints = ['left_shoulder_roll_joint']
        right_shoulder_roll_joints = ['right_shoulder_roll_joint']
        waist_joints = []  # BHR8FC2 没有腰部关节

        keyframe_name = ''  # BHR8FC2 没有专门的 keyframe links
        head_name = 'head'
        trunk_names = ['torso']
        base_name = 'torso_link'
        tracking_body_names = []

        left_thigh_name = 'left_hip_pitch'
        right_thigh_name = 'right_hip_pitch'
        left_knee_name = 'left_knee'
        right_knee_name = 'right_knee'
        left_foot_name = 'left_ankle_pitch'
        right_foot_name = 'right_ankle_pitch'
        foot_name = 'ankle_roll'
        left_shoulder_name = 'left_shoulder'
        right_shoulder_name = 'right_shoulder'

        left_upper_body_names = ['left_shoulder_pitch',
                                 'left_elbow']
        right_upper_body_names = ['right_shoulder_pitch',
                                  'right_elbow']
        left_lower_body_names = ['left_hip_pitch',
                                 'left_ankle_roll',
                                 'left_knee']
        right_lower_body_names = ['right_hip_pitch',
                                  'right_ankle_roll',
                                  'right_knee']
        left_ankle_names = ['left_ankle_roll']
        right_ankle_names = ['right_ankle_roll']

        density = 0.001         # 密度 [kg/m^3]
        angular_damping = 0.01  # 角阻尼
        linear_damping = 0.01   # 线阻尼
        max_angular_velocity = 1000.0  # 最大角速度 [rad/s]
        max_linear_velocity = 1000.0   # 最大线速度 [m/s]
        armature = 0.01       # 关节惯量补偿 [kg*m^2]
        thickness = 0.01      # 碰撞检测厚度 [m]
        self_collisions = 0   # 0：启用自碰撞，1：禁用自碰撞（可穿透）
        flip_visual_attachments = False  # 是否翻转视觉附件

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.8  # 【调整】目标质心高度
        target_head_height = 1.2  # 【调整】目标头部高度
        target_head_margin = 1.2  # 目标头部高度容差范围
        target_base_height_phase1 = 0.5
        target_base_height_phase2 = 0.5
        target_base_height_phase3 = 0.7

        base_height_sigma = 0.25
        tracking_dof_sigma = 0.25

        soft_dof_pos_limit = 0.9  # 软关节位置限制（安全范围比例）
        soft_dof_vel_limit = 0.9  # 软关节速度限制（安全范围比例）
        only_positive_rewards = False  # 是否只计算正奖励
        orientation_sigma = 1  # 姿态奖励的敏感度（越小越严格）
        orientation_threshold = 0.99  # 姿态奖励阈值（姿态向量点积要大于 0.99）
        is_gaussian = True  # 是否使用高斯函数计算奖励
        # 脚的位置与参考轨迹的误差敏感度（模仿学习用）
        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        target_dof_pos_sigma = -0.1  # 关节位置目标奖励敏感度
        tracking_sigma = 0.25  # 速度追踪奖励敏感度

        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [1, 0.1, 1, 1]

        class scales:
            task_orientation = 1
            task_head_height = 1

    class constraints( LeggedRobotCfg.rewards ):
        is_gaussian = True
        target_head_height = 1.2
        target_head_margin = 1.2
        orientation_height_threshold = 0.9
        target_base_height = 0.5

        left_foot_displacement_sigma = -2
        right_foot_displacement_sigma = -2
        hip_yaw_var_sigma = -2
        target_dof_pos_sigma = -0.1
        post_task = False
        
        class scales:
            # regularization reward
            regu_dof_acc = -2.5e-7
            regu_action_rate = -0.01
            regu_smoothness = -0.01
            regu_torques = -2.5e-6
            regu_joint_power = -2.5e-5
            regu_dof_vel = -1e-3
            regu_joint_tracking_error = -0.00025
            regu_dof_pos_limits = -100.0
            regu_dof_vel_limits = -1

            # style reward
            style_waist_deviation = 0  # BHR8FC2没有腰部关节，禁用
            style_hip_yaw_deviation = -10
            style_hip_roll_deviation = -10
            style_hip_pitch_deviation = -10
            style_shoulder_roll_deviation = -2.5
            style_left_foot_displacement = 2.5
            style_right_foot_displacement = 2.5
            style_knee_deviation = -0.25
            style_thigh_ori = 10
            style_feet_distance = -10
            style_style_ang_vel_xy = 25

            # post-task reward
            target_ang_vel_xy = 10
            target_lin_vel_xy = 10
            target_feet_height_var = 2.5
            target_target_upper_dof_pos = 10
            target_lower_body_deviation = 10
            target_target_orientation = 10
            target_target_base_height = 10

    class domain_rand:
        use_random = True

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        randomize_payload_mass = use_random
        payload_mass_range = [-2, 5]

        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]

        randomize_friction = use_random
        friction_range = [0.1, 1]

        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]

        randomize_kp = use_random
        kp_range = [0.85, 1.15]

        randomize_kd = use_random
        kd_range = [0.85, 1.15]

        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.1, 0.1]

        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 0.5

        delay = use_random
        max_delay_timesteps = 5

    class curriculum:
        pull_force = True
        force = 350
        dof_vel_limit = 300
        base_vel_limit = 20
        threshold_height = 0.9
        no_orientation = True

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class BHR8FC2CfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256]

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = 'bhr8fc2_ground_prone'
        algorithm_class_name = 'PPO'
        init_at_random_ep_len = True
        max_iterations = 12000  # number of policy updates
