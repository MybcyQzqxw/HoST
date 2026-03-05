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
            'left_hip_pitch_joint': 0.15,
            'right_hip_pitch_joint': -0.15,
            'left_knee_joint': -0.4,
            'right_knee_joint': 0.4,
            'left_ankle_pitch_joint': 0.25,
            'right_ankle_pitch_joint': -0.25,
            'left_ankle_roll_joint': 0.0,
            'right_ankle_roll_joint': 0.0,

            'left_shoulder_pitch_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': -0.6,
            'right_elbow_joint': 0.6,
        }
        target_joint_angles = {
            'left_hip_yaw_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': 0.15,
            'right_hip_pitch_joint': -0.15,
            'left_knee_joint': -0.4,
            'right_knee_joint': 0.4,
            'left_ankle_pitch_joint': 0.25,
            'right_ankle_pitch_joint': -0.25,
            'left_ankle_roll_joint': 0.0,
            'right_ankle_roll_joint': 0.0,

            'left_shoulder_pitch_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.2,
            'right_shoulder_roll_joint': -0.2,
            'left_shoulder_yaw_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': -0.4,
            'right_elbow_joint': 0.4,
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
        episode_length_s = 20  # 每个episode的时长（秒）
        unactuated_timesteps = 30  # 环境启动后无动作控制的时间步数（用于稳定初始状态）

    class control(LeggedRobotCfg.control):
        # PD Drive parameters
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {
            'hip': 150,
            'knee': 200,
            'ankle': 150,
            'shoulder': 150,
            'elbow': 150,
        }  # [N*m/rad]
        damping = {
            'hip': 2.5,
            'knee': 2.5,
            'ankle': 2.5,
            'shoulder': 2.5,
            'elbow': 2.5,
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

        base_name = 'torso'
        head_name = 'head'
        foot_name = 'ankle_roll'
        keyframe_name = ''  # BHR8FC2 没有专门的 keyframe links
        left_shoulder_name = 'left_shoulder'
        right_shoulder_name = 'right_shoulder'
        left_shoulder_pitch_name = 'left_shoulder_pitch'
        right_shoulder_pitch_name = 'right_shoulder_pitch'
        left_shoulder_roll_name = 'left_shoulder_roll'
        right_shoulder_roll_name = 'right_shoulder_roll'
        left_shoulder_yaw_name = 'left_shoulder_yaw'
        right_shoulder_yaw_name = 'right_shoulder_yaw'
        left_elbow_name = 'left_elbow'
        right_elbow_name = 'right_elbow'
        left_hip_yaw_name = 'left_hip_yaw'
        right_hip_yaw_name = 'right_hip_yaw'
        left_hip_roll_name = 'left_hip_roll'
        right_hip_roll_name = 'right_hip_roll'
        left_thigh_name = 'left_hip_pitch'
        right_thigh_name = 'right_hip_pitch'
        left_knee_name = 'left_knee'
        right_knee_name = 'right_knee'
        left_ankle_pitch_name = 'left_ankle_pitch'
        right_ankle_pitch_name = 'right_ankle_pitch'
        left_foot_name = 'left_ankle_roll'
        right_foot_name = 'right_ankle_roll'

        left_shoulder_pitch_joints = ['left_shoulder_pitch_joint']
        right_shoulder_pitch_joints = ['right_shoulder_pitch_joint']
        left_shoulder_roll_joints = ['left_shoulder_roll_joint']
        right_shoulder_roll_joints = ['right_shoulder_roll_joint']
        left_shoulder_yaw_joints = ['left_shoulder_yaw_joint']
        right_shoulder_yaw_joints = ['right_shoulder_yaw_joint']
        left_elbow_joints = ['left_elbow_joint']
        right_elbow_joints = ['right_elbow_joint']
        waist_joints = []  # BHR8FC2 没有腰部关节
        left_hip_yaw_joints = ['left_hip_yaw_joint']
        right_hip_yaw_joints = ['right_hip_yaw_joint']
        left_hip_roll_joints = ['left_hip_roll_joint']
        right_hip_roll_joints = ['right_hip_roll_joint']
        left_hip_pitch_joints = ['left_hip_pitch_joint']
        right_hip_pitch_joints = ['right_hip_pitch_joint']
        left_knee_joints = ['left_knee_joint']
        right_knee_joints = ['right_knee_joint']
        left_ankle_pitch_joints = ['left_ankle_pitch_joint']
        right_ankle_pitch_joints = ['right_ankle_pitch_joint']
        left_ankle_roll_joints = ['left_ankle_roll_joint']
        right_ankle_roll_joints = ['right_ankle_roll_joint']

    class rewards(LeggedRobotCfg.rewards):
        is_gaussian = True  # 是否使用高斯函数计算奖励
        only_positive_rewards = False  # 是否只计算正奖励
        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [1, 0.01, 1, 1]

        target_base_height_phase1 = 0.45
        target_base_height_phase2 = 0.65

        # task reward
        target_base_height = 0.80
        target_head_height = 1.30
        orientation_threshold = 0.99

        class scales:
            # task_base_height = 1
            task_head_height = 1
            task_orientation = 1

    class constraints(LeggedRobotCfg.rewards):
        is_gaussian = True  # 是否使用高斯函数计算奖励
        only_positive_rewards = False  # 是否只计算正奖励

        # style reward (单位: 角度)
        # off_center  a * 总值
        # inside      a * 内值
        shoulder_roll_deviation_off_center_threshold = 40
        shoulder_roll_deviation_inside_threshold = 4
        shoulder_yaw_deviation_off_center_threshold = 60
        shoulder_yaw_deviation_inside_threshold = 20
        waist_deviation_threshold = 80
        hip_yaw_deviation_off_center_threshold = 20
        hip_yaw_deviation_inside_threshold = 10
        hip_roll_deviation_off_center_threshold = 30
        hip_roll_deviation_inside_threshold = 15
        ankle_roll_deviation_off_center_threshold = 20
        ankle_roll_deviation_inside_threshold = 10

        import math
        shoulder_roll_deviation_off_center_threshold = math.radians(shoulder_roll_deviation_off_center_threshold)
        shoulder_roll_deviation_inside_threshold = math.radians(shoulder_roll_deviation_inside_threshold)
        shoulder_yaw_deviation_off_center_threshold = math.radians(shoulder_yaw_deviation_off_center_threshold)
        shoulder_yaw_deviation_inside_threshold = math.radians(shoulder_yaw_deviation_inside_threshold)
        waist_deviation_threshold = math.radians(waist_deviation_threshold)
        hip_yaw_deviation_off_center_threshold = math.radians(hip_yaw_deviation_off_center_threshold)
        hip_yaw_deviation_inside_threshold = math.radians(hip_yaw_deviation_inside_threshold)
        hip_roll_deviation_off_center_threshold = math.radians(hip_roll_deviation_off_center_threshold)
        hip_roll_deviation_inside_threshold = math.radians(hip_roll_deviation_inside_threshold)
        ankle_roll_deviation_off_center_threshold = math.radians(ankle_roll_deviation_off_center_threshold)
        ankle_roll_deviation_inside_threshold = math.radians(ankle_roll_deviation_inside_threshold)

        no_supine_threshold = -0.1
        feet_distance_threshold = 0.8
        upper_body_deviation_sigma = -2
        lower_body_deviation_sigma = -2

        # ----- phase related
        # ---------- before1
        before1_prone_orientation_threshold = 0.7
        before1_thigh_ori_threshold = 0.8
        before1_shank_ori_threshold = 0.2
        # ---------- after1
        after1_thigh_ori_threshold = 0.8
        after1_shank_ori_threshold = 0.8
        # ---------- before2
        before2_base_ang_vel_x_sigma = -2
        before2_base_lin_vel_y_sigma = -5
        # ---------- after1_before2
        # ---------- after2
        after2_base_ang_vel_xy_sigma = -2
        after2_base_lin_vel_xy_sigma = -5
        after2_upper_body_deviation_sigma = -2
        after2_lower_body_deviation_sigma = -2
        after2_arm_pos_sigma = -0.1
        after2_leg_ori_threshold = 0.8
        after2_feet_distance_threshold = 0.4
        after2_feet_height_var_sigma = -2
        after2_left_foot_displacement_sigma = -2
        after2_right_foot_displacement_sigma = -2

        # target reward
        target_base_height_sigma = -20
        target_orientation_sigma = -5

        class scales:
            # regularization reward
            regu_dof_acc = -2.5e-7
            regu_dof_vel = -1e-3
            regu_action_rate = -1e-5
            regu_smoothness = -2.5e-6
            regu_torques = -1e-5
            regu_joint_power = -1e-4
            regu_dof_pos_limits = -10
            regu_dof_vel_limits = -1
            regu_torque_limits = 0

            # style reward
            style_shoulder_roll_deviation = -2.5
            style_shoulder_yaw_deviation = -2.5
            style_waist_deviation = 0  # BHR8FC2没有腰部关节，禁用
            style_hip_yaw_deviation = -2.5
            style_hip_roll_deviation = -2.5
            style_ankle_roll_deviation = -2.5

            style_no_head_contact = -20
            style_no_shoulder_contact = -2.5
            style_no_bigarm_contact = -1
            style_no_torso_contact = -2.5
            style_no_hip_contact = -2.5
            style_no_thigh_contact = -1
            style_no_supine = -20
            style_tripod_contact = -20
            style_lower_body_contact = -20
            style_feet_distance = -20
            style_upper_body_deviation = 2.5
            style_lower_body_deviation = 2.5
            # ----- phase related
            # ---------- before1
            style_before1_prone_orientation = 0
            style_before1_forearm_contact = 10
            style_before1_knee_contact = 10
            style_before1_foot_contact = 2.5
            style_before1_thigh_ori = 20
            style_before1_shank_ori = -20
            # ---------- after1
            style_after1_no_torso_above_head = -20
            style_after1_no_torso_below_leg = -20
            style_after1_thigh_ori = 20
            style_after1_shank_ori = 20
            # ---------- after1_before2
            style_after1_before2_base_ang_vel_y = 0
            # ---------- before2
            style_before2_base_ang_vel_x = 10
            style_before2_base_lin_vel_y = 10
            # ---------- after2
            style_after2_base_ang_vel_xy = 10
            style_after2_base_lin_vel_xy = 10
            style_after2_upper_body_deviation = 0
            style_after2_lower_body_deviation = 0
            style_after2_arm_pos = 10
            style_after2_leg_ori = 0
            style_after2_feet_distance = 0
            style_after2_feet_height_var = 2.5
            style_after2_left_foot_displacement = 2.5
            style_after2_right_foot_displacement = 2.5
            style_after2_no_forearm_contact = -20
            style_after2_no_knee_contact = -20
            style_after2_foot_contact = -20

            # target reward
            target_target_base_height = 20
            target_target_orientation = 20

    class domain_rand:
        use_random = True

        # _create_envs 中初始化下面 5 个
        # 负载质量
        randomize_payload_mass = use_random
        payload_mass_range = [-2, 5]
        # 质心偏移
        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]
        # 摩擦系数
        randomize_friction = use_random
        friction_range = [0.1, 1]
        # 恢复系数
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        # 连杆质量
        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]

        # _init_buffers 中初始化下面 4 个
        # kp
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        # kd
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        # 驱动偏置
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]
        # 电机力矩
        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        # 关节位置和速度初始化 _reset_dofs 中使用
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.1, 0.1]

        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 0.5

        delay = use_random
        max_delay_timesteps = 5

    class curriculum:
        # 施加向上拉力
        pull_force = True
        force = 400
        no_orientation = True  # 所有姿态都施加力

        # 增加难度的高度阈值
        threshold_head_height = 1.00

    class limitation:
        # 关节和基座速度限制
        dof_vel_limit = 300
        base_vel_limit = 20
        soft_dof_pos_limit = 0.9  # 软关节位置限制（安全范围比例）
        soft_dof_vel_limit = 0.9  # 软关节速度限制（安全范围比例）
        soft_torque_limit = 0.9   # 软关节力矩限制（安全范围比例）

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

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = 'bhr8fc2_ground_prone'
        algorithm_class_name = 'PPO'
        init_at_random_ep_len = True
        max_iterations = 12000  # number of policy updates
