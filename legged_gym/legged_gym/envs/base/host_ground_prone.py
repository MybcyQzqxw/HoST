from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

from legged_gym.utils.motion_lib import (
    MotionLib,
    load_imitation_dataset,
    compute_residual_observations,
)
from legged_gym.utils.math import (
    tolerance,
    torch_rand_float,
)


class LeggedRobot(BaseTask):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False  # 初始化flag
        self._parse_cfg(self.cfg)
        self.num_real_dofs = cfg.env.num_dofs

        # 初始化中包含一句 self.create_sim()
        # self.create_sim() 中包含 self._create_envs()
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # 单步观测维度
        self.num_one_step_obs = self.cfg.env.num_one_step_observations  # if not self.cfg.env.add_force else self.cfg.env.num_one_step_observations + 1
        # 历史观测长度
        self.actor_history_length = self.cfg.env.num_actor_history
        # 总观测数 = 单步观测维度 * 历史观测长度
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True  # 初始化flag
        self.unactuated_time = self.cfg.env.unactuated_timesteps
        self.unactuated_time *= 0.02 / self.dt  # 保证实际时间一致（30个步长始终等于0.6s）
        self.is_gaussian = cfg.rewards.is_gaussian

    def step(self, actions):
        """Execute one simulation step with the given actions.

        Args:
            actions: Action commands for the robot
        Returns:
            Tuple of (observations, privileged_observations, rewards, dones, extras)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.actions *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.cfg.curriculum.pull_force:
                force_tensor = torch.zeros([self.num_envs, self.num_bodies, 3], device=self.device)
                force_tensor[:, self.base_indices, 2] = self.force
                force_tensor *= (self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time).unsqueeze(1)
                if not self.cfg.curriculum.no_orientation:
                    force_tensor *= (self.projected_gravity[:, 2] < -0.8).unsqueeze(1).unsqueeze(1)
                force_tensor = gymtorch.unwrap_tensor(force_tensor)
                self.gym.apply_rigid_body_force_tensors(self.sim, force_tensor)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.real_episode_length_buf += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.init_rpy = None
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # self.compute_motions()
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_last_dof_pos[:] = self.last_dof_pos[:]
        self.last_dof_pos[:] = self.dof_pos[:]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        self.dof_vel_out = (torch.abs(self.dof_vel.max(dim=1).values) > self.cfg.curriculum.dof_vel_limit) & (self.real_episode_length_buf > self.unactuated_time)
        self.reset_buf |= self.dof_vel_out

        self.base_vel_out = (torch.norm(self.base_lin_vel[:, :3], dim=-1) > self.cfg.curriculum.base_vel_limit) & (self.real_episode_length_buf > self.unactuated_time)
        self.reset_buf |= self.base_vel_out

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
            
        self.extras["episode"] = {}
        self.extras['episode']['base_height'] = self.old_headheight[env_ids].mean()

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.update_force_curriculum(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_last_dof_pos[env_ids] = 0
        self.last_dof_pos[env_ids] = 0
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.real_episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.old_headheight[env_ids] = 0
        self.max_headheight[env_ids] = 0
        self.feet_ori[env_ids] = 0
        # fill extras
        self.delay_buffer[:, env_ids, :] = 0.

        for key in self.episode_sums.keys():
            self.extras['episode']['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras['episode']['max_command_x'] = self.command_ranges['lin_vel_x'][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras['time_outs'] = self.time_out_buf

        # self._reset_motions(env_ids)
        self.extras['episode']['force'] = self.force.mean()
        self.extras['episode']['action_scale'] = self.action_rescale

        # reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_real_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_real_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_real_dofs), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), self.num_real_dofs), device=self.device)
        if self.cfg.domain_rand.delay:
            self.delay_idx[env_ids] = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(len(env_ids), ), device=self.device)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
            
        【组合策略】：
        1. rewards (task类): 先计算所有 reward 项，用【乘法】组合成 task 奖励
           → 确保所有任务目标同时满足（AND逻辑）
           
        2. constraints: 计算所有 constraint 项，用【加法】累积惩罚
           → 每个约束独立惩罚，可以部分违反（独立惩罚）
           
        3. 最终总奖励 = task奖励（乘法） + 各类约束（加法）
           → 平衡任务完成度与约束满足度
        """
        if not self.is_gaussian:
            raise NotImplementedError
        else:
            self.rew_buf[:, :] = 0
            task_group_index = self.reward_groups.index('task')
            self.rew_buf[:, task_group_index] = 1
            
            # 【乘法组合】所有 rewards (task类) 项
            for i in range(len(self.reward_functions)):
                name = self.reward_names[i]
                rew = self.reward_functions[i]() * self.reward_scales[name]
                if len(rew.shape) == 2 and rew.shape[1] == 1:
                    rew = rew.squeeze(1)
                # print(name, rew.shape)
                self.rew_buf[:, task_group_index] *= rew  # ← 注意这里是乘法！
                self.episode_sums[name] += rew

        # 【加法累积】所有 constraints 惩罚项
        for i in range(len(self.constraint_functions)):
            name = self.constraint_names[i]
            reward_group_name = name.split('_')[0]  # 提取前缀: regu/style/target
            rew = self.constraint_functions[i]() * self.constraint_scales[name]
            task_group_index = self.reward_groups.index(reward_group_name)

            self.rew_buf[:, task_group_index] += rew  # ← 注意这里是加法！
            self.episode_sums[name] += rew
            if self.cfg.constraints.only_positive_rewards:
                self.rew_buf[:, task_group_index] = torch.clip(self.rew_buf[:, task_group_index], min=0.)

        if "termination" in self.constraint_scales:
            rew = self._reward_termination() * self.constraint_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        for rg in self.reward_groups:
            idx = self.reward_groups.index(rg)
            self.episode_sums[rg] = self.rew_buf[:, idx]

    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat((
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.dof_pos * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.action_rescale + (torch.rand_like(self.action_rescale) - 0.5) * 0.05,
                                    ),dim=-1)
            
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec

        current_obs *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_obs), dim=-1)

    def compute_motions(self):
        """Compute and update motion states for the robot."""
        # resample motions
        timeout = self.motions.check_timeout(self.motion_ids, self.motion_time)
        num_timeout = timeout.long().sum().item()
        if num_timeout > 0:
            motion_ids = self.motions.sample_motions(num_timeout)
            motion_time = self.motions.sample_time(motion_ids, uniform=True)
            self.motion_ids[timeout], self.motion_time[timeout] = motion_ids, motion_time
            self.recovery_mask[timeout], self.recovery_init_time[timeout] = True, motion_time
            self.init_base_pos_xy[timeout], self.init_base_quat[timeout] = self.base_pos[timeout, 0:2], self.base_quat[timeout]
        self.motion_dict = self.motions.get_motion_state(self.motion_ids, self.motion_time, self.init_base_pos_xy, self.init_base_quat)

    def create_sim(self):
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()  # 未定义
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError('Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]')
        self._create_envs()

    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order='C'),
            self.terrain.triangles.flatten(order='C'),
            tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ---------- Callbacks
    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs, 1), device=self.device)
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs, 1), device=self.device)
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_real_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props['lower'][i].item()
                self.dof_pos_limits[i, 1] = props['upper'][i].item()
                self.dof_vel_limits[i] = props['velocity'][i].item()
                self.torque_limits[i] = props['effort'][i].item()
                # soft limits
                # self.dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0] * self.cfg.rewards.soft_dof_pos_limit
                # self.dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1] * self.cfg.rewards.soft_dof_pos_limit
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_link_index].mass = self.default_rigid_body_mass[self.torso_link_index] + self.payload[env_id, 0]

        if self.cfg.domain_rand.randomize_com_displacement:
            props[self.torso_link_index].com = self.default_com_torso + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(0, len(props)):
                if i == self.torso_link_index:
                    pass
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.action_rescale
    
        self.joint_pos_target = self.dof_pos + actions_scaled
        if self.cfg.domain_rand.delay:
            self.delay_buffer = torch.concat((self.delay_buffer[1:], actions_scaled.unsqueeze(0)), dim=0)
            self.joint_pos_target = self.dof_pos + self.delay_buffer[self.delay_idx, torch.arange(len(self.delay_idx)), :]
        else:
            self.joint_pos_target = self.dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains *  self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        torques = self.motor_strength *  torques + self.actuation_offset
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_real_dofs), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_real_dofs), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_real_dofs), device=self.device) + int(self.cfg.domain_rand.random_pose) * torch_rand_float(-1, 1, (len(env_ids), self.num_real_dofs), device=self.device) 
            self.dof_vel[env_ids] = 0.

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:10] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _pull_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_z
        self.root_states[:, 9] = torch_rand_float(0, max_vel, (self.num_envs,), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
   
    def update_force_curriculum(self, env_ids):
        """Update force curriculum based on performance.
        
        Args:
            env_ids: Environment IDs to update
        """
        if torch.mean(self.old_headheight[env_ids]) > self.cfg.curriculum.threshold_height:
            self.force[env_ids] = (self.force[env_ids] - 20).clamp(0, np.inf)
            self.action_rescale[env_ids] = (self.action_rescale[env_ids] - 0.02).clamp(0.25, np.inf)

    def _get_noise_scale_vec(self, cfg):
        # 构造一个与单步观测 num_one_step_obs 同长度的噪声缩放向量 noise_vec
        # 把统一分布噪声（在 [-1,1]）按每个观测维度的尺度放大后加到观测上
        start_index = 6
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # 基座角速度
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # 重力向量
        noise_vec[3:6] = noise_scales.gravity * noise_level
        # 各关节位置
        noise_vec[start_index:start_index + self.num_real_dofs] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # 各关节速度
        noise_vec[start_index + self.num_real_dofs:start_index + 2 * self.num_real_dofs] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # 各关节力矩
        noise_vec[start_index + 2 * self.num_real_dofs:start_index + 3 * self.num_real_dofs] = 0.0
        return noise_vec

    def _init_buffers(self):
        # ---------- get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # ---------- create some wrapper tensors for different slices

        # root_states (num_envs, 13)
        # [:,0:3]：位置 px,py,pz
        # [:,3:7]：四元数 qx,qy,qz,qw（orientation）
        # [:,7:10]：线速度 vx,vy,vz
        # [:,10:13]：角速度 wx,wy,wz
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.base_quat = self.root_states[:self.num_envs, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        # dof_state (num_envs, num_dof, 2)
        # [...,0]：关节位置（position）
        # [...,1]：关节速度（velocity）
        self.dof_state = gymtorch.wrap_tensor(dof_state)
        self.dof_states = self.dof_state.view(self.num_envs, self.num_real_dofs, 2)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_real_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_real_dofs, 2)[..., 1]

        # rigid_body_states (num_envs, num_bodies, 13)
        # [...,0:3]：位置 px,py,pz
        # [...,3:7]：四元数 qx,qy,qz,qw（orientation）
        # [...,7:10]：线速度 vx,vy,vz
        # [...,10:13]：角速度 wx,wy,wz
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        # contact_forces (num_envs, num_bodies, xyz axis)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_force).view(self.num_envs, -1, 3)

        # ---------- initialize some data used later on

        # 仿真步数计数器
        self.common_step_counter = 0
        # 额外信息
        self.extras = {}
        # 噪声
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # 重力方向
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # 前进方向
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

        # 基座坐标系下的线速度、角速度、重力向量
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 各关节力矩
        self.torques = torch.zeros(self.num_envs, self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        # PD 增益
        self.p_gains = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        # 动作 actions（当前、上一个、上上个）
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:self.num_envs, 7:13])

        # 指令 commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)

        # rewards 相关
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.old_headheight = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.max_headheight = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.feet_ori = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)

        # 向上拉力
        self.force = self.cfg.curriculum.force * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        # 动作缩放
        self.action_rescale = self.cfg.control.action_scale * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        # 模拟延迟
        self.delay_buffer = torch.zeros(self.cfg.domain_rand.max_delay_timesteps, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # 关节位置
        self.default_dof_pos = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_dof_pos = torch.zeros(self.num_envs, self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_real_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            self.target_dof_pos[:, i] = self.cfg.init_state.target_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ['P', 'V']:
                    print(f'PD gain of joint {name} were not defined, setting them to zero')
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # 随机化 kp、kd、actuation_offset、motor_strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength = torch.ones(self.num_envs, self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_real_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_real_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_real_dofs), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_real_dofs), device=self.device)
        if self.cfg.domain_rand.delay:
            self.delay_idx = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(self.num_envs,), device=self.device)

    def _prepare_reward_function(self):
        """准备奖励函数，筛选并初始化 rewards 和 constraints

        【关键区别】rewards vs constraints：
        1. **数学组合方式不同**：
           - rewards (task类): 使用【乘法】组合 → self.rew_buf *= rew
             所有 reward 项相乘，形成一个综合奖励
             例如: task_reward = orientation × head_height × ...
           - constraints: 使用【加法】组合 → self.rew_buf += rew
             所有 constraint 项相加，各自独立惩罚
             例如: total = task_reward + torques_penalty + collision_penalty + ...
        2. **时间缩放不同**：
           - rewards: 权重不变 (× 1)
           - constraints: 权重乘以 dt (× 0.02)，使惩罚与控制频率匹配
        3. **命名约定不同**：
           - rewards: 名称如 "task_xxx" 表示任务奖励
           - constraints: 名称如 "regu_xxx", "style_xxx", "target_xxx" 表示约束类型
        4. **设计目的不同**：
           - rewards: 核心任务目标（如站起来、保持平衡）
           - constraints: 软约束惩罚（如力矩限制、碰撞惩罚、风格要求）
        实际效果：
        - task类奖励项互相关联（一个失败则整体失败）→ 用乘法
        - 约束类惩罚项独立累积（每个独立惩罚）→ 用加法
        """
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= 1  # rewards 权重保持不变

        for key in list(self.constraint_scales.keys()):
            scale = self.constraint_scales[key]
            if scale == 0:
                self.constraint_scales.pop(key)
            else:
                self.constraint_scales[key] *= self.dt  # constraints 权重乘以时间步长

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == 'termination':
                continue
            self.reward_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.reward_functions.append(getattr(self, name))

        self.constraint_functions = []
        self.constraint_names = []
        for name, scale in self.constraint_scales.items():
            self.constraint_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.constraint_functions.append(getattr(self, name))

        self.episode_sums = {}
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        for name in self.constraint_scales.keys():
            self.episode_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        # Creates environments:
        # 1. loads the robot URDF/MJCF asset
        # 2. For each environment
        #     2.1 creates the environment
        #     2.2 calls DOF and Rigid shape properties callbacks
        #     2.3 create actor with these properties and add them to the env
        # 3. Store indices of different bodies of the robot
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)  # 被下面覆盖了
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s and 'auxiliary' not in s]

        # 惩罚和终止条件
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        self.torso_link_index = body_names.index('torso_link')

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.envs = []
        self.actor_handles = []

        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            # xyz 方向上放大倍数
            self.com_displacement[:, 0] = self.com_displacement[:, 0] * 4
            self.com_displacement[:, 1] = self.com_displacement[:, 1] * 4
            self.com_displacement[:, 2] = self.com_displacement[:, 2] * 2

        for i in range(self.num_envs):
            # env handle 创建
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # actor 初始位置随机化
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # 摩擦系数和恢复系数随机化
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)  # 【1】

            # actor handle 创建
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)

            # 关节属性处理
            dof_props = self._process_dof_props(dof_props_asset, i)

            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)  # 【2】

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            if i == 0:
                self.default_com_torso = copy.deepcopy(body_props[self.torso_link_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass

            # 负载质量、质心偏移、连杆质量随机化
            body_props = self._process_rigid_body_props(body_props, i)

            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)  # 【3】

            # 存储在 envs 和 actor_handles 中
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 下面开始录入各个连杆的索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        base_name = [s for s in body_names if self.cfg.asset.base_name in s]
        self.base_indices = torch.zeros(len(base_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(base_name)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name[i])

        self.head_names = [s for s in body_names if self.cfg.asset.head_name in s]
        self.head_indices = torch.zeros(len(self.head_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.head_names):
            self.head_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.keyframe_names = [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.zeros(len(self.keyframe_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.keyframe_names):
            self.keyframe_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        left_shoulder_names = [s for s in body_names if self.cfg.asset.left_shoulder_name in s and 'keyframe' not in s]
        right_shoulder_names = [s for s in body_names if self.cfg.asset.right_shoulder_name in s and 'keyframe' not in s]
        self.left_shoulder_indices = torch.zeros(len(left_shoulder_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_shoulder_names)):
            self.left_shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_shoulder_names[i])
        self.right_shoulder_indices = torch.zeros(len(right_shoulder_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_shoulder_names)):
            self.right_shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_shoulder_names[i])

        left_thigh_names = [s for s in body_names if self.cfg.asset.left_thigh_name in s and 'keyframe' not in s]
        right_thigh_names = [s for s in body_names if self.cfg.asset.right_thigh_name in s and 'keyframe' not in s]
        self.left_thigh_indices = torch.zeros(len(left_thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_thigh_names)):
            self.left_thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_thigh_names[i])
        self.right_thigh_indices = torch.zeros(len(right_thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_thigh_names)):
            self.right_thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_thigh_names[i])

        left_knee_names = [s for s in body_names if self.cfg.asset.left_knee_name in s and 'keyframe' not in s]
        right_knee_names = [s for s in body_names if self.cfg.asset.right_knee_name in s and 'keyframe' not in s]
        self.left_knee_indices = torch.zeros(len(left_knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_knee_names)):
            self.left_knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_knee_names[i])
        self.right_knee_indices = torch.zeros(len(right_knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_knee_names)):
            self.right_knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_knee_names[i])

        left_foot_names = [s for s in body_names if self.cfg.asset.left_foot_name in s and 'keyframe' not in s and 'auxiliary' not in s]
        right_foot_names = [s for s in body_names if self.cfg.asset.right_foot_name in s and 'keyframe' not in s and 'auxiliary' not in s]
        self.left_foot_indices = torch.zeros(len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_foot_names[i])
        self.right_foot_indices = torch.zeros(len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_foot_names[i])

        # 下面开始录入各个关节的索引
        self.left_shoulder_roll_joint_indices = torch.zeros(len(self.cfg.asset.left_shoulder_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_shoulder_roll_joints)):
            self.left_shoulder_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_shoulder_roll_joints[i])
        self.right_shoulder_roll_joint_indices = torch.zeros(len(self.cfg.asset.right_shoulder_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_shoulder_roll_joints)):
            self.right_shoulder_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_shoulder_roll_joints[i])
        self.shoulder_roll_joint_indices = torch.cat((self.left_shoulder_roll_joint_indices, self.right_shoulder_roll_joint_indices))

        self.waist_joint_indices = torch.zeros(len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_joints)):
            self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])

        self.left_hip_yaw_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_yaw_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_yaw_joints)):
            self.left_hip_yaw_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_yaw_joints[i])
        self.right_hip_yaw_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_yaw_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_yaw_joints)):
            self.right_hip_yaw_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_yaw_joints[i])
        self.hip_yaw_joint_indices = torch.cat((self.left_hip_yaw_joint_indices, self.right_hip_yaw_joint_indices))

        self.left_hip_roll_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_roll_joints)):
            self.left_hip_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_roll_joints[i])
        self.right_hip_roll_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_roll_joints)):
            self.right_hip_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_roll_joints[i])
        self.hip_roll_joint_indices = torch.cat((self.left_hip_roll_joint_indices, self.right_hip_roll_joint_indices))

        self.left_hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_pitch_joints)):
            self.left_hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_pitch_joints[i])
        self.right_hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_pitch_joints)):
            self.right_hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_pitch_joints[i])
        self.hip_pitch_joint_indices = torch.cat((self.left_hip_pitch_joint_indices, self.right_hip_pitch_joint_indices))

        self.all_hip_joint_indices = torch.cat([self.hip_yaw_joint_indices, self.hip_roll_joint_indices, self.hip_pitch_joint_indices])

        self.left_knee_joint_indices = torch.zeros(len(self.cfg.asset.left_knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_knee_joints)):
            self.left_knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_knee_joints[i])
        self.right_knee_joint_indices = torch.zeros(len(self.cfg.asset.right_knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_knee_joints)):
            self.right_knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_knee_joints[i])

        self.knee_joint_indices = torch.zeros(len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])

        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])

        self.left_arm_joint_indices = torch.zeros(len(self.cfg.asset.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_arm_joints[i])
        self.right_arm_joint_indices = torch.zeros(len(self.cfg.asset.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_arm_joints[i])

        self.upper_body_joint_indices = torch.cat([self.right_arm_joint_indices, self.left_arm_joint_indices, self.waist_joint_indices])
        self.lower_body_joint_indices = torch.cat([self.all_hip_joint_indices, self.knee_joint_indices, self.ankle_joint_indices])

        tracking_body_names = []
        for target_name in self.cfg.asset.tracking_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    tracking_body_names.append(source_name)
        self.tracking_body_indices = torch.zeros(len(tracking_body_names), dtype=torch.long, device=self.device)
        self.tracking_body_names = tracking_body_names
        for i, name in enumerate(tracking_body_names):
            self.tracking_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        left_upper_body_names = []
        for target_name in self.cfg.asset.left_upper_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    left_upper_body_names.append(source_name)
        self.left_upper_body_indices = torch.zeros(len(left_upper_body_names), dtype=torch.long, device=self.device)
        self.left_upper_body_names = left_upper_body_names
        for i, name in enumerate(left_upper_body_names):
            self.left_upper_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        right_upper_body_names = []
        for target_name in self.cfg.asset.right_upper_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    right_upper_body_names.append(source_name)
        self.right_upper_body_indices = torch.zeros(len(right_upper_body_names), dtype=torch.long, device=self.device)
        self.right_upper_body_names = right_upper_body_names
        for i, name in enumerate(right_upper_body_names):
            self.right_upper_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        left_lower_body_names = []
        for target_name in self.cfg.asset.left_lower_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    left_lower_body_names.append(source_name)
        self.left_lower_body_indices = torch.zeros(len(left_lower_body_names), dtype=torch.long, device=self.device)
        self.left_lower_body_names = left_lower_body_names
        for i, name in enumerate(left_lower_body_names):
            self.left_lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        right_lower_body_names = []
        for target_name in self.cfg.asset.right_lower_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    right_lower_body_names.append(source_name)
        self.right_lower_body_indices = torch.zeros(len(right_lower_body_names), dtype=torch.long, device=self.device)
        self.right_lower_body_names = right_lower_body_names
        for i, name in enumerate(right_lower_body_names):
            self.right_lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        left_ankle_names = []
        for target_name in self.cfg.asset.left_ankle_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name:
                    left_ankle_names.append(source_name)
        self.left_ankle_indices = torch.zeros(len(left_ankle_names), dtype=torch.long, device=self.device)
        self.left_ankle_names = left_ankle_names
        for i, name in enumerate(left_ankle_names):
            self.left_ankle_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        right_ankle_names = []
        for target_name in self.cfg.asset.right_ankle_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name:
                    right_ankle_names.append(source_name)
        self.right_ankle_indices = torch.zeros(len(right_ankle_names), dtype=torch.long, device=self.device)
        self.right_ankle_names = right_ankle_names
        for i, name in enumerate(right_ankle_names):
            self.right_ankle_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

    def _get_env_origins(self):
        # 为每个机器人实例分配一个不重叠的初始位置
        self.custom_origins = False
        # env_origins (num_envs, 3) 存储每个环境的原点位置
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        """解析配置文件，初始化关键参数。

        Args:
            cfg: 配置对象，包含所有环境参数

        参数说明：
        1. self.dt: 控制步长（秒）
           - 来源: cfg.control.decimation * sim_params.dt
           - decimation是物理模拟步数/控制步数的比值，sim_params.dt是物理仿真时间步长
           - 作用: 决定机器人控制频率，用于计算速度、加速度等时间相关量
           - 去向: 在step()、compute_reward()等方法中用于时间计算
        2. self.obs_scales: 观测值缩放因子
           - 来源: cfg.normalization.obs_scales (包含ang_vel, lin_vel, dof_pos, dof_vel等)
           - 作用: 标准化观测值到合适的数值范围，提升神经网络训练稳定性
           - 去向: compute_observations()中对角速度、关节位置/速度等进行缩放
        3. self.reward_scales: 奖励函数权重字典
           - 来源: cfg.rewards.scales 转换为字典
           - 作用: 控制各个奖励项的权重（如方向、高度、关节限制等）
           - 去向: _prepare_reward_function()中筛选非零奖励，compute_reward()中计算总奖励
        4. self.constraint_scales: 约束惩罚权重字典
           - 来源: cfg.constraints.scales 转换为字典
           - 作用: 控制各个约束项的惩罚力度（如碰撞、力矩限制等）
           - 去向: compute_reward()中添加约束惩罚项到总奖励
        5. self.command_ranges: 命令范围字典
           - 来源: cfg.commands.ranges (如lin_vel_x, lin_vel_y, ang_vel_yaw的范围)
           - 作用: 定义速度命令的采样范围（用于运动任务）
           - 去向: 当前代码中主要用于课程学习和命令生成（如果启用）
        6. self.max_episode_length_s: 回合最大时长（秒）
           - 来源: cfg.env.episode_length_s
           - 作用: 设置每个训练回合的最大持续时间
           - 去向: 用于计算max_episode_length（步数）
        7. self.max_episode_length: 回合最大步数
           - 来源: max_episode_length_s / dt 向上取整
           - 作用: 控制何时触发超时重置
           - 去向: check_termination()中判断是否超时 (episode_length_buf > max_episode_length)
        8. cfg.domain_rand.push_interval: 随机推动间隔（步数）
           - 来源: push_interval_s / dt 向上取整
           - 作用: 将秒数转换为仿真步数，控制随机外力施加频率（domain randomization）
           - 去向: 用于增强训练鲁棒性的随机扰动
        """
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        """
        【重要】rewards / constraints 配置结构说明：

        cfg.rewards (外层类) 包含两类参数：
          1. cfg.rewards.scales (内层类) - 各奖励项的权重系数
             例如：termination=-0.0, torques=-0.00001, orientation=-0.0
             作用：控制每个奖励项在总奖励中的比重（类似loss权重）
             使用：仅在此处提取为字典，后续在 _prepare_reward_function() 中筛选非零项
          2. cfg.rewards 的其他属性 (外层参数) - 奖励计算的配置参数
             例如：tracking_sigma=0.25, soft_dof_pos_limit=1.0, base_height_target=1.0
             作用：各奖励函数内部计算时使用的超参数
             使用：在各个 _reward_xxx() 函数中直接访问 self.cfg.rewards.xxx
             示例：
               - _reward_tracking_lin_vel() 中用 self.cfg.rewards.tracking_sigma
               - _reward_base_height() 中用 self.cfg.rewards.base_height_target
               - _reward_dof_pos_limits() 中用 self.cfg.rewards.soft_dof_pos_limit

        分离"权重"和"参数"使配置更清晰，便于调整奖励权重而不影响内部计算逻辑，constraints 同理
        """
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.constraint_scales = class_to_dict(self.cfg.constraints.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self._refresh_tensor_state()
        terrain_sphere = gymutil.WireframeSphereGeometry(0.02, 5, 5, None, color=(1, 1, 0))
        marker_sphere = gymutil.WireframeSphereGeometry(0.03, 5, 5, None, color=(0, 1, 1))
        axes_geom = gymutil.AxesGeometry(scale=0.2)

        for i in range(self.num_envs):
            base_pos = self.base_pos[i].cpu().numpy()

            motion_body_pos = self.motion_dict['keyframe_pos'][i].clone()
            motion_body_pos = motion_body_pos.cpu().numpy()
            motion_body_quat = self.motion_dict['keyframe_quat'][i].cpu().numpy()

            for j in range(len(self.keyframe_indices)):
                x, y, z = motion_body_pos[j, 0], motion_body_pos[j, 1], motion_body_pos[j, 2]
                a, b, c, d = motion_body_quat[j, 0], motion_body_quat[j, 1], motion_body_quat[j, 2], motion_body_quat[j, 3]
                target_sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(a, b, c, d))
                gymutil.draw_lines(marker_sphere, self.gym, self.viewer, self.envs[i], target_sphere_pose)
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], target_sphere_pose)

    def _refresh_tensor_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _get_shoulder_height(self, env_ids=None):
        """Get shoudler height"""
        left_shoulder_pos = self.rigid_body_states[:, self.left_shoulder_indices, :3].clone()
        right_shoulder_pos = self.rigid_body_states[:, self.right_shoulder_indices, :3].clone()

        if self.cfg.terrain.mesh_type == 'plane':
            left_shoulder_height = torch.mean(left_shoulder_pos[:, :, 2], dim=-1, keepdim=True)
            right_shoulder_height = torch.mean(right_shoulder_pos[:, :, 2], dim=-1, keepdim=True)
            return torch.cat((left_shoulder_height, right_shoulder_height), dim=-1)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

    # ------------ reward functions----------------

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_lin_vel_xy(self):
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) * -5) * base_height

    def _reward_ang_vel_xy(self):
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2) * base_height

    def _reward_style_ang_vel_xy(self):
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2) * base_height

    def _reward_orientation(self):
        if not self.is_gaussian:
            # 下面两行有问题，不使用
            mse_error = torch.sum(torch.square(self.projected_gravity - torch.tensor([0, 0, -1], device=self.device)), dim=-1)
            reward = torch.exp(-mse_error / self.cfg.rewards.orientation_sigma) * (self.root_states[:, 2] > 0.4)
        else:
            base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
            reward = tolerance(-self.projected_gravity[:, 2], [self.cfg.rewards.orientation_threshold, np.inf], 1.0, 0.05) * base_height
        return reward

    def _reward_base_height(self):
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_head_height(self):
        if not self.is_gaussian:
            head_height = self.rigid_body_states[:, self.head_indices, 2].clone()
            return head_height.squeeze(1).clamp(0, 1)
        else:
            head_height = self.rigid_body_states[:, self.head_indices, 2].clone()
            feet_height = self.rigid_body_states[:, self.feet_indices, 2].clone().mean(-1).unsqueeze(-1)
            head_height -= feet_height
            reward = tolerance(head_height, (self.cfg.rewards.target_head_height, np.inf), self.cfg.rewards.target_head_margin, 0.1)
            delta_max_headheight = head_height - self.max_headheight
            delta_headheight = head_height - self.old_headheight
            self.max_headheight = torch.max(torch.cat((head_height, self.old_headheight), dim=1), dim=1)[0].unsqueeze(-1)
            self.old_headheight = head_height
            return reward

    def _reward_shank_orientation(self):
        # 获取膝盖和脚的位置
        left_knee_pos = self.rigid_body_states[:, self.left_knee_indices, :3].clone()
        right_knee_pos = self.rigid_body_states[:, self.right_knee_indices, :3].clone()
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        # 计算小腿向量中Z分量的比例（Z分量/向量长度）
        left_feet_orientation = (left_knee_pos - left_foot_pos)[:, :, 2] / torch.norm(left_knee_pos - left_foot_pos, dim=-1)
        right_feet_orientation = (right_knee_pos - right_foot_pos)[:, :, 2] / torch.norm(right_knee_pos - right_foot_pos, dim=-1)
        # 左右取平均
        feet_orientation = torch.mean(torch.concat([left_feet_orientation, right_feet_orientation], dim=-1), dim=-1)
        # 仅在1阶段后生效
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        reward = tolerance(feet_orientation, [0.8, np.inf], 1, 0.1) * base_height
        # 3阶段后奖励恒为1【post_task控制】
        if self.cfg.constraints.post_task:
            standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        return reward

    def _reward_thigh_ori(self):
        # 获取膝盖和大腿的位置
        left_knee_pos = self.rigid_body_states[:, self.left_knee_indices, :3].clone()
        right_knee_pos = self.rigid_body_states[:, self.right_knee_indices, :3].clone()
        left_thigh_pos = self.rigid_body_states[:, self.left_thigh_indices, :3].clone()
        right_thigh_pos = self.rigid_body_states[:, self.right_thigh_indices, :3].clone()
        # 计算大腿向量中Z分量的比例（Z分量/向量长度）
        left_feet_orientation = (left_thigh_pos - left_knee_pos)[:, :, 2] / torch.norm(left_thigh_pos - left_knee_pos, dim=-1)
        right_feet_orientation = (right_thigh_pos - right_knee_pos)[:, :, 2] / torch.norm(right_thigh_pos - right_knee_pos, dim=-1)
        # 左右取平均
        feet_orientation = torch.min(torch.concat([left_feet_orientation, right_feet_orientation], dim=-1), dim=-1)[0]
        # 仅在1阶段后生效
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        reward = tolerance(feet_orientation, [0.8, np.inf], 1, 0.1) * base_height
        # 3阶段后奖励恒为1【post_task控制】
        if self.cfg.constraints.post_task:
            standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        return reward

    def _reward_feet_ori_rate(self):
        # 获取膝盖和脚的位置
        left_knee_pos = self.rigid_body_states[:, self.left_knee_indices, :3].clone()
        right_knee_pos = self.rigid_body_states[:, self.right_knee_indices, :3].clone()
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        # 计算小腿向量中Z分量的比例（Z分量/向量长度）
        left_feet_orientation = (left_knee_pos - left_foot_pos)[:, :, 2] / torch.norm(left_knee_pos - left_foot_pos, dim=-1)
        right_feet_orientation = (right_knee_pos - right_foot_pos)[:, :, 2] / torch.norm(right_knee_pos - right_foot_pos, dim=-1)
        # 左右取最大值
        feet_orientation = torch.max(left_feet_orientation, right_feet_orientation)
        # 和上一帧比较，越接近则reward越小
        reward = torch.sum(torch.square(self.feet_ori - feet_orientation), dim=1)
        self.feet_ori = feet_orientation
        return reward

    def _reward_target_upper_dof_pos(self):
        mse = torch.sum(torch.square(self.dof_pos[:, self.upper_body_joint_indices] - self.target_dof_pos[:, self.upper_body_joint_indices]), dim=-1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(mse * self.cfg.rewards.target_dof_pos_sigma)
        reward = reward * standup
        return reward

    def _reward_target_lower_dof_pos(self):
        mse = torch.sum(torch.square(self.dof_pos[:, self.lower_body_joint_indices] - self.target_dof_pos[:, self.lower_body_joint_indices]), dim=-1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(mse * self.cfg.rewards.target_dof_pos_sigma)
        reward = reward * standup
        return reward

    def _reward_low_base_vel(self):
        th = 0.3
        base_x_vel = self.base_lin_vel[:, 0]
        base_y_vel = self.base_lin_vel[:, 1]
        base_z_vel = self.base_lin_vel[:, 2]
        reward_x = 1 / 3 * tolerance(base_x_vel, [-th, th], 1.2, 0.1)
        reward_y = 1 / 3 * tolerance(base_y_vel, [-th, th], 1.2, 0.1)
        reward_z = 1 / 3 * tolerance(base_z_vel, [-th, th], 1.2, 0.1)
        return reward_x + reward_y + reward_z

    def _reward_feet_distance(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=-1)
        reward = tolerance(feet_distances, [0, 0.4], 0.38, 0.05)
        return reward

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_hip_torques(self):
    #     return torch.sum((self.torques[:, self.all_hip_joint_indices] - 15).clamp(0, np.inf), dim=1)

    def _reward_collision(self):
        return torch.sum(1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # 速度限制需要截断
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0.0, max=1.0), dim=1)

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1)

    # # 命令跟踪线速度误差惩罚【行走/奔跑专用】
    # def _reward_tracking_lin_vel(self):
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    # # 命令跟踪角速度误差惩罚【行走/奔跑专用】
    # def _reward_tracking_ang_vel(self):
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # # 0.5s 步频运动【行走/奔跑专用】
    # def _reward_feet_air_time(self):
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact
    #     # 之前在空中 + 现在落地 = 第一次接触地面
    #     first_contact = (self.feet_air_time > 0.0) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

    def _reward_stumble(self):
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    # # 不给命令时维持站立不动【行走/奔跑专用】
    # def _reward_stand_still(self):
    #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.0), dim=1)

    def _reward_body_height(self):
        base_height = self._get_base_heights()
        return base_height

    def _reward_feet_on_ground(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1
        # 双脚着地：0，非双脚着地：-1
        contact = (torch.sum(1.0 * contacts, dim=1) == 2).float() - 1
        return contact

    def _reward_penalize_torso_contact(self):
        return torch.sum(1.0 * (torch.norm(self.contact_forces[:, self.upper_body_index, :], dim=-1).unsqueeze(1) > 0.1), dim=1)

    def _reward_lower_body_contact(self):
        contacts = torch.norm(self.contact_forces[:, self.lower_body_contact_indices], dim=-1) > 1.0
        contact = (torch.sum(contacts.float(), dim=-1) >= 1).float() - 1
        return contact

    def _reward_shoulder_height(self):
        # TODO：硬编码肩膀高度
        shoulder_height = self._get_shoulder_height()
        return torch.min(shoulder_height[:, 0], shoulder_height[:, 1]) - 1

    def _reward_penalize_shoulder_height_variance(self):
        shoulder_height = self._get_shoulder_height()
        return shoulder_height.var(-1)

    def _reward_joint_power(self):
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_smoothness(self):
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)

    def _reward_tracking_dof_pos(self):
        dof_pos_error = torch.sum(torch.square(self.target_dof_pos - self.dof_pos), dim=1)
        return torch.exp(-dof_pos_error / self.cfg.rewards.tracking_dof_sigma)

    def _reward_deviation_knee_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.target_dof_pos)[:, self.knee_joint_indices], dim=-1)

    def _reward_left_foot_displacement(self):
        # 左脚限制在基座0.3m半径以内
        base_xy = self.root_states[:, :2].clone()
        left_foot_xy = self.rigid_body_states[:, self.left_foot_indices, :2].squeeze(1)
        mse_error = torch.sum(torch.square(base_xy - left_foot_xy), dim=-1).clamp(0.3, np.inf)
        # 脚贴地（脚高度小于0.3）才计算
        reward = torch.exp(mse_error * self.cfg.rewards.left_foot_displacement_sigma) * (self.rigid_body_states[:, self.left_foot_indices, 2] < 0.3).squeeze(1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return reward * standup

    def _reward_right_foot_displacement(self):
        # 右脚限制在基座0.3m半径以内
        base_xy = self.root_states[:, :2].clone()
        right_foot_xy = self.rigid_body_states[:, self.right_foot_indices, :2].squeeze(1)
        mse_error = torch.sum(torch.square(base_xy - right_foot_xy), dim=-1).clamp(0.3, np.inf)
        # 脚贴地（脚高度小于0.3）才计算
        reward = torch.exp(mse_error * self.cfg.rewards.right_foot_displacement_sigma) * (self.rigid_body_states[:, self.right_foot_indices, 2] < 0.3).squeeze(1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return reward * standup

    # # 腰关节活动范围限制【有腰关节专属】
    # def _reward_waist_deviation(self):
    #     wrist_dof = self.dof_pos[:, self.waist_joint_indices]
    #     reward = (torch.abs(wrist_dof) > 1.4).float()
    #     return reward.squeeze(1)

    def _reward_hip_yaw_deviation(self):
        hip_yaw_dof = self.dof_pos[:, self.hip_yaw_joint_indices]
        hip_yaw_abs = torch.abs(hip_yaw_dof)
        reward = (torch.max(hip_yaw_abs, dim=-1)[0] > 0.33) | (torch.min(hip_yaw_abs, dim=-1)[0] > 0.3)
        return reward

    def _reward_hip_roll_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.hip_roll_joint_indices]
        hip_roll_abs = torch.abs(hip_roll_dof)
        reward = (torch.max(hip_roll_abs, dim=-1)[0] > 0.5) | (torch.min(hip_roll_abs, dim=-1)[0] > 0.45)
        return reward

    def _reward_hip_pitch_deviation(self):
        hip_pitch_dof = self.dof_pos[:, self.hip_pitch_joint_indices]
        hip_pitch_abs = torch.abs(hip_pitch_dof)
        reward = (torch.max(hip_pitch_abs, dim=-1)[0] > 1.6) | (torch.min(hip_pitch_dof, dim=-1)[0] < -0.38)
        return reward

    def _reward_knee_deviation(self):
        knee_dof = self.dof_pos[:, self.knee_joint_indices]
        knee_abs = torch.abs(knee_dof)
        reward = (torch.max(knee_abs, dim=-1)[0] > 2.85) | (torch.min(knee_dof, dim=-1)[0] < -0.06)
        return reward

    def _reward_shoulder_roll_deviation(self):
        shoulder_roll_dof = self.dof_pos[:, self.shoulder_roll_joint_indices]
        reward = (shoulder_roll_dof[:, 0] < -0.02) | (shoulder_roll_dof[:, 1] > 0.02)
        return reward

    def _reward_lower_body_deviation(self):
        lower_body_dof_left = torch.cat([self.left_hip_roll_joint_indices, self.left_hip_pitch_joint_indices, self.left_hip_yaw_joint_indices, self.left_knee_joint_indices])
        lower_body_dof_right = torch.cat([self.right_hip_roll_joint_indices, self.right_hip_pitch_joint_indices, self.right_hip_yaw_joint_indices, self.right_knee_joint_indices])
        left_dof_pos = self.dof_pos[:, lower_body_dof_left].unsqueeze(1)
        left_dof_pos[:, :, 0] = -left_dof_pos[:, :, 0]
        left_dof_pos[:, :, 2] = -left_dof_pos[:, :, 2]
        right_dof_pos = self.dof_pos[:, lower_body_dof_right].unsqueeze(1)
        reward = torch.sum(torch.var(torch.cat([left_dof_pos, right_dof_pos], dim=1), dim=1), dim=-1)
        reward = torch.exp(reward * -2)
        return reward

    def _reward_hip_yaw_var(self):
        hip_yaw_dof = self.dof_pos[:, self.hip_yaw_joint_indices]
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase2
        reward = torch.exp(torch.var(hip_yaw_dof, dim=-1) * self.cfg.constraints.hip_yaw_var_sigma) * standup
        return reward

    def _reward_hip_roll_var(self):
        hip_roll_dof = self.dof_pos[:, self.hip_roll_joint_indices]
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase2
        reward = torch.exp(torch.var(hip_roll_dof, dim=-1) * self.cfg.constraints.hip_yaw_var_sigma) * standup
        return reward

    def _reward_hip_pitch_var(self):
        hip_pitch_dof = self.dof_pos[:, self.hip_pitch_joint_indices]
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase2
        reward = torch.exp(torch.var(hip_pitch_dof, dim=-1) * self.cfg.constraints.hip_yaw_var_sigma) * standup
        return reward

    def _reward_feet_height_var(self):
        left_foot_height = self.rigid_body_states[:, self.left_foot_indices, 2].clone() * 10
        right_foot_height = self.rigid_body_states[:, self.right_foot_indices, 2].clone() * 10
        feet_distance = torch.abs(left_foot_height - right_foot_height).squeeze(1).clamp(0.2, np.inf)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(feet_distance * -2) * standup

    def _reward_upper_body_var(self):
        left_upper_body_pos = self.rigid_body_states[:, self.left_upper_body_indices, :3].clone() * 10
        right_upper_body_pos = self.rigid_body_states[:, self.right_upper_body_indices, :3].clone() * 10
        upper_body_distance = torch.norm(left_upper_body_pos - right_upper_body_pos, dim=-1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(upper_body_distance.var(1) * -2) * standup

    def _reward_lower_body_var(self):
        left_lower_body_pos = self.rigid_body_states[:, self.left_lower_body_indices, :3].clone() * 10
        right_lower_body_pos = self.rigid_body_states[:, self.right_lower_body_indices, :3].clone() * 10
        lower_body_distance = torch.norm(left_lower_body_pos - right_lower_body_pos, dim=-1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(lower_body_distance.var(1) * -2) * standup

    def _reward_ground_parallel(self):
        left_ankle_pos = self.rigid_body_states[:, self.left_ankle_indices, 2].clone() * 10
        right_ankle_pos = self.rigid_body_states[:, self.right_ankle_indices, 2].clone() * 10
        var = torch.mean(torch.concat([left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1), dim=-1)
        reward = var < 0.05
        # 3阶段后奖励恒为1【post_task控制】
        if self.cfg.constraints.post_task:
            standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        return reward

    def _reward_target_orientation(self):
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -5) * standup

    def _reward_target_base_height(self):
        base_height = self.root_states[:, 2]
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(torch.abs(base_height - self.cfg.rewards.base_height_target) * - 20) * standup

    def _reward_feet_parallel(self):
        left_feet_pos = self.rigid_body_states[:, self.left_ankle_indices, :2].clone() * 10
        right_feet_pos = self.rigid_body_states[:, self.right_ankle_indices, :2].clone() * 10
        feet_distances = torch.norm(left_feet_pos - right_feet_pos, dim=-1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return (torch.var(feet_distances, dim=-1) < 1) * standup

    # # 腰关节和目标差【有腰关节专属】
    # def _reward_target_waist_dof_pos(self):
    #     mse = torch.sum(torch.square(self.dof_pos[:, self.waist_joint_indices] - self.target_dof_pos[:, self.waist_joint_indices]), dim=-1)
    #     standup =self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
    #     reward = torch.exp(mse * self.cfg.rewards.target_dof_pos_sigma)
    #     reward = reward * standup
    #     return reward
