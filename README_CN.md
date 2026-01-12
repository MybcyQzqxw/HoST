# HoST: 人形机器人站立控制

[![arXiv](https://img.shields.io/badge/arXiv-2502.08378-brown)](https://arxiv.org/abs/2502.08378)
[![](https://img.shields.io/badge/Website-%F0%9F%9A%80-yellow)](https://taohuang13.github.io/humanoid-standingup.github.io/)
[![](https://img.shields.io/badge/Youtube-🎬-red)](https://www.youtube.com/watch?v=Yruh-3CFwE4)
[![](https://img.shields.io/badge/Bilibili-📹-blue)](https://www.bilibili.com/video/BV1o2KPeUEob/?spm_id_from=333.337.search-card.all.click&vd_source=ef6a9a20816968cc19099a3f662afd86)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)]()


这是RSS会议论文"[**Learning Humanoid Standing-up Control across Diverse Postures**](https://arxiv.org/abs/2502.08378)"的官方PyTorch实现，作者为：

[Tao Huang](https://taohuang13.github.io/)、[Junli Ren](https://renjunli99.github.io/)、[Huayi Wang](https://why618188.github.io/)、[Zirui Wang](https://scholar.google.com/citations?user=Vc3DCUIAAAAJ&hl=zh-TW)、[Qingwei Ben](https://www.qingweiben.com/)、[Muning Wen](https://scholar.google.com/citations?user=Zt1WFtQAAAAJ&hl=en)、[Xiao Chen](https://xiao-chen.tech/)、[Jianan Li](https://github.com/OpenRobotLab/HoST)、[Jiangmiao Pang](https://oceanpang.github.io/)

<p align="left">
  <img width="98%" src="docs/teaser.png" style="box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;">
</p>

## 📑 目录
- [🔥 新闻](#-新闻)
- [📝 待办事项](#-待办事项)
- [🛠️ 安装说明](#-安装说明)
- [🤖 在Unitree G1上运行HoST](#-在unitree-g1上运行host)
- [🧭 将HoST扩展到其他人形机器人](#-将host扩展到其他人形机器人建议)
- [✉️ 联系方式](#-联系方式)
- [🏷️ 许可证](#-许可证)
- [🎉 致谢](#-致谢)
- [📝 引用](#-引用)

## 🔥 新闻
- \[2025-06\] HoST入选RSS 2025最佳系统论文提名！
- \[2025-05\] [DroidUp](https://droidup.com/)现已支持HoST！代码即将发布。
<p align="center">
  <img width="26%" src="docs/droidup.gif" style="box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.3); border-radius: 4px; margin: -5px -0px -10px 0px;">
</p>

- \[2025-05\] [High Torque Mini Pi](https://www.hightorquerobotics.com/pi/)现已支持HoST！代码已发布。
<table style="width: 100%; border-collapse: collapse; margin: -5px -0px -0px 0px;">
    <tr>
        <td align="center" style="width: 24%; padding: 2px;">
            <img src="docs/pi_gym.gif" alt="IsaacGym" style="width: 98%; max-width: 100%;"/><br/>
            <span style="font-size: 0.9em;">IsaacGym</span>
        </td>
        <td align="center" style="width: 24%; padding: 2px;">
            <img src="docs/pi_ground.gif" alt="Supine" style="width: 98%; max-width: 100%;"/><br/>
            <span style="font-size: 0.9em;">仰卧</span>
        </td>
        <td align="center" style="width: 24%; padding: 2px;">
            <img src="docs/pi_prone.gif" alt="Prone" style="width: 98%; max-width: 100%;"/><br/>
            <span style="font-size: 0.9em;">俯卧</span>
        </td>
        <td align="center" style="width: 24%; padding: 2px;">
            <img src="docs/pi_side.gif" alt="Side" style="width: 98%; max-width: 100%;"/><br/>
            <span style="font-size: 0.9em;">侧卧</span>
        </td>
    </tr>
</table>

- \[2025-04\] 我们发布了训练代码、评估脚本和可视化工具。
- \[2025-04\] HoST被RSS 2025接收！
- \[2025-02\] 我们发布了HoST的[论文](https://taohuang13.github.io/humanoid-standingup.github.io/assets/paper.pdf)和[演示](https://taohuang13.github.io/humanoid-standingup.github.io/)。


## 📝 待办事项
- [x] Unitree G1跨俯卧姿势的训练代码。
- [x] Unitree H1的训练代码。
- [ ] 仰卧和俯卧姿势的联合训练。
- [ ] 所有地形的联合训练。


## 🛠️ 安装说明
克隆此仓库：
```bash
git clone https://github.com/OpenRobotLab/HoST.git
cd HoST
```
创建conda环境：
```bash
conda env create -f conda_env.yml 
conda activate host
```
安装pytorch 1.10和cuda-11.3：
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

下载并安装[Isaac Gym](https://developer.nvidia.com/isaac-gym)：
```bash
cd isaacgym/python && pip install -e .
```
安装rsl_rl（PPO实现）和legged gym：
```bash
cd rsl_rl && pip install -e . && cd .. 
cd legged_gym &&  pip install -e . && cd .. 
```
### 错误处理
关于潜在的安装错误，请参考[此文档](docs/ERROR.md)获取解决方案。

## 🤖 在Unitree G1上运行HoST
### 主要仿真动作概览
<table style="width: 100%; border-collapse: collapse; margin: -5px -0px -12px 0px;">
    <tr>
        <td align="center" style="width: 24%; padding: 2px;">
            <img src="docs/results_ground_10000.gif" alt="Ground" style="width: 98%; max-width: 100%;"/><br/>
            <span style="font-size: 0.9em;">地面</span>
        </td>
        <td align="center" style="width: 24%; padding: 2px;">
            <img src="docs/results_platform_12000.gif" alt="Platform" style="width: 98%; max-width: 100%;"/><br/>
            <span style="font-size: 0.9em;">平台</span>
        </td>
        <td align="center" style="width: 24%; padding: 2px;">
            <img src="docs/results_wall_4000.gif" alt="Platform" style="width: 98%; max-width: 100%;"/><br/>
            <span style="font-size: 0.9em;">墙壁</span>
        </td>
        <td align="center" style="width: 24%; padding: 2px;">
            <img src="docs/results_slope_8000.gif" alt="Slope" style="width: 98%; max-width: 100%;"/><br/>
            <span style="font-size: 0.9em;">斜坡</span>
        </td>
    </tr>
</table>

### 策略训练
在不同地形上训练站立策略：
```bash
python legged_gym/scripts/train.py --task g1_${terrain} --run_name test_g1 # [ground, platform, slope, wall]
```

训练完成后，你可以运行生成的检查点：
```bash
python legged_gym/scripts/play.py --task g1_${terrain} --checkpoint_path ${/path/to/ckpt.pt} # [ground, platform, slope, wall]
```

### 策略评估
我们还提供了评估脚本来记录成功率、脚部移动距离、运动平滑度和消耗能量：
```bash
python legged_gym/scripts/eval/eval_${terrain}.py --task g1_${terrain} --checkpoint_path ${/path/to/ckpt.pt} # [ground, platform, slope, wall]
```
在评估过程中应用领域随机化，使结果更具泛化性。

### 运动可视化
<p align="left">
  <img width="98%" src="docs/motion_vis.png" style="box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;">
</p>


首先，运行以下命令收集生成的运动：
```bash
python legged_gym/scripts/visualization/motion_collection.py --task g1_${terrain} --checkpoint_path ${/path/to/ckpt.pt} # [ground, platform, slope, wall]
```

其次，绘制运动关键帧的3D轨迹：
```bash
python legged_gym/scripts/visualization/trajectory_hands_feet.py  --terrain ${terrain} # [ground, platform, slope, wall]
python legged_gym/scripts/visualization/trajectory_head_pelvis.py  --terrain ${terrain} # [ground, platform, slope, wall]
```

### 从俯卧姿势开始训练
<table style="width: 100%; border-collapse: collapse; margin: -5px -0px -0px 0px;">
    <tr>
        <td align="center" style="width: 33%; padding: 3px;">
            <img src="docs/results_leftside.gif" alt="Ground" style="width: 98%; max-width: 100%; height: auto; box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;"/><br/>
            <span style="font-size: 0.9em;">左侧卧</span>
        </td>
        <td align="center" style="width: 33%; padding: 3px;">
            <img src="docs/results_prone.gif" alt="Platform" style="width: 98%; max-width: 100%; height: auto; box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;"/><br/>
            <span style="font-size: 0.9em;">俯卧</span>
        </td>
        <td align="center" style="width: 33%; padding: 3px;">
            <img src="docs/results_rightside.gif" alt="Slope" style="width: 98%; max-width: 100%; height: auto; box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;"/><br/>
            <span style="font-size: 0.9em;">右侧卧</span>
        </td>
    </tr>
</table>

我们还支持从俯卧姿势开始的训练：
```bash
python legged_gym/scripts/train.py --task g1_ground_prone --run_name test_g1_ground_prone
```
学习到的策略也可以处理侧卧姿势。然而，从俯卧姿势训练时，需要对髋关节施加更严格的约束以防止剧烈运动。这个问题使得俯卧和仰卧姿势的联合训练的可行性目前尚不明确。解决这个问题将是未来有价值的工作。

## 🧭 将HoST扩展到其他人形机器人：建议
### 从Unitree H1和H1-2中学到的经验
<p align="left">
  <img width="98%" src="docs/results_sim_h1_h12.png" style="box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;">
</p>
要尝试其他机器人，应遵循以下步骤来使算法工作：

* [在urdf中添加关键帧](./legged_gym/resources/robots/g1/g1_23dof.urdf#L970)：建议添加与我们相同的关键帧（包括脚踝周围的关键点），以增强与新机器人的兼容性。这些关键帧设计用于奖励计算。
* [拉力](./legged_gym/legged_gym/envs/g1/g1_config_ground.py#L296)：约为机器人重力的60%。请注意，我们在G1的urdf中有两个躯干链接（一个真实的，一个虚拟的），因此训练期间力将乘以2。此外，你可以修改施加力的条件，例如，移除基座方向条件。
* [课程高度](./legged_gym/legged_gym/envs/g1/g1_config_ground.py#L299)：约为机器人高度的70%。
* [阶段划分高度](./legged_gym/legged_gym/envs/g1/g1_config_ground.py#L189)：阶段1和2约为机器人高度的35%，阶段3约为机器人高度的70%。
* [奖励高度](./legged_gym/legged_gym/envs/g1/g1_config_ground.py#L187)：target_head_height约为75%。关于成功站立后的[目标基座高度](./legged_gym/legged_gym/envs/g1/g1_config_ground.py#L183)，这取决于你的偏好。
* [关节偏差奖励](./legged_gym/legged_gym/envs/g1/g1_config_ground.py#L231)：你可以调整风格奖励函数，特别是关节偏差惩罚，以更好地约束运动风格。轻量级机器人通常需要更窄的期望关节角度范围，因为它们更容易达到极端关节角度。
* [奖励组权重](./legged_gym/legged_gym/envs/g1/g1_config_ground.py#L200)：例如，提高风格奖励的权重可能会优先优化运动。这对学习H1-2或跨俯卧姿势很有帮助。
* [其他](./legged_gym/legged_gym/envs/g1/g1_config_ground.py)：你还应该修改默认/目标姿势、PD控制器、观察/动作空间、身体名称等。

作为示例，我们提供了Unitree H1和[High Torque Mini Pi](https://www.hightorquerobotics.com/pi/)在地面上的训练代码：
```bash
python legged_gym/scripts/train.py --task h1_ground --run_name test_h1_ground 
python legged_gym/scripts/train.py --task pi_ground --run_name test_minipi_ground
```

### 硬件部署的潜在建议
<p align="left">
  <img width="98%" src="docs/results_real_h12.png" style="box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;">
</p>
我们发现一些对G1和H1-2硬件系统有用的建议：

* **膝关节和髋关节的高刚度：**我们发现，将这些关节的kp系数提高到仿真值的1.33倍到1.5倍会显著有帮助。我们认为这是由关节力矩之间的仿真到现实差距造成的。有关更多分析，请参阅[论文](https://arxiv.org/abs/2502.08378)。
* **高动作缩放器：**虽然默认的动作缩放器（0.25）已经产生了良好的运动，但我们发现稍微提高此系数（0.3）可以显著减轻抖动运动。
* **检查碰撞模型：**我们发现使用完整网格作为脚踝的碰撞模型会导致巨大的仿真到现实差距。为了解决这个问题，我们使用离散点来近似碰撞，遵循[Unitree的官方代码](https://github.com/unitreerobotics/unitree_rl_gym)。也就是说，强烈建议对碰撞模型更加小心。

## ✉️ 联系方式
如有任何问题，请随时发送电子邮件至taou.cs13@gmail.com。我们会尽快回复。

## 🏷️ 许可证
本仓库在MIT许可证下发布。更多详情请参见[LICENSE](LICENSE)。

## 🎉 致谢
本仓库建立在以下开源项目的支持和贡献之上。特别感谢：

* [legged_gym](https://github.com/leggedrobotics/legged_gym)和[HIMLoco](https://github.com/OpenRobotLab/HIMLoco)：训练和运行代码的基础。
* [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git)：强化学习算法实现。
* [walk these ways](https://github.com/Improbable-AI/walk-these-ways)：硬件代码骨架。
* [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)：硬件通信接口。
* [HoST-pytorch](https://github.com/lucidrains/HoST-pytorch)：我们感谢[Phil Wang](https://github.com/lucidrains)复现我们的代码库并指出论文中的一些错误。

## 📝 引用

如果你觉得我们的工作有用，请考虑引用：
```
@article{huang2025learning,
  title={Learning Humanoid Standing-up Control across Diverse Postures},
  author={Huang, Tao and Ren, Junli and Wang, Huayi and Wang, Zirui and Ben, Qingwei and Wen, Muning and Chen, Xiao and Li, Jianan and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2502.08378},
  year={2025}
}
```
