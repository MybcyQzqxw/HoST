# BHR8FC2 机器人集成说明

## ✅ 已完成的工作

1. **目录结构已创建**
   - `/legged_gym/resources/robots/bhr8fc2/` - 机器人资源目录
   - `/legged_gym/resources/robots/bhr8fc2/meshes/` - 网格文件目录
   - `/legged_gym/legged_gym/envs/bhr8fc2/` - 配置文件目录

2. **配置文件已创建**
   - `bhr8fc2_config_ground.py` - 地面训练配置文件

3. **任务已注册**
   - 在 `__init__.py` 中注册了 `bhr8fc2_ground` 任务

## 📋 后续需要完成的工作

### 1. 准备URDF文件

将 BHR8FC2 的 URDF 文件和相关 mesh 文件放置到正确位置：

```bash
# URDF文件应该放在：
legged_gym/resources/robots/bhr8fc2/bhr8fc2.urdf

# Mesh文件应该放在：
legged_gym/resources/robots/bhr8fc2/meshes/
```

### 2. 修改配置文件

打开 `legged_gym/legged_gym/envs/bhr8fc2/bhr8fc2_config_ground.py`，根据实际情况修改：

#### 2.1 关节名称
所有标记了 `TODO` 的地方都需要根据 URDF 中的实际关节名称修改：
- `target_joint_angles` 字典
- `default_joint_angles` 字典
- `asset` 类中的关节列表

#### 2.2 自由度数量
```python
class env(LeggedRobotCfg.env):
    num_dofs = 20  # 修改为实际自由度数量
    num_actions = 20  # 修改为实际动作数量
```

#### 2.3 机器人尺寸参数
根据 BHR8FC2 的实际尺寸调整以下参数：
```python
# 在 rewards 类中：
base_height_target = 0.75  # 站立时基座高度（约为机器人高度的75%）
target_head_height = 1.0   # 头部目标高度（约为机器人总高度）

# 在 curriculum 类中：
base_height_target = 0.75  # 约为机器人高度的70%
pull_force_value = 200     # 约为机器人重量的60%（单位：N）
```

#### 2.4 Link名称
修改 `asset` 类中的 link 名称以匹配 URDF：
```python
left_foot_name = "left_ankle_pitch"  # 根据URDF修改
right_foot_name = "right_ankle_pitch"
base_name = 'torso_link'
# 等等...
```

#### 2.5 PD控制参数
根据机器人的实际特性调整刚度和阻尼：
```python
class control(LeggedRobotCfg.control):
    stiffness = {
        'hip': 150,    # 根据实际调整
        'knee': 200,
        'ankle': 40,
        # ...
    }
    damping = {
        'hip': 4,      # 根据实际调整
        'knee': 6,
        'ankle': 2,
        # ...
    }
```

### 3. 添加关键帧（可选但推荐）

如果需要奖励计算的关键帧，在 URDF 中添加类似以下的关键帧定义：
```xml
<link name="keyframe_head">
    <!-- 头部关键帧 -->
</link>

<joint name="keyframe_head_joint" type="fixed">
    <parent link="head_link"/>
    <child link="keyframe_head"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>
```

### 4. 测试训练

完成配置后，运行训练命令测试：
```bash
cd /home/mxqwthl/gitspace/HoST
python legged_gym/scripts/train.py --task bhr8fc2_ground --run_name test_bhr8fc2
```

### 5. 调试建议

如果遇到问题，按以下顺序检查：

1. **检查URDF加载**
   - 确认URDF文件路径正确
   - 检查所有mesh文件都在正确位置
   
2. **检查关节名称**
   - 配置文件中的关节名称必须与URDF完全匹配
   - 使用以下命令查看URDF中的关节名称：
     ```bash
     grep -o 'joint name="[^"]*"' legged_gym/resources/robots/bhr8fc2/bhr8fc2.urdf
     ```

3. **检查自由度匹配**
   - `num_dofs` 应该等于URDF中可控关节的数量
   - `num_actions` 通常等于 `num_dofs`

4. **逐步调整参数**
   - 先让机器人能够加载和站立
   - 然后调整奖励权重
   - 最后优化运动风格

## 📖 参考资料

- 参考 G1 的配置：`legged_gym/legged_gym/envs/g1/g1_config_ground.py`
- 参考 H1 的配置：`legged_gym/legged_gym/envs/h1/h1_config_ground.py`
- 参考 Pi 的配置：`legged_gym/legged_gym/envs/pi/pi_config_ground.py`

## 🚀 快速开始命令

```bash
# 训练
python legged_gym/scripts/train.py --task bhr8fc2_ground --run_name test_bhr8fc2

# 回放
python legged_gym/scripts/play.py --task bhr8fc2_ground --checkpoint_path legged_gym/logs/bhr8fc2_ground/xxx/model_xxx.pt
```
