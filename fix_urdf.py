#!/usr/bin/env python
import re

# 读取 URDF
with open('legged_gym/resources/robots/bhr8fc2/BHR8FC2.urdf', 'r') as f:
    content = f.read()

# 定义关节限制（根据人形机器人的常见限制）
joint_limits = {
    'hip_yaw': (-1.57, 1.57, 150),      # ±90度
    'hip_roll': (-0.52, 0.52, 150),     # ±30度  
    'hip_pitch': (-1.92, 0.87, 200),    # -110度到50度
    'knee': (0.0, 2.61, 200),           # 0到150度
    'ankle_pitch': (-0.87, 0.52, 40),   # -50度到30度
    'ankle_roll': (-0.35, 0.35, 40),    # ±20度
    'shoulder_pitch': (-2.87, 2.87, 100), # ±165度
    'shoulder_roll': (-2.87, 2.87, 100),  # ±165度
    'shoulder_yaw': (-2.87, 2.87, 100),   # ±165度
    'elbow': (0.0, 2.61, 100),           # 0到150度
}

# 替换所有 continuous 关节为 revolute 并添加 limit
def replace_joint(match):
    joint_name = match.group(1)
    joint_block = match.group(0)
    
    # 确定使用哪个限制
    limit_key = None
    for key in joint_limits.keys():
        if key in joint_name:
            limit_key = key
            break
    
    if limit_key:
        lower, upper, effort = joint_limits[limit_key]
        # 替换 type 为 revolute
        joint_block = joint_block.replace('type="continuous"', 'type="revolute"')
        # 在 </joint> 前添加 limit
        limit_tag = f'    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="30"/>\n  '
        joint_block = joint_block.replace('</joint>', f'{limit_tag}</joint>')
    
    return joint_block

# 匹配所有 continuous 关节（除了 fixed的head_joint）
pattern = r'<joint name="([^"]+)"[^>]*type="continuous"[^>]*>.*?</joint>'
content = re.sub(pattern, replace_joint, content, flags=re.DOTALL)

# 写回文件
with open('legged_gym/resources/robots/bhr8fc2/BHR8FC2.urdf', 'w') as f:
    f.write(content)

print("URDF updated successfully! All continuous joints converted to revolute with limits.")
