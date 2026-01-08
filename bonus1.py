import math
import time

import mujoco
from mujoco import viewer
import numpy as np

MJCF =''' 
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- 杯子主体：用一根竖直圆柱近似 -->
    <body name="cup" pos="0 0 0.5">
      <!-- free 关节：位置 + 姿态都由 qpos 控制 -->
      <joint name="cup_free" type="free"/>

      <!-- 杯子本体，高圆柱 -->
      <geom name="cup_body" type="capsule" size="0.1 0.25" rgba="1 0 255 255"/>

      <!-- 显示局部坐标轴：X 红、Y 绿、Z 蓝 -->
      <geom type="capsule" size="0.01 0.3" pos="0.3 0 0" euler="0 90 0" rgba="1 0 0 1"/>
      <geom type="capsule" size="0.01 0.3" pos="0 0.3 0" euler="90 0 0" rgba="0 1 0 1"/>
      <geom type="capsule" size="0.01 0.3" pos="0 0 0.3" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
'''
def axis_angle_to_quat(axis: str, angle_deg: float) -> np.ndarray:
    """绕单个坐标轴旋转 angle_deg（度），返回四元数 [qw, qx, qy, qz]."""
    half = math.radians(angle_deg) / 2.0
    c = math.cos(half)
    s = math.sin(half)
    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=float)
    elif axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=float)
    elif axis == "z":
        return np.array([c, 0.0, 0.0, s], dtype=float)
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 q = q1 ⊗ q2，均为 [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=float)

def main() -> None:
  # 1. 建模 + 初始姿态
  model = mujoco.MjModel.from_xml_string(MJCF)
  data = mujoco.MjData(model)

  data.qpos[0:3] = [0.0, 0.0, 0.5]           # 位置
  q0 = np.array([1.0, 0.0, 0.0, 0.0])        # 初始单位四元数
  data.qpos[3:7] = q0
  mujoco.mj_forward(model, data)

  # 目标角度
  target_y = 40.0
  target_x = -30.0
  target_z = 100.0

  print("初始姿态 q0 =", q0)

  # 2. 阶段 0：绕“自身 y 轴”转 40°
  q_step_y = axis_angle_to_quat("y", target_y)
  q1 = quat_mul(q0, q_step_y)   # 右乘 -> 自身轴
  q1 /= np.linalg.norm(q1)
  print("阶段0（自身 y 轴 40°）后的 q1 =", q1)

  # 3. 阶段 1：在当前姿态基础上，绕“自身 x 轴”转 -30°
  q_step_x = axis_angle_to_quat("x", target_x)
  q2 = quat_mul(q1, q_step_x)
  q2 /= np.linalg.norm(q2)
  print("阶段1（自身 x 轴 -30°）后的 q2 =", q2)

  # 4. 阶段 2：在当前姿态基础上，绕“自身 z 轴”转 100°
  q_step_z = axis_angle_to_quat("z", target_z)
  q3 = quat_mul(q2, q_step_z)
  q3 /= np.linalg.norm(q3)
  print("阶段2（自身 z 轴 100°）后的 q3 =", q3)

  # 预先把四个阶段的姿态排好：0=初始,1,2,3
  quats = [q0, q1, q2, q3]

  # 5. 用 viewer 按阶段展示，每个阶段停留一段时间
  stage = 0
  hold_time = 1.5  # 每个阶段停留秒数
  last_switch = time.time()

  with viewer.launch_passive(model, data) as v:
    while v.is_running():
      now = time.time()
      # 到时间就切换到下一阶段
      if now - last_switch > hold_time and stage < 3:
        stage += 1
        last_switch = now

      # 使用当前阶段的四元数
      data.qpos[3:7] = quats[stage]
      mujoco.mj_forward(model, data)

      v.sync()

if __name__ == "__main__":
    main()