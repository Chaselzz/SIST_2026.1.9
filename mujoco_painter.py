import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys 

# 隐藏tkinter主窗口
root = tk.Tk()
root.withdraw()

xml_path = 'SIST_SI100B_RoboWriter-main\\models\\universal_robots_ur5e\\scene.xml'
simend = 20  # 缩短仿真时长，适配曲线运动
print_camera_config = 0

# 全局变量
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0
target_qpos = None  # 传递IK计算的目标关节角度

# 抬笔加落笔部分
GAP_THRESHOLD = 0.005  # 距离大于此值判定为间断，不插值    （防止笔画错误粘连）
PEN_DOWN_Z = 0.1       # 落笔
PEN_UP_Z = 0.2         # 抬笔
pen_state = "up"       # 初始状态：抬笔

# Helper function
def IK_controller(model, data, X_ref, q_pos):
    # Compute Jacobian
    position_Q = data.site_xpos[0]
    jacp = np.zeros((3, 6))
    mj.mj_jac(model, data, jacp, None, position_Q, 7)
    J = jacp.copy()
    Jinv = np.linalg.pinv(J)

    # Reference point
    X = position_Q.copy()
    dX = X_ref - X

    # Compute control input
    dq = Jinv @ dX
    return q_pos + dq

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

# controller回调
def controller(model, data):
    global target_qpos
    if target_qpos is not None:
        data.ctrl[:] = target_qpos 

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos
    if not button_left and not button_middle and not button_right:
        return
    width, height = glfw.get_window_size(window)
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)
    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

# 1. 保留原有坐标读取逻辑
def select_coordinate_file():
    """打开文件选择窗口，选择坐标数据文件"""
    file_path = filedialog.askopenfilename(
        title="选择坐标数据文件",
        filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
    )
    if not file_path:
        raise FileNotFoundError("未选择文件")
    return file_path

def load_character_coords(file_path):    #从坐标缓存文件中读取数据
    """读取新格式坐标文件（x,y），暂不补充z值（后续按抬笔/落笔动态设置）"""
    coords = []
    read_dat = False  # 标记是否开始读取坐标数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 找到坐标数据开始的标志
            if line == "格式: X坐标 Y坐标":
                read_dat = True
                continue
            # 跳过分隔线
            if read_dat and line.startswith("------------------------------------------------------------"):
                continue
            # 读取坐标数据
            if read_dat and line:
                try:
                    x, y = map(float, line.split())
                    coords.append([x, y])  # 仅保存后续动态设置
                except ValueError:
                    continue  # 跳过无效行
    coords = np.array(coords)
    print(f"【坐标加载成功】共读取 {len(coords)} 个离散点")
    return coords

# 2. 重构插值函数
def LinearInterpolate(q0, q1, t, t_total):
    """
    线性插值：从q0到q1，在时间t（0~t_total）内的位置
    :param q0: 起始点 [x,y,z]
    :param q1: 终止点 [x,y,z]
    :param t: 当前时间（0<=t<=t_total）
    :param t_total: 总插值时间
    :return: 插值后的位置 [x,y,z]
    """
    if t_total == 0:
        return q0
    s = np.clip(t / t_total, 0.0, 1.0)  # 插值因子(0~1)
    return q0 + s * (q1 - q0)

def QuadBezierInterpolate(q0, q1, q2, t, t_total):
    """
    二次贝塞尔插值（更平滑的曲线）：q0=起点, q1=控制点, q2=终点
    :param q0: 起始点 [x,y,z]
    :param q1: 控制点 [x,y,z]
    :param q2: 终止点 [x,y,z]
    :param t: 当前时间（0<=t<=t_total）
    :param t_total: 总插值时间
    :return: 插值后的位置 [x,y,z]
    """
    if t_total == 0:
        return q0
    s = np.clip(t / t_total, 0.0, 1.0)  # 插值因子（0~1）
    return (1-s)**2 * q0 + 2*(1-s)*s * q1 + s**2 * q2

def generate_smooth_curve(discrete_points_xy, t_total, interpolate_type="linear"):
    """
    重构：生成平滑曲线（新增间断判断+抬笔/落笔z轴控制）
    :param discrete_points_xy: 离散点阵 (N×2) 仅x,y
    :param t_total: 总运动时间（秒）
    :param interpolate_type: 插值类型 "linear"/"bezier"
    :return: 平滑曲线的时间序列 + 位置序列（含z轴抬笔/落笔）
    """
    global pen_state
    n_discrete = len(discrete_points_xy)
    if n_discrete < 2:
        raise ValueError("离散点数量至少为2才能生成曲线")
    
    # 初始化平滑序列
    t_smooth = []
    pos_smooth = []
    current_time = 0.0
    # 非间断段基础时间步长
    time_step = 0.01
    pen_state_log = []  # 记录抬笔/落笔状态日志
    
    # 遍历相邻离散点对，判断是否插值&设置z轴
    for seg_idx in range(n_discrete - 1):
        # 提取当前段的x,y(默认抬)
        q0_xy = discrete_points_xy[seg_idx]
        q1_xy = discrete_points_xy[seg_idx + 1]
        q0 = np.array([q0_xy[0], q0_xy[1], PEN_UP_Z])  # 初始抬笔
        q1 = np.array([q1_xy[0], q1_xy[1], PEN_UP_Z])
        
        # 核心：计算两点距离，判断是否间断
        distance = np.linalg.norm(q1_xy - q0_xy)
        print(f"【段{seg_idx+1}】点{seg_idx}→点{seg_idx+1} 距离: {distance:.6f} | 阈值: {GAP_THRESHOLD}")
        
        if distance > GAP_THRESHOLD:
            # 间断点：抬笔（z=PEN_UP_Z），不插值，快速跳过
            print(f"   → 判定为间断，抬笔（z={PEN_UP_Z}），跳过插值")
            pen_state_log.append("up")
            # 仅保留起点（抬笔状态），推进时间（快速移动，避免停留）
            t_smooth.append(current_time)
            pos_smooth.append(q0)
            current_time += time_step * 5  # 抬笔时快速移动
            continue
        
        # 非间断点：落笔，正常插值
        pen_state_log.append("down")
        print(f"   → 判定为连续，落笔（z={PEN_DOWN_Z}），进行{interpolate_type}插值")
        # 重置z轴->落笔高度
        q0[2] = PEN_DOWN_Z
        q1[2] = PEN_DOWN_Z
        # 总插值时间 
        seg_t_total = min(time_step * 50, distance * 10)  # 距离越远，插值时间越长
        # 生成该段的时间序列
        seg_t = np.arange(0, seg_t_total, time_step)
        
        # 生成该段的插值点
        for t_in_seg in seg_t:
            if interpolate_type == "linear":
                pos = LinearInterpolate(q0, q1, t_in_seg, seg_t_total)
            elif interpolate_type == "bezier":
                # 二次贝塞尔：用中点作为控制点（z轴保持落笔）
                q_ctrl_xy = (q0_xy + q1_xy) / 2
                q_ctrl = np.array([q_ctrl_xy[0], q_ctrl_xy[1], PEN_DOWN_Z])
                pos = QuadBezierInterpolate(q0, q_ctrl, q1, t_in_seg, seg_t_total)
            
            t_smooth.append(current_time + t_in_seg)
            pos_smooth.append(pos)
        
        # 更新时间
        current_time += seg_t_total
    
    # 添加最后一个点（默认落笔）
    last_point_xy = discrete_points_xy[-1]
    last_point = np.array([last_point_xy[0], last_point_xy[1], PEN_DOWN_Z])
    t_smooth.append(current_time)
    pos_smooth.append(last_point)
    pen_state_log.append("down")
    
    # 归一化时间到总运动时长
    t_smooth = np.array(t_smooth)
    if t_smooth[-1] > 0:
        t_smooth = t_smooth / t_smooth[-1] * t_total
    
    pos_smooth = np.array(pos_smooth)
    # 输出状态统计
    up_count = pen_state_log.count("up")
    down_count = pen_state_log.count("down")
    print(f"\n【平滑曲线生成完成】")
    print(f"  - 离散点 {n_discrete} → 平滑点 {len(pos_smooth)}")
    print(f"  - 抬笔段数: {up_count} | 落笔段数: {down_count}")
    print(f"  - 落笔高度: {PEN_DOWN_Z}m | 抬笔高度: {PEN_UP_Z}m")
    return t_smooth, pos_smooth

# Get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# Init GLFW
glfw.init()
window = glfw.create_window(1920, 1080, "UR5e 平滑曲线书写仿真（抬笔/落笔）", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Initialize visualization
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Install callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Camera config
cam.azimuth = 85.66333333333347
cam.elevation = -35.33333333333329
cam.distance = 2.22
cam.lookat = np.array([-0.09343103051557476, 0.31359595076587915, 0.22170312166086661])

# Initialize controller
init_controller(model, data)
mj.set_mjcb_control(controller)

# Initial joint config
init_qpos = np.array([-1.6353559, -1.28588984, 2.14838487, -2.61087434, -1.5903009, -0.06818645])
data.qpos[:] = init_qpos
cur_q_pos = init_qpos.copy()
traj_points = []  # 仅记录落笔笔迹
MAX_TRAJ = 5e5
LINE_RGBA = np.array([1.0, 0.0, 0.0, 1.0])  # 笔迹颜色（红）

# 3. 加载坐标并生成平滑曲线
try:
    # 选择+读取离散坐标点（仅x,y）
    coords_file_path = select_coordinate_file()
    discrete_points_xy = load_character_coords(coords_file_path)
    
    # 生成平滑曲线
    t_total = simend - 2  # 预留2秒
    t_smooth, smooth_curve_points = generate_smooth_curve(
        discrete_points_xy, 
        t_total=t_total, 
        interpolate_type="bezier"  # 切换为"linear"可看线性插值效果
    )
    print(f"【平滑曲线生成成功】最终平滑点数量: {len(smooth_curve_points)}")
except Exception as e:
    print(f"轨迹加载失败: {e}")
    glfw.terminate()
    sys.exit(1)

# 4. 仿真主循环
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        # 抬笔/落笔状态判断
        # 末端轨迹 
        mj_end_eff_pos = data.site_xpos[0]
        # 仅当z轴≤落笔高度时生成笔记
        if mj_end_eff_pos[2] <= PEN_DOWN_Z + 0.001:
            if pen_state != "down":
                print(f"【状态切换】{data.time:.2f}s 落笔（z={mj_end_eff_pos[2]:.3f}）")
                pen_state = "down"
            traj_points.append(mj_end_eff_pos.copy())
        else:
            if pen_state != "up":
                print(f"【状态切换】{data.time:.2f}s 抬笔（z={mj_end_eff_pos[2]:.3f}）")
                pen_state = "up"
            # 抬笔状态：不记录轨迹，清空临时轨迹
            pass
        
        # 限制轨迹长度，避免内存溢出
        if len(traj_points) > MAX_TRAJ:
            traj_points.pop(0)
        
        # 获取当前关节角度
        cur_q_pos = data.qpos.copy()
        
        # 按时间采样平滑曲线点（含z轴）
        X_ref = smooth_curve_points[-1].copy()  # 默认最后一个点
        if data.time < t_total:
            # 找到当前时间对应平滑曲线点
            X_ref = np.array([
                np.interp(data.time, t_smooth, smooth_curve_points[:, 0]),
                np.interp(data.time, t_smooth, smooth_curve_points[:, 1]),
                np.interp(data.time, t_smooth, smooth_curve_points[:, 2])  # 动态z轴（抬笔/落笔）
            ])
        
        # IK目标关节角度
        target_qpos = IK_controller(model, data, X_ref, cur_q_pos)
        
        mj.mj_step(model, data)

    # 超时退出
    if (data.time >= simend):
        break

    # 可视化
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    if (print_camera_config == 1):
        print('cam.azimuth = ',cam.azimuth,'\n','cam.elevation = ',cam.elevation,'\n','cam.distance = ',cam.distance)
        print('cam.lookat = np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # 更新场景
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    
    # 仅绘制落笔状态的笔迹
    # 绘制轨迹：仅落笔段生成笔迹
    for j in range(1, len(traj_points)):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        p1 = traj_points[j-1]
        p2 = traj_points[j]
        midpoint = (p1 + p2) / 2.0
        geom.type = mj.mjtGeom.mjGEOM_SPHERE
        geom.rgba[:] = LINE_RGBA
        geom.size[:] = np.array([0.002, 0.002, 0.002])
        geom.pos[:] = midpoint
        geom.mat[:] = np.eye(3)
        geom.dataid = -1
        geom.segid = -1
        geom.objtype = 0
        geom.objid = 0
    
    # 渲染和交互
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

# 仿真结束：输出统计信息
print("\n==================== 仿真结束 ====================")
print(f"总仿真时间: {data.time:.2f}s")
print(f"落笔轨迹点数量: {len(traj_points)}")
print(f"抬笔高度: {PEN_UP_Z}m | 落笔高度: {PEN_DOWN_Z}m")
glfw.terminate()