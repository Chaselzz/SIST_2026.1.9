import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import sys

# 隐藏tkinter主窗口（仅用对话框）
root = tk.Tk()
root.withdraw()

xml_path = 'SIST_SI100B_RoboWriter-main\\models\\universal_robots_ur5e\\scene.xml'
simend = 60  # 适当减少总时间，但仍然保持可写性
print_camera_config = 0

# 全局变量
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0
target_qpos = None  # 新增：传递IK计算的目标关节角度

# -------------------------- 修改：字符串处理相关配置 --------------------------
CHAR_SPACING = 0.08   # 修改：无空格时两个字的默认间距（从0.2缩小到0.08）
LINE_SPACING = 0.35   # 行之间的垂直间距
AVERAGE_CHAR_WIDTH = 0.15  # 新增：平均字符宽度，用于计算空格大小

# -------------------------- 修改：抬笔/落笔核心配置 --------------------------
GAP_THRESHOLD = 0.02  # 增加间隙阈值，适应更大字体间距
PEN_DOWN_Z = 0.1       # 落笔高度（生成笔迹）
PEN_UP_Z = 0.2         # 抬笔高度（取消笔迹）
pen_state = "up"       # 初始状态：抬笔

# 修改：书写区域偏移配置
WRITING_AREA_X_OFFSET = 0.2   # 整体向右偏移，避免机械臂自身遮挡
WRITING_AREA_Y_OFFSET = 0.15  # 整体向上偏移
SCALE_FACTOR = 0.25           # 扩大为原来的2倍

# -------------------------- 新增：球面书写参数 --------------------------
# 球面方程：(x-0)² + (y-0.35)² + (z-1.3)² = R², 且 z ≤ 0.1
SPHERE_CENTER = np.array([0.0, 0.35, 1.3])  # 球心
SPHERE_RADIUS = 1.2  # 球半径 (1.3 - 0.1 = 1.2)
SPHERE_WRITING_HEIGHT = 0.1  # 球面书写高度限制
LIFT_HEIGHT = 0.4  # 悬停高度，高于抬笔高度，确保不会产生笔迹

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

# 修改：重构controller回调，符合MuJoCo规范
def controller(model, data):
    global target_qpos
    if target_qpos is not None:
        data.ctrl[:] = target_qpos  # 控制指令仅在回调中赋值

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

# -------------------------- 新增：字符串输入和字符坐标组合功能 --------------------------
def get_input_string():
    """获取用户输入的字符串"""
    # 提供示例选项，方便用户测试
    examples = [
        "12 3 4",
        "1,2,34",
        "你好世界",
        "Hello World",
        "测试\n换行",
        "机械臂书写测试"
    ]
    
    # 创建选择对话框
    choice = simpledialog.askinteger("选择输入方式", 
                                     "请选择:\n1. 输入自定义字符串\n2. 使用示例字符串\n\n输入 1 或 2:",
                                     minvalue=1, maxvalue=2, initialvalue=1)
    
    if choice == 1:
        input_string = simpledialog.askstring("输入字符串", 
                                              "请输入要书写的字符串（支持中文、英文、数字、符号）:\n"
                                              "注意：使用空格分隔单词，使用\\n换行\n"
                                              "示例：\"12 3 4\" 或 \"1,2,34\"",
                                              parent=root)
        if not input_string:
            # 如果没有输入，使用默认测试字符串
            input_string = "机械臂测试"
            print(f"【警告】未输入字符串，使用默认值: {input_string}")
    else:
        # 显示示例选择
        example_str = "\n".join([f"{i+1}. {examples[i]}" for i in range(len(examples))])
        example_choice = simpledialog.askinteger("选择示例", 
                                                f"请选择示例字符串:\n{example_str}",
                                                minvalue=1, maxvalue=len(examples), 
                                                initialvalue=1)
        input_string = examples[example_choice-1]
        print(f"【信息】使用示例字符串: {input_string}")
    
    return input_string

def select_coordinate_directory():
    """选择坐标文件所在的目录"""
    dir_path = filedialog.askdirectory(
        title="选择坐标文件目录（包含contour_*.txt文件）"
    )
    if not dir_path:
        # 如果没有选择，尝试使用程序目录下的character_contours文件夹
        program_dir = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(program_dir, "character_contours")
        if os.path.exists(dir_path):
            print(f"【信息】使用默认目录: {dir_path}")
        else:
            raise FileNotFoundError("未选择目录且默认目录不存在")
    return dir_path

def find_coordinate_file(char, directory):
    """
    根据字符在目录中查找对应的坐标文件
    
    参数:
        char: 要查找的字符
        directory: 坐标文件目录
    
    返回:
        坐标文件路径，如果没有找到则返回None
    """
    # 尝试几种可能的文件名格式
    patterns = [
        f"contour_{char}_*.txt",           # 格式: contour_中_20240101_120000.txt
        f"contour_{char}.txt",             # 格式: contour_中.txt
        f"contour_U{ord(char):04X}_*.txt", # 格式: contour_U4E2D_20240101_120000.txt
    ]
    
    import glob
    for pattern in patterns:
        files = glob.glob(os.path.join(directory, pattern))
        if files:
            # 返回最新创建的文件
            files.sort(key=os.path.getctime, reverse=True)
            return files[0]
    
    return None

def load_character_coords(file_path, x_offset=0, y_offset=0, scale_factor=1.0):
    """
    读取单个字符的坐标文件，并应用偏移和缩放
    
    参数:
        file_path: 坐标文件路径
        x_offset: x方向偏移量
        y_offset: y方向偏移量
        scale_factor: 缩放因子
    
    返回:
        偏移和缩放后的坐标点列表
    """
    coords = []
    read_data = False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "格式: X坐标 Y坐标":
                    read_data = True
                    continue
                if read_data and line.startswith("------------------------------------------------------------"):
                    continue
                if read_data and line:
                    try:
                        x, y = map(float, line.split())
                        # 先缩放，后偏移
                        x_scaled = x * scale_factor
                        y_scaled = y * scale_factor
                        coords.append([x_scaled + x_offset, y_scaled + y_offset])
                    except ValueError:
                        continue
        
        print(f"   ✓ 加载 '{os.path.basename(file_path)}': {len(coords)} 个点")
        return np.array(coords)
    
    except Exception as e:
        print(f"   ✗ 加载文件失败: {e}")
        return np.array([])

def combine_multiple_characters(input_string, coord_dir, scale_factor=1.0):
    """
    组合多个字符的坐标点，形成连续轨迹（支持缩放）
    
    参数:
        input_string: 输入的字符串
        coord_dir: 坐标文件目录
        scale_factor: 缩放因子
    
    返回:
        组合后的所有坐标点
    """
    print(f"【开始组合字符串】'{input_string}'")
    print(f"【坐标目录】{coord_dir}")
    print(f"【缩放因子】{scale_factor}")
    
    all_coords = []
    current_x = 0  # 当前x位置
    current_y = 0  # 当前y位置（用于处理多行）
    
    # 计算实际间距
    char_spacing_scaled = CHAR_SPACING * scale_factor
    avg_char_width_scaled = AVERAGE_CHAR_WIDTH * scale_factor
    
    print(f"【字符间距】{char_spacing_scaled:.3f} (缩放后)")
    print(f"【空格大小】{avg_char_width_scaled:.3f} (约1个字符宽度)")
    
    # 遍历字符串中的每个字符
    for i, char in enumerate(input_string):
        print(f"\n处理第 {i+1}/{len(input_string)} 个字符: '{char}' (ASCII: {ord(char)})")
        
        if char == ' ':
            # 空格：移动一个字的宽度
            current_x += avg_char_width_scaled
            print(f"   → 空格，移动 {avg_char_width_scaled:.3f} 单位（1个字符宽度）")
            continue
        
        if char == '\n':
            # 换行符：移动到下一行开头
            current_y -= LINE_SPACING * scale_factor  # 向下移动
            current_x = 0
            print(f"   → 换行，移动到新行 (y={current_y:.3f})")
            continue
        
        # 查找字符对应的坐标文件
        coord_file = find_coordinate_file(char, coord_dir)
        
        if coord_file:
            # 加载并应用偏移和缩放
            char_coords = load_character_coords(coord_file, current_x, current_y, scale_factor)
            
            if len(char_coords) > 0:
                # 添加到总坐标列表
                all_coords.extend(char_coords.tolist())
                
                # 更新当前位置（基于字符的宽度）
                if len(char_coords) > 0:
                    char_min_x = np.min(char_coords[:, 0])
                    char_max_x = np.max(char_coords[:, 0])
                    char_width = char_max_x - char_min_x
                    current_x = char_max_x + char_spacing_scaled  # 使用缩小的字符间距
                    
                    print(f"   → 字符宽度: {char_width:.3f}, 移动到 x={current_x:.3f}")
            else:
                print(f"   → 警告: 坐标文件为空，跳过字符 '{char}'")
        else:
            print(f"   → 警告: 未找到字符 '{char}' 的坐标文件，跳过")
    
    # 转换为numpy数组
    if all_coords:
        combined_coords = np.array(all_coords)
        print(f"\n【组合完成】总点数: {len(combined_coords)}")
        print(f"【覆盖范围】x: [{np.min(combined_coords[:, 0]):.3f}, {np.max(combined_coords[:, 0]):.3f}]")
        print(f"          y: [{np.min(combined_coords[:, 1]):.3f}, {np.max(combined_coords[:, 1]):.3f}]")
        
        # 应用整体偏移
        combined_coords[:, 0] += WRITING_AREA_X_OFFSET
        combined_coords[:, 1] += WRITING_AREA_Y_OFFSET
        
        print(f"【应用偏移】x偏移: {WRITING_AREA_X_OFFSET}, y偏移: {WRITING_AREA_Y_OFFSET}")
        print(f"【最终范围】x: [{np.min(combined_coords[:, 0]):.3f}, {np.max(combined_coords[:, 0]):.3f}]")
        print(f"          y: [{np.min(combined_coords[:, 1]):.3f}, {np.max(combined_coords[:, 1]):.3f}]")
        
        return combined_coords
    else:
        raise ValueError("没有成功加载任何字符的坐标点")

# -------------------------- 2. 重构插值函数（新增抬笔/落笔逻辑） --------------------------
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
    s = np.clip(t / t_total, 0.0, 1.0)  # 限制插值因子在0~1之间
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
    # 二次贝塞尔公式：B(s) = (1-s)²*q0 + 2*(1-s)*s*q1 + s²*q2
    return (1-s)**2 * q0 + 2*(1-s)*s * q1 + s**2 * q2

def generate_smooth_curve(discrete_points_xy, t_total, interpolate_type="bezier"):
    """
    重构：生成平滑曲线（新增间断判断+抬笔/落笔z轴控制）
    调整：保持足够多的插值点以保证轨迹连续性
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
    
    # 适当减小时间步长，增加插值点密度
    time_step = 0.01  # 回到0.01，保证足够多的插值点
    
    pen_state_log = []  # 记录抬笔/落笔状态日志
    
    # 仅对非常密集的离散点进行轻微下采样，保留大部分点
    if n_discrete > 2000:  # 只有点非常多时才下采样
        downsampling_factor = max(1, n_discrete // 1500)  # 目标约1500个离散点
        if downsampling_factor > 1:
            sampled_indices = list(range(0, n_discrete, downsampling_factor))
            # 确保包含最后一个点
            if sampled_indices[-1] != n_discrete - 1:
                sampled_indices.append(n_discrete - 1)
            discrete_points_xy = discrete_points_xy[sampled_indices]
            n_discrete = len(discrete_points_xy)
            print(f"【下采样】原始点 {n_discrete} 个点")
    
    print(f"【开始生成曲线】使用 {n_discrete} 个离散点")
    
    # 遍历所有相邻离散点对，逐段判断是否插值+设置z轴
    for seg_idx in range(n_discrete - 1):
        # 提取当前段的x,y（补充z轴：默认抬笔）
        q0_xy = discrete_points_xy[seg_idx]
        q1_xy = discrete_points_xy[seg_idx + 1]
        q0 = np.array([q0_xy[0], q0_xy[1], PEN_UP_Z])  # 初始抬笔
        q1 = np.array([q1_xy[0], q1_xy[1], PEN_UP_Z])
        
        # 核心：计算两点距离，判断是否间断
        distance = np.linalg.norm(q1_xy - q0_xy)
        
        # 调试信息：显示距离
        if seg_idx % 50 == 0:  # 每50段显示一次，避免输出太多
            print(f"  段{seg_idx+1}: 距离={distance:.4f}, 阈值={GAP_THRESHOLD}")
        
        if distance > GAP_THRESHOLD:
            # 间断点：抬笔（z=PEN_UP_Z），不插值，快速跳过
            pen_state_log.append("up")
            # 仅保留起点（抬笔状态），推进时间（快速移动，避免停留）
            t_smooth.append(current_time)
            pos_smooth.append(q0)
            current_time += time_step * 2  # 抬笔时快速移动
            continue
        
        # 非间断点：落笔（z=PEN_DOWN_Z），正常插值
        pen_state_log.append("down")
        # 重置z轴为落笔高度
        q0[2] = PEN_DOWN_Z
        q1[2] = PEN_DOWN_Z
        
        # 根据距离计算插值时间，确保足够多的插值点
        # 基础插值时间 = 距离 × 比例因子，确保最小时间
        seg_t_total = max(time_step * 10, distance * 20)
        
        # 生成该段的时间序列
        # 确保每段至少有3个点，距离越大点越多
        num_points = max(3, int(seg_t_total / time_step))
        seg_t = np.linspace(0, seg_t_total, num_points)
        
        # 生成该段的插值点（含落笔z轴）
        for t_in_seg in seg_t:
            if interpolate_type == "linear":
                pos = LinearInterpolate(q0, q1, t_in_seg, seg_t_total)
            elif interpolate_type == "bezier":
                # 二次贝塞尔：用中点作为控制点（z轴保持落笔）
                # 简化贝塞尔曲线，使用更接近原始线段的控制点
                q_ctrl_xy = (q0_xy + q1_xy) / 2
                # 使用更小的偏移量，使曲线更贴近原始线段
                direction = q1_xy - q0_xy
                if np.linalg.norm(direction) > 0.001:
                    normal = np.array([-direction[1], direction[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-6)
                    # 使用更小的偏移量，使曲线更贴近原始线段
                    q_ctrl_xy = q_ctrl_xy + normal * distance * 0.05  # 从0.1减小到0.05
                
                q_ctrl = np.array([q_ctrl_xy[0], q_ctrl_xy[1], PEN_DOWN_Z])
                pos = QuadBezierInterpolate(q0, q_ctrl, q1, t_in_seg, seg_t_total)
            
            # 添加到平滑序列
            t_smooth.append(current_time + t_in_seg)
            pos_smooth.append(pos)
        
        # 更新当前时间
        current_time += seg_t_total
        
        # 每100段显示一次进度
        if seg_idx % 100 == 0:
            print(f"  进度: {seg_idx+1}/{n_discrete-1}段，已生成 {len(pos_smooth)} 个点")
    
    # 添加最后一个点（保证轨迹完整，默认落笔）
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
    
    # 不再对最终点进行下采样，保持足够密度
    # 输出状态统计
    up_count = pen_state_log.count("up")
    down_count = pen_state_log.count("down")
    print(f"\n【平滑曲线生成完成】")
    print(f"  - 离散点 {n_discrete} → 平滑点 {len(pos_smooth)}")
    print(f"  - 抬笔段数: {up_count} | 落笔段数: {down_count}")
    print(f"  - 落笔高度: {PEN_DOWN_Z}m | 抬笔高度: {PEN_UP_Z}m")
    print(f"  - 间隙阈值: {GAP_THRESHOLD}m | 时间步长: {time_step}s")
    
    return t_smooth, pos_smooth

# -------------------------- 新增：球面投影函数 --------------------------
def project_to_sphere(flat_points, sphere_center=SPHERE_CENTER, sphere_radius=SPHERE_RADIUS, max_z=SPHERE_WRITING_HEIGHT):
    """
    将平面点投影到球面内表面
    
    参数:
        flat_points: N×3的numpy数组，平面上的点 [x,y,z]
        sphere_center: 球心坐标 [cx, cy, cz]
        sphere_radius: 球半径
        max_z: z轴最大高度限制
    
    返回:
        投影到球面上的点
    """
    print(f"\n【开始球面投影】")
    print(f"  球心: {sphere_center}")
    print(f"  半径: {sphere_radius}")
    print(f"  z轴限制: z ≤ {max_z}")
    
    projected_points = []
    
    for i, point in enumerate(flat_points):
        # 获取平面点坐标（忽略原来的z轴高度，使用平面高度）
        x_flat, y_flat, z_original = point
        
        # 平面点映射到球面：从球心到平面点的射线与球面的交点
        # 平面点相对于球心的向量
        vector_to_point = np.array([x_flat, y_flat, 0]) - sphere_center
        
        # 归一化向量
        norm_vector = np.linalg.norm(vector_to_point)
        if norm_vector > 0:
            unit_vector = vector_to_point / norm_vector
        else:
            unit_vector = np.array([0, 0, -1])  # 如果点在球心正下方
            
        # 计算射线与球面的交点
        # 解方程: |sphere_center + t*unit_vector - sphere_center| = sphere_radius
        # 简化: t = sphere_radius (因为单位向量长度为1)
        t = sphere_radius
        
        # 计算球面上的点
        sphere_point = sphere_center + t * unit_vector
        
        # 确保z轴高度限制
        if sphere_point[2] > max_z:
            # 如果z超过限制，重新计算交点，使得z = max_z
            # 解方程: sphere_center[2] + t*unit_vector[2] = max_z
            t = (max_z - sphere_center[2]) / unit_vector[2]
            sphere_point = sphere_center + t * unit_vector
            
            # 确保点在球面上
            distance_from_center = np.linalg.norm(sphere_point - sphere_center)
            if distance_from_center > sphere_radius:
                # 如果不在球面上，缩放回球面
                sphere_point = sphere_center + (sphere_radius / distance_from_center) * (sphere_point - sphere_center)
        
        # 根据原始点的z轴高度决定是否抬笔
        # 如果原始点的z是抬笔高度，则将球面上的点向外偏移（法线方向）
        if z_original >= PEN_UP_Z - 0.001:  # 抬笔状态
            # 沿着法线方向向外偏移
            normal_vector = (sphere_point - sphere_center) / np.linalg.norm(sphere_point - sphere_center)
            sphere_point = sphere_point + normal_vector * 0.05  # 向外偏移5cm
        
        projected_points.append(sphere_point)
        
        # 每1000个点显示一次进度
        if i % 1000 == 0 and i > 0:
            print(f"  进度: {i}/{len(flat_points)} 个点已投影")
    
    projected_points = np.array(projected_points)
    
    print(f"【球面投影完成】")
    print(f"  原始点范围: x[{np.min(flat_points[:, 0]):.3f}, {np.max(flat_points[:, 0]):.3f}]")
    print(f"              y[{np.min(flat_points[:, 1]):.3f}, {np.max(flat_points[:, 1]):.3f}]")
    print(f"  投影后范围: x[{np.min(projected_points[:, 0]):.3f}, {np.max(projected_points[:, 0]):.3f}]")
    print(f"              y[{np.min(projected_points[:, 1]):.3f}, {np.max(projected_points[:, 1]):.3f}]")
    print(f"              z[{np.min(projected_points[:, 2]):.3f}, {np.max(projected_points[:, 2]):.3f}]")
    
    return projected_points

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
window = glfw.create_window(1920, 1080, "UR5e 平滑曲线书写仿真（球面书写）", None, None)
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

# Camera config（沿用你的配置）
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
traj_points = []  # 仅记录落笔状态的轨迹（笔迹）
MAX_TRAJ = 5e5
LINE_RGBA = np.array([1.0, 0.0, 0.0, 1.0])  # 笔迹颜色（红色）

# -------------------------- 3. 加载字符串并生成平滑曲线 --------------------------
try:
    # 3.1 获取用户输入的字符串
    input_string = get_input_string()
    print(f"【输入字符串】'{input_string}'")
    print(f"【字符数】{len(input_string)}")
    
    # 3.2 选择坐标文件目录
    coord_dir = select_coordinate_directory()
    
    # 3.3 组合多个字符的坐标点（应用缩放因子）
    discrete_points_xy = combine_multiple_characters(
        input_string, 
        coord_dir, 
        scale_factor=SCALE_FACTOR  # 应用扩大2倍
    )
    
    # 3.4 生成平滑曲线（含抬笔/落笔z轴控制）
    # 询问用户选择插值方式
    choice = simpledialog.askstring("选择插值方式", 
                                    "请选择插值方式:\n1. 线性插值（更快、更直接）\n2. 贝塞尔曲线（更平滑、更自然）\n\n输入 1 或 2:",
                                    initialvalue="2")
    
    interpolate_type = "bezier" if choice == "2" else "linear"
    print(f"【插值方式】{interpolate_type}")
    
    # 询问用户是否需要调整速度
    speed_choice = simpledialog.askstring("速度设置", 
                                         "请选择书写速度:\n1. 快速（适合短字符串）\n2. 中速（平衡）\n3. 慢速（适合长字符串）\n\n输入 1, 2 或 3:",
                                         initialvalue="2")
    
    # 根据选择调整总时间
    if speed_choice == "1":
        t_total = simend - 2  # 快速
        print("【速度设置】快速")
    elif speed_choice == "3":
        t_total = simend * 1.5 - 2  # 慢速，增加50%时间
        print("【速度设置】慢速")
    else:
        t_total = simend - 2  # 中速
        print("【速度设置】中速")
    
    t_smooth, smooth_curve_points = generate_smooth_curve(
        discrete_points_xy, 
        t_total=t_total, 
        interpolate_type=interpolate_type
    )
    print(f"【平滑曲线生成成功】最终平滑点数量: {len(smooth_curve_points)}")
    
    # -------------------------- 新增：询问是否进行球面投影 --------------------------
    sphere_choice = simpledialog.askstring("选择书写表面", 
                                          "请选择书写表面:\n1. 平面书写\n2. 球面书写\n\n"
                                          "球面方程: (x-0)² + (y-0.35)² + (z-1.3)² = 1.2², 且 z ≤ 0.1\n"
                                          "输入 1 或 2:",
                                          initialvalue="1")
    
    use_sphere = (sphere_choice == "2")
    
    if use_sphere:
        print("【书写表面】球面")
        # 应用球面投影
        smooth_curve_points = project_to_sphere(smooth_curve_points)
    else:
        print("【书写表面】平面")
    
    # 计算实际的间距值（缩放后）
    char_spacing_scaled = CHAR_SPACING * SCALE_FACTOR
    avg_char_width_scaled = AVERAGE_CHAR_WIDTH * SCALE_FACTOR
    
    # 显示预览信息
    preview_info = f"""
    字符串: '{input_string}'
    字符数: {len(input_string)}
    书写表面: {'球面' if use_sphere else '平面'}
    插值方式: {interpolate_type}
    缩放因子: {SCALE_FACTOR} (扩大2倍)
    """
    
    if not use_sphere:
        preview_info += f"""
    偏移量: X={WRITING_AREA_X_OFFSET}, Y={WRITING_AREA_Y_OFFSET}
    
    间距设置（缩放后）:
      无空格字符间距: {char_spacing_scaled:.3f}
      空格大小: {avg_char_width_scaled:.3f} (1个字符宽度)
      行间距: {LINE_SPACING * SCALE_FACTOR:.3f}
    """
    else:
        preview_info += f"""
    球面参数:
      球心: {SPHERE_CENTER}
      半径: {SPHERE_RADIUS}
      z轴限制: z ≤ {SPHERE_WRITING_HEIGHT}
    """
    
    preview_info += f"""
    离散点数: {len(discrete_points_xy)}
    平滑点数: {len(smooth_curve_points)}
    总仿真时间: {simend}秒
    有效书写时间: {t_total:.1f}秒
    
    工作空间范围:
      X: [{np.min(smooth_curve_points[:, 0]):.3f}, {np.max(smooth_curve_points[:, 0]):.3f}]
      Y: [{np.min(smooth_curve_points[:, 1]):.3f}, {np.max(smooth_curve_points[:, 1]):.3f}]
      Z: [{np.min(smooth_curve_points[:, 2]):.3f}, {np.max(smooth_curve_points[:, 2]):.3f}]
    """
    
    if not use_sphere:
        preview_info += f"""
    优化说明: 
      - 无空格时：字符间距较小 ({char_spacing_scaled:.3f})
      - 有空格时：空格占1个字符宽度 ({avg_char_width_scaled:.3f})
      - 起笔位置更靠右 (X={WRITING_AREA_X_OFFSET})
      - 避免机械臂自身遮挡
      - 字体扩大2倍，提升清晰度
    """
    else:
        preview_info += f"""
    球面书写说明:
      - 在球面内表面书写，球心在 {SPHERE_CENTER}
      - 球半径: {SPHERE_RADIUS}
      - 书写区域限制在 z ≤ {SPHERE_WRITING_HEIGHT}
      - 抬笔/落笔逻辑适配球面法线方向
    """
    
    preview_info += "\n\n点击确定开始仿真..."
    
    messagebox.showinfo("准备开始", preview_info)
    
except Exception as e:
    print(f"❌ 轨迹加载失败: {e}")
    messagebox.showerror("错误", f"轨迹加载失败: {e}")
    glfw.terminate()
    sys.exit(1)

# -------------------------- 新增：完成书写后抬笔悬停变量 --------------------------
post_writing_lifted = False  # 标记是否已经完成抬笔悬停动作

# -------------------------- 4. 仿真主循环（抬笔/落笔+笔迹控制+抬笔悬停） --------------------------
while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        # -------------------------- 核心1：抬笔/落笔状态判断 --------------------------
        # 记录末端轨迹（仅落笔状态生成笔迹）
        mj_end_eff_pos = data.site_xpos[0]
        
        # 根据书写表面调整状态判断
        if use_sphere:
            # 球面书写：根据到球面的距离判断
            # 计算到球心的距离
            dist_to_center = np.linalg.norm(mj_end_eff_pos - SPHERE_CENTER)
            # 如果接近球面（在球面附近±0.01范围内），认为是落笔
            if abs(dist_to_center - SPHERE_RADIUS) <= 0.01:
                if pen_state != "down":
                    print(f"【状态切换】{data.time:.2f}s 落笔（距球面 {abs(dist_to_center - SPHERE_RADIUS):.3f}）")
                    pen_state = "down"
                traj_points.append(mj_end_eff_pos.copy())
            else:
                if pen_state != "up":
                    print(f"【状态切换】{data.time:.2f}s 抬笔（距球面 {abs(dist_to_center - SPHERE_RADIUS):.3f}）")
                    pen_state = "up"
        else:
            # 平面书写：根据z轴高度判断
            if mj_end_eff_pos[2] <= PEN_DOWN_Z + 0.001:  # 加小余量避免浮点误差
                if pen_state != "down":
                    print(f"【状态切换】{data.time:.2f}s 落笔（z={mj_end_eff_pos[2]:.3f}）")
                    pen_state = "down"
                traj_points.append(mj_end_eff_pos.copy())
            else:
                if pen_state != "up":
                    print(f"【状态切换】{data.time:.2f}s 抬笔（z={mj_end_eff_pos[2]:.3f}）")
                    pen_state = "up"
        
        # 限制轨迹长度，避免内存溢出
        if len(traj_points) > MAX_TRAJ:
            traj_points.pop(0)
        
        # 获取当前关节角度
        cur_q_pos = data.qpos.copy()
        
        # -------------------------- 核心2：按时间采样平滑曲线点（含z轴） --------------------------
        X_ref = smooth_curve_points[-1].copy()  # 默认最后一个点
        if data.time < t_total:
            # 找到当前时间对应的平滑曲线点（线性插值匹配时间）
            X_ref = np.array([
                np.interp(data.time, t_smooth, smooth_curve_points[:, 0]),
                np.interp(data.time, t_smooth, smooth_curve_points[:, 1]),
                np.interp(data.time, t_smooth, smooth_curve_points[:, 2])  # 动态z轴（抬笔/落笔）
            ])
        
        # -------------------------- 新增：完成书写后抬笔悬停 --------------------------
        # 当书写时间结束后，执行抬笔悬停动作
        if data.time >= t_total and not post_writing_lifted:
            # 获取当前末端执行器位置
            current_pos = data.site_xpos[0].copy()
            
            if use_sphere:
                # 球面书写：沿着球面法线方向抬笔
                # 计算法线方向（从球心指向当前位置）
                normal_vector = (current_pos - SPHERE_CENTER)
                norm_normal = np.linalg.norm(normal_vector)
                if norm_normal > 0:
                    normal_vector = normal_vector / norm_normal
                
                # 设置目标点为当前位置沿法线方向向外移动
                X_ref = current_pos + normal_vector * 0.1  # 向外移动10cm
            else:
                # 平面书写：向上抬笔
                X_ref = np.array([
                    current_pos[0],        # x保持不变
                    current_pos[1],        # y保持不变
                    LIFT_HEIGHT            # z轴抬到悬停高度
                ])
            
            # 计算IK目标关节角度
            target_qpos = IK_controller(model, data, X_ref, cur_q_pos)
            
            # 检查是否已经到达悬停位置
            dist_to_target = np.linalg.norm(current_pos - X_ref)
            if dist_to_target <= 0.02:  # 2cm范围内认为到达
                print(f"【抬笔悬停完成】{data.time:.2f}s 机械臂已悬停")
                post_writing_lifted = True
                
                # 保持悬停状态，不再更新目标位置
                target_qpos = IK_controller(model, data, X_ref, cur_q_pos)
            
            # 打印抬笔悬停进度
            elif data.time % 0.5 < 0.01:  # 每0.5秒打印一次
                print(f"【抬笔悬停中】{data.time:.2f}s 距目标 {dist_to_target:.3f}m")
        
        # 计算IK目标关节角度（传递给controller回调）
        target_qpos = IK_controller(model, data, X_ref, cur_q_pos)
        
        # 原生仿真步进
        mj.mj_step(model, data)

    # 仿真超时退出（确保有足够时间完成抬笔悬停）
    if (data.time >= simend and post_writing_lifted):
        break

    # 渲染可视化
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    if (print_camera_config == 1):
        print('cam.azimuth = ',cam.azimuth,'\n','cam.elevation = ',cam.elevation,'\n','cam.distance = ',cam.distance)
        print('cam.lookat = np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # 更新场景
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    
    # -------------------------- 核心3：绘制落笔状态的笔迹 --------------------------
    # 绘制轨迹（小球可视化）：仅落笔段生成笔迹
    # 减少绘制密度，提高性能但保持连续性
    draw_step = max(1, len(traj_points) // 1000)  # 最多绘制1000个点
    for j in range(1, len(traj_points), draw_step):
        if j >= len(traj_points):
            break
            
        if scene.ngeom >= scene.maxgeom:
            break
            
        p1 = traj_points[j-1]
        p2 = traj_points[j]
        midpoint = (p1 + p2) / 2.0
        
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        geom.type = mj.mjtGeom.mjGEOM_SPHERE
        geom.rgba[:] = LINE_RGBA
        geom.size[:] = np.array([0.003, 0.003, 0.003])  # 稍微增大笔迹点，适应更大字体
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
print(f"书写的字符串: '{input_string}'")
print(f"书写表面: {'球面' if use_sphere else '平面'}")
print(f"总仿真时间: {data.time:.2f}s")
print(f"落笔轨迹点数量: {len(traj_points)}")

if not use_sphere:
    print(f"缩放因子: {SCALE_FACTOR} (扩大2倍)")
    print(f"偏移量: X={WRITING_AREA_X_OFFSET}, Y={WRITING_AREA_Y_OFFSET}")
    print(f"字符间距: {CHAR_SPACING} (缩放后: {CHAR_SPACING * SCALE_FACTOR:.3f})")
    print(f"空格大小: {AVERAGE_CHAR_WIDTH} (缩放后: {AVERAGE_CHAR_WIDTH * SCALE_FACTOR:.3f}, 1个字符宽度)")
    print(f"起笔位置: X={WRITING_AREA_X_OFFSET:.3f}, Y={WRITING_AREA_Y_OFFSET:.3f}")
    print(f"抬笔高度: {PEN_UP_Z}m | 落笔高度: {PEN_DOWN_Z}m | 悬停高度: {LIFT_HEIGHT}m")
    print(f"行间距: {LINE_SPACING}m")
    print(f"间隙阈值: {GAP_THRESHOLD}m")
else:
    print(f"球面参数:")
    print(f"  球心: {SPHERE_CENTER}")
    print(f"  半径: {SPHERE_RADIUS}")
    print(f"  z轴限制: z ≤ {SPHERE_WRITING_HEIGHT}")
    print(f"  缩放因子: {SCALE_FACTOR} (扩大2倍)")

print(f"插值方式: {interpolate_type}")
print("=================================================")

# 显示完成提示
completion_info = f"字符串: '{input_string}'\n仿真时间: {data.time:.2f}秒\n轨迹点数: {len(traj_points)}\n"

if not use_sphere:
    completion_info += f"书写表面: 平面\n缩放因子: {SCALE_FACTOR} (扩大2倍)\n起始位置: X={WRITING_AREA_X_OFFSET}, Y={WRITING_AREA_Y_OFFSET}\n字符间距: {CHAR_SPACING * SCALE_FACTOR:.3f} (无空格时)\n空格大小: {AVERAGE_CHAR_WIDTH * SCALE_FACTOR:.3f} (1个字符宽度)\n悬停高度: {LIFT_HEIGHT}m (完成书写后抬笔悬停)\n"
else:
    completion_info += f"书写表面: 球面\n球心: {SPHERE_CENTER}\n半径: {SPHERE_RADIUS}\nz轴限制: z ≤ {SPHERE_WRITING_HEIGHT}\n"

completion_info += f"插值方式: {interpolate_type}\n\n仿真已完成！"

messagebox.showinfo("仿真完成", completion_info)

glfw.terminate()