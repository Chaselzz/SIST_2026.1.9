###
# 基于MuJoCo的逆运动学实践
# SI100B 机器人编程
# 本代码基于MuJoCo模板代码修改，模板代码地址：https://github.com/pab47/pab47.github.io/tree/master.
# 日期：2025年12月
###

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import scipy as sp
target_file_path = "D:/skd20255333015/1_1/si100/project/lec3/SIST_SI100B_RoboWriter-main/scripts/lec3_kinematics/ik_code.py"  
with open(target_file_path, "r", encoding="utf-8") as f:
    ik_code = f.read()
    exec(ik_code)
xml_path = '../../universal_robots_ur5e/scene.xml' # xml文件路径（假设该文件与本代码文件在同一文件夹下）
simend = 100 # 仿真时长（秒）
print_camera_config = 0 # 设置为1时打印相机配置
                        # 这对初始化模型视角非常有用

# 用于回调函数的全局变量
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# 辅助函数
def random_cube_apex(
    cx, cy, cz,
    cube_size=0.08  # 立方体边长缩小以提升模拟精度

):
    """
    随机返回3D空间中立方体的一个顶点位置。

    参数
    ----------
    cube_size : 浮点数
        立方体的边长。
    center_range : 包含3个元组的元组
        立方体中心在x、y、z轴上的采样范围（注：原函数参数列表中无此参数，为文档字符串笔误）。

    返回值
    -------
    apex : 多维数组，形状为(3,)
        随机选择的立方体顶点的3D坐标。
    """
    # 随机立方体中心（注：实际使用传入的cx、cy、cz作为中心）
    center = np.array([cx, cy, cz])#作用是把「普通 Python 列表 / 元组」转换成「NumPy 数组」（比普通列表更高效，支持向量化运算）

    # 半边长
    h = cube_size / 2.0

    # 8个顶点的偏移量
    offsets = np.array([
        [ h,  h,  h],
        [ h,  h, -h],
        [ h, -h,  h],
        [ h, -h, -h],
        [-h,  h,  h],
        [-h,  h, -h],
        [-h, -h,  h],
        [-h, -h, -h],
    ])

    # 随机选择一个顶点
    apex_offset = offsets[np.random.randint(0, 7)]
    apex = center + apex_offset

    return apex

def init_controller(model,data):
    # 在此处初始化控制器。该函数仅在仿真开始时调用一次
    pass

def controller(model, data):
    # 在此处编写控制器逻辑。该函数在仿真过程中被持续调用
    pass

def keyboard(window, key, scancode, act, mods): #键盘输入逻辑代码
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:   #当你按下键盘上的「退格键（BACKSPACE）」时，会重置 MuJoCo 仿真的所有状态，让机械臂回到初始位置，同时同步更新模型的运动学数据
        mj.mj_resetData(model, data)  # 重置MuJoCo数据
        mj.mj_forward(model, data)    # 前向计算更新模型状态

def mouse_button(window, button, act, mods):#鼠标按键逻辑代码
    # 更新鼠标按键状态
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
#按住右键 → 移动相机；
#按住左键 → 旋转相机；
#按住中键 → 缩放相机
    # 更新鼠标位置
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):#鼠标移动逻辑代码
    # 计算鼠标位移并保存
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos      #上四行计算鼠标位移

    # 无按键按下时不执行任何操作
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # 获取当前窗口尺寸
    width, height = glfw.get_window_size(window)

    # 获取Shift键状态
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # 根据鼠标按键确定操作类型
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H  # 水平移动相机
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V  # 垂直移动相机
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H  # 水平旋转相机
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V  # 垂直旋转相机
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM  # 缩放相机

    # 移动相机
    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM  # 缩放操作
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

# 获取文件完整路径
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo数据结构初始化
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo模型（加载UR5e机械臂的XML配置）
data = mj.MjData(model)                # MuJoCo数据（存储模型的实时状态）
cam = mj.MjvCamera()                        # 抽象相机（控制仿真可视化视角）
opt = mj.MjvOption()                        # 可视化选项

# 初始化GLFW、创建窗口、设置OpenGL上下文、开启垂直同步
glfw.init()
window = glfw.create_window(1920, 1080, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# 初始化可视化数据结构
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)  # 场景（存储可视化几何信息）
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)  # 渲染上下文

# 注册GLFW鼠标和键盘回调函数
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# 相机配置示例（初始化UR5e机械臂的可视化视角）
cam.azimuth = -130  # 方位角
cam.elevation = -5  # 仰角
cam.distance =  2   # 相机到目标点的距离
cam.lookat =np.array([ 0.0 , 0.0 , 0.5 ])  # 相机注视点

# 初始化控制器
init_controller(model,data)

# 设置MuJoCo的控制回调函数
mj.set_mjcb_control(controller)

# 初始化机械臂初始位姿（home位姿）
key_qpos = model.key("home").qpos  # 从XML中读取home位姿的关节角度
q_home = key_qpos.copy()

############################################################
##  尝试修改以下数值以观察不同的机器人行为               ##
##  注意：数值需保持在机器人的工作空间范围内！            ##
############################################################
x_ref = 0.5    # 末端执行器参考位置x轴
y_ref = 0.4    # 末端执行器参考位置y轴
z_ref = 0.3    # 末端执行器参考位置z轴
phi_ref = 3.14 # 末端执行器参考姿态（绕x轴欧拉角）
theta_ref = 0  # 末端执行器参考姿态（绕y轴欧拉角）
psi_ref = 0    # 末端执行器参考姿态（绕z轴欧拉角）
############################################################

apex_ref = np.array([x_ref,y_ref,z_ref])  # 参考位置初始值
alter_flag = True  # 标记是否允许更新参考位置
is_pose_reached = False  # 新增：标记是否已达到目标位姿
pose_tolerance = 1e-3    # 新增：位姿偏差阈值 目的是在逼近某一精确度时停止后续模拟，防止抽搐


while not glfw.window_should_close(window):
    time_prev = data.time  # 记录上一帧的仿真时间

    while (data.time - time_prev < 1.0/60.0):  # 保证60Hz的仿真步长
        # 如果已达到目标位姿，直接跳过后续
        if is_pose_reached:
            data.time += 0.02  # 仅推进仿真时间
            continue
        
        # 获取当前末端执行器的位置和姿态
        position_Q = data.site_xpos[0]  # 末端执行器位置（site_xpos[0]对应末端site）
        mat_Q = data.site_xmat[0]       # 末端执行器姿态矩阵
        quat_Q = np.zeros(4)
        mj.mju_mat2Quat(quat_Q, mat_Q)  # 将姿态矩阵转换为四元数
        r_Q = sp.spatial.transform.Rotation.from_quat([quat_Q[1],quat_Q[2],quat_Q[3],quat_Q[0]])
        euler_Q = r_Q.as_euler('xyz')   # 将四元数转换为xyz欧拉角
        
        # 计算雅可比矩阵J
        # mj_jac函数参数说明：
        # void mj_jac(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr,
        #     const mjtNum point[3], int body);
        # jacp: 位置雅可比（3x6），jacr: 旋转雅可比（3x6），point: 计算雅可比的点，body: 关联的body ID（7对应UR5e末端body）
        jacp = np.zeros((3, 6))
        jacr = np.zeros((3, 6))
        mj.mj_jac(model, data, jacp, jacr, position_Q, 7)

        # 拼接完整雅可比矩阵并计算伪逆
        J = np.vstack((jacp, jacr))  # 6x6雅可比矩阵（位置+旋转）
        Jinv = np.linalg.pinv(J)     # 雅可比伪逆（解决奇异性问题）

        # 计算期望位姿与当前位姿的偏差dX
        # 每5秒更新一次参考位置（随机选择立方体顶点）
        if (int(data.time)%5==0) and (alter_flag==True) and (not is_pose_reached):
            apex_ref = random_cube_apex(x_ref,y_ref,z_ref)
            alter_flag = False  # 标记已更新，避免同一秒重复更新
        if (int(data.time)%5!=0):
            alter_flag = True   # 非5秒整数倍时允许更新
        
        # 构建参考位姿（位置+欧拉角）和当前位姿
        X_ref = np.array([apex_ref[0],apex_ref[1],apex_ref[2],phi_ref,theta_ref,psi_ref])
        X = np.concatenate((position_Q, euler_Q))
        dX = X_ref - X  # 位姿偏差

        # 判断位姿偏差是否小于阈值，若满足跳过后续
        if np.linalg.norm(dX) < pose_tolerance:
            is_pose_reached = True
            data.time += 0.02
            continue

        # 计算关节角度增量dq = 雅可比伪逆 * 位姿偏差
        dq = Jinv.dot(dX)

        # 更新关节角度并同步到MuJoCo数据
        q_home += dq
        data.qpos = q_home.copy()
        mj.mj_forward(model, data)  # 前向计算更新模型状态
        data.time += 0.02  # 仿真时间步进

    if (data.time>=simend):  # 达到仿真时长后退出循环
        break

    # 获取帧缓冲视口尺寸
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # 打印相机配置（用于初始化视角时参考）
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # 更新场景并渲染
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # 交换OpenGL缓冲区（因垂直同步会阻塞）
    glfw.swap_buffers(window)

    # 处理待处理的GUI事件，调用GLFW回调函数
    glfw.poll_events()

glfw.terminate()  # 终止GLFW，释放资源