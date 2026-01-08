from PIL import Image, ImageDraw
import json
import os
import tkinter as tk
from tkinter import filedialog

# ---------------------- 读取JSON坐标文件 ----------------------
def load_json_coords(json_path: str) -> list:
    """
    读取JSON文件，返回点阵坐标列表
    :param json_path: JSON文件路径
    :return: 坐标列表 [(x1,y1), (x2,y2), ...]
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            coords = json.load(f)
        # 校验坐标格式（避免无效JSON）
        if not isinstance(coords, list) or len(coords) == 0:
            raise ValueError("JSON文件中无有效坐标数据")
        # 确保坐标是数字类型
        coords = [(int(x), int(y)) for x, y in coords]
        return coords
    except FileNotFoundError:
        raise Exception(f"未找到文件：{json_path}")
    except json.JSONDecodeError:
        raise Exception(f"JSON文件格式错误：{json_path}")
    except Exception as e:
        raise Exception(f"读取JSON失败：{e}")

# ---------------------- 根据坐标绘制汉字（修复倒置+颜色浅） ----------------------
def draw_char_from_coords(
    coords: list,
    point_size: int = 8,  # 增大默认点大小（解决颜色浅）
    point_color: tuple = (0, 0, 0),  # 黑色（默认更醒目）
    bg_color: tuple = (255, 255, 255)  # 白色背景
) -> Image.Image:
    """
    根据点阵坐标绘制汉字（修复倒置+优化点显示）
    :param coords: 坐标列表
    :param point_size: 点的大小（像素），步长越大建议点越大
    :param point_color: 点的RGB颜色
    :param bg_color: 背景RGB颜色
    :return: 绘制好的PIL Image对象
    """
    # 计算坐标的最大/最小值，确定画布大小（留10像素边距）
    xs = [x for x, y in coords]
    ys = [y for x, y in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # 画布大小 = 坐标范围 + 20像素边距（避免点贴边）
    canvas_w = max_x - min_x + 20
    canvas_h = max_y - min_y + 20
    
    # 创建画布
    img = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 绘制每个点（偏移坐标，让汉字居中）
    offset_x = 10 - min_x
    offset_y = 10 - min_y
    for x, y in coords:
        # 改为绘制实心圆形点（解决颜色浅，视觉更集中）
        draw.ellipse(
            [
                (x + offset_x - point_size//2, y + offset_y - point_size//2),
                (x + offset_x + point_size//2, y + offset_y + point_size//2)
            ],
            fill=point_color,  # 实心填充（关键：解决颜色浅）
            outline=None       # 去掉轮廓，纯实心
        )
    
    # 修复图像倒置：先垂直翻转（上下镜像），再旋转180度（彻底修正方向）
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # 垂直翻转
    img = img.rotate(180, expand=True)          # 180度旋转（expand避免裁剪）
    
    return img

# ---------------------- 主程序（交互+绘制） ----------------------
def main():
    # 隐藏tkinter主窗口（仅用文件选择对话框）
    root = tk.Tk()
    root.withdraw()
    
    print("📌 汉字点阵还原绘制工具（修复版）")
    print("------------------------")
    
    # 步骤1：选择JSON文件
    print("请选择之前生成的点阵JSON文件（如：人_宋体_点阵覆盖坐标.json）")
    json_path = filedialog.askopenfilename(
        title="选择JSON坐标文件",
        filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
    )
    if not json_path:
        print("❌ 未选择文件，程序退出")
        return
    print(f"✅ 已选择文件：{os.path.basename(json_path)}")
    
    # 步骤2：读取坐标
    try:
        coords = load_json_coords(json_path)
        print(f"✅ 成功读取 {len(coords)} 个点阵坐标")
    except Exception as e:
        print(f"❌ 读取坐标失败：{e}")
        return
    
    # 步骤3：设置绘制参数
    try:
        point_size = int(input("\n请输入点的大小（像素，建议步长的1/2，如步长10则输5，默认8）：").strip())
        if point_size < 1:
            raise ValueError("点大小必须≥1")
    except ValueError:
        print("⚠️ 输入无效，使用默认点大小：8")
        point_size = 8
    
    # 可选：自定义点颜色（默认黑色更醒目）
    color_choice = input("\n是否自定义点颜色？(Y/N，默认黑色)：").strip().upper()
    if color_choice == "Y":
        try:
            r = int(input("  请输入R值（0-255）：").strip())
            g = int(input("  请输入G值（0-255）：").strip())
            b = int(input("  请输入B值（0-255）：").strip())
            if not (0<=r<=255 and 0<=g<=255 and 0<=b<=255):
                raise ValueError("颜色值需在0-255之间")
            point_color = (r, g, b)
        except ValueError:
            print("⚠️ 颜色输入无效，使用默认黑色")
            point_color = (0, 0, 0)
    else:
        point_color = (0, 0, 0)
    
    # 步骤4：绘制汉字
    try:
        print("\n🎨 正在绘制汉字...")
        img = draw_char_from_coords(
            coords=coords,
            point_size=point_size,
            point_color=point_color
        )
    except Exception as e:
        print(f"❌ 绘制失败：{e}")
        return
    
    # 步骤5：预览+保存图像
    # 自动生成保存文件名（和JSON文件同名，后缀为png）
    json_dir = os.path.dirname(json_path)
    json_name = os.path.splitext(os.path.basename(json_path))[0]
    save_path = os.path.join(json_dir, f"{json_name}_还原汉字_修正版.png")
    
    img.save(save_path)
    print(f"✅ 绘制完成！图像已保存到：{save_path}")
    
    # 预览图像（自动打开系统默认图片查看器）
    print("🖼️ 正在打开预览窗口...")
    img.show()

if __name__ == "__main__":
    main()