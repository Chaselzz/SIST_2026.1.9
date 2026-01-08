import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# -------------------------- 核心：保留原有坐标读取逻辑 --------------------------
def select_coordinate_file():
    """打开文件选择窗口，选择坐标数据文件（兼容原有逻辑）"""
    # 隐藏tkinter主窗口
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="选择坐标数据文件 [Debug模式]",
        filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
        initialdir=os.path.dirname(os.path.abspath(__file__))  # 默认打开程序所在目录
    )
    
    if not file_path:
        raise FileNotFoundError("未选择任何文件，程序退出")
    return file_path

def load_character_coords(file_path):
    """
    读取坐标文件（仅提取x/y坐标，用于2D可视化）
    返回：x坐标数组, y坐标数组, 原始坐标列表（含z）
    """
    coords_xy = []  # 存储x/y坐标
    coords_xyz = [] # 存储x/y/z坐标（z固定为0.1，兼容原有逻辑）
    read_data = False  # 标记是否开始读取坐标数据
    
    print(f"\n📝 正在解析文件: {os.path.basename(file_path)}")
    print("-" * 60)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 先解析文件头信息（Debug用）
        for line in lines[:20]:  # 只看前20行表头
            line = line.strip()
            if line.startswith("字符:"):
                print(f"🔤 字符: {line.replace('字符:', '').strip()}")
            elif line.startswith("总点数:"):
                print(f"📊 声明点数: {line.replace('总点数:', '').strip()}")
            elif line.startswith("坐标范围:"):
                print(f"📍 坐标范围: {line.replace('坐标范围:', '').strip()}")
        
        # 读取坐标数据
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 找到坐标数据开始的标志
            if line == "格式: X坐标 Y坐标":
                read_data = True
                print(f"\n📌 坐标数据起始行: 第{line_num}行")
                continue
            
            # 跳过分隔线
            if read_data and line.startswith("-" * 50):
                continue
            
            # 读取有效坐标行
            if read_data and line:
                try:
                    x, y = map(float, line.split())
                    coords_xy.append([x, y])
                    coords_xyz.append([x, y, 0.1])  # z固定0.1，兼容原有逻辑
                except ValueError as e:
                    print(f"⚠️  第{line_num}行数据无效，跳过: {line} | 错误: {e}")
                    continue
    
    # 转换为numpy数组（方便计算和可视化）
    coords_xy = np.array(coords_xy)
    coords_xyz = np.array(coords_xyz)
    
    # Debug信息输出
    print(f"\n✅ 解析完成！")
    print(f"📊 实际读取点数: {len(coords_xy)} 个")
    print(f"📈 X坐标范围: [{coords_xy[:,0].min():.6f}, {coords_xy[:,0].max():.6f}]")
    print(f"📈 Y坐标范围: [{coords_xy[:,1].min():.6f}, {coords_xy[:,1].max():.6f}]")
    print(f"📏 X/Y均值: ({coords_xy[:,0].mean():.6f}, {coords_xy[:,1].mean():.6f})")
    print("-" * 60)
    
    return coords_xy[:,0], coords_xy[:,1], coords_xyz

# -------------------------- Debug可视化核心函数 --------------------------
def visualize_coords(x, y):
    """
    2D点阵可视化（支持缩放、平移、保存图像）
    功能：显示点阵分布 + 统计信息 + 网格 + 坐标范围标注
    """
    # 创建画布（大尺寸，方便查看细节）
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制点阵（核心可视化）
    scatter = ax.scatter(
        x, y, 
        s=3,          # 点大小（可调，越大越清晰）
        c='red',      # 点颜色（红色醒目，方便debug）
        alpha=0.8,    # 透明度（避免点重叠看不清）
        marker='.'    # 点样式（小点，密集显示）
    )
    
    # -------------------------- Debug增强配置 --------------------------
    # 1. 添加网格（方便定位坐标）
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)  # 网格在点下方
    
    # 2. 坐标轴标注（清晰显示坐标范围）
    ax.set_xlabel('X 坐标', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y 坐标', fontsize=12, fontweight='bold')
    ax.set_title(f'字符点阵可视化 [总点数: {len(x)}] | Debug模式', fontsize=14, fontweight='bold')
    
    # 3. 标注坐标极值点（Debug关键）
    x_min_idx = np.argmin(x)
    x_max_idx = np.argmax(x)
    y_min_idx = np.argmin(y)
    y_max_idx = np.argmax(y)
    
    # 标注极值点（红色大圆点）
    ax.scatter(x[x_min_idx], y[x_min_idx], s=50, c='blue', label=f'X最小 ({x[x_min_idx]:.6f})', zorder=5)
    ax.scatter(x[x_max_idx], y[x_max_idx], s=50, c='green', label=f'X最大 ({x[x_max_idx]:.6f})', zorder=5)
    ax.scatter(x[y_min_idx], y[y_min_idx], s=50, c='orange', label=f'Y最小 ({y[y_min_idx]:.6f})', zorder=5)
    ax.scatter(x[y_max_idx], y[y_max_idx], s=50, c='purple', label=f'Y最大 ({y[y_max_idx]:.6f})', zorder=5)
    
    # 4. 显示图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 5. 等比例显示（避免变形）
    ax.set_aspect('equal', adjustable='box')
    
    # 6. 添加统计信息文本框（右上角）
    stats_text = f"""
    点数: {len(x)}
    X范围: [{x.min():.6f}, {x.max():.6f}]
    Y范围: [{y.min():.6f}, {y.max():.6f}]
    X均值: {x.mean():.6f}
    Y均值: {y.mean():.6f}
    """
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # -------------------------- 交互功能 --------------------------
    plt.tight_layout()
    plt.show()
    
    # 询问是否保存图像（Debug存档用）
    save_choice = input("\n📸 是否保存可视化图像？(y/n): ").strip().lower()
    if save_choice == 'y':
        save_path = os.path.join(os.path.dirname(select_coordinate_file.__globals__['__file__']), f"debug_点阵_{os.path.basename(select_coordinate_file.__globals__['file_path']).replace('.txt', '.png')}")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图像已保存到: {save_path}")

# -------------------------- 主函数（Debug流程） --------------------------
def main():
    """Debug程序主流程：选择文件 → 解析坐标 → 可视化 → 输出Debug信息"""
    print("=" * 60)
    print("🎯 字符点阵Debug可视化工具")
    print("=" * 60)
    
    try:
        # 1. 选择坐标文件
        file_path = select_coordinate_file()
        
        # 2. 解析坐标数据
        x, y, _ = load_character_coords(file_path)
        
        # 3. 输出前10个坐标（快速Debug）
        print("\n🔍 前10个坐标示例:")
        print("序号 | X坐标       | Y坐标")
        print("-" * 30)
        for i in range(min(10, len(x))):
            print(f"{i+1:3d} | {x[i]:10.6f} | {y[i]:10.6f}")
        
        # 4. 可视化点阵
        visualize_coords(x, y)
        
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 确保matplotlib中文显示（可选，避免字符乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
    plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示
    
    main()