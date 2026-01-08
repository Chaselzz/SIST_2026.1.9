import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
import sys
import math
from typing import List, Tuple, Optional
from datetime import datetime

class CharacterContourExtractor:
    def __init__(self, font_path: str = "C://Windows//Fonts//simhei.ttf",  #windows默认黑体路径，更改路径可满足输出不同字体的要求
                 min_points: int = 600, 
                 max_points: int = 1200,
                 target_min: float = 0.2,
                 target_max: float = 0.6):
        """初始化提取器（默认黑体）"""
        self.font_path = font_path
        self.min_points = min_points
        self.max_points = max_points
        self.target_min = target_min
        self.target_max = target_max
        self.font_size = 900  # 黑体适配，保证笔画清晰
        
        # 获取程序目录
        if hasattr(sys, 'frozen'):
            self.program_dir = os.path.dirname(sys.executable)
        else:
            self.program_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        
        self.output_dir = os.path.join(self.program_dir, "character_contours")
        
        print(f"程序文件目录: {self.program_dir}")
        print(f"字体: 黑体 ({self.font_path})")
        print(f"总点数控制范围: {min_points}-{max_points} 个")
        print(f"坐标转换范围: [{target_min}, {target_max}]")
        print(f"特性: 全局归一化+足够点数，保留笔画原始姿态")
        
        # 检查字体文件
        if not os.path.exists(self.font_path):
            print(f"文件未找到: {self.font_path}")
    
    def create_high_res_image(self, char: str) -> np.ndarray:
        """创建字符图像（创新点，使用opencv扫描字符轮廓）"""
        img_size = 1400
        img = Image.new('L', (img_size, img_size), 255)  # 白色背景
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except Exception as e:
            print(f"无法加载字体: {e}")
            font = ImageFont.load_default()
            print("使用默认字体")
        
        # 计算字符居中位置
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # 保证居中
        position = (
            (img_size - text_width) // 2 - bbox[0],
            (img_size - text_height) // 2 - bbox[1]             #取中点值
        )
        
        # 绘制
        draw.text(position, char, fill=0, font=font)
        
        return np.array(img)
    
    def calculate_contour_length(self, contour: np.ndarray) -> float:
        """计算单个轮廓的总弧长（逻辑不变）"""
        if len(contour) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(contour)):
            length += np.linalg.norm(contour[i] - contour[i-1])
        return length
    
    def uniform_resample_contour(self, contour: np.ndarray, target_points: int) -> np.ndarray:
        """对单个轮廓均匀重采样（逻辑不变）"""
        if len(contour) < 2 or target_points < 2:
            return contour
        
        distances = [0.0]
        for i in range(1, len(contour)):
            distances.append(distances[-1] + np.linalg.norm(contour[i] - contour[i-1]))
        total_length = distances[-1]
        if total_length < 1e-6:
            return contour
        
        step = total_length / (target_points - 1)
        uniform_points = []
        current_dist, idx = 0.0, 0
        for _ in range(target_points):
            while idx < len(distances)-1 and distances[idx+1] < current_dist:
                idx += 1
            if idx == len(distances)-1:
                uniform_points.append(contour[-1])
            else:
                t = (current_dist - distances[idx]) / (distances[idx+1] - distances[idx])
                uniform_points.append(contour[idx]*(1-t) + contour[idx+1]*t)
            current_dist += step
        
        return np.array(uniform_points)
    
    def extract_contour_points(self, char_image: np.ndarray) -> List[Tuple[float, float]]:
        """提取轮廓"""
        print("提取字符轮廓（保留原始姿态）...")
        
        # 二值化
        _, binary = cv2.threshold(char_image, 150, 255, cv2.THRESH_BINARY_INV)  
        
        # 提取所有轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print("未提取到轮廓，使用边缘检测兜底")
            edges = cv2.Canny(char_image, 80, 200)  
            edge_points = np.column_stack(np.where(edges > 0))
            sample_size = min(1500, len(edge_points))
            edge_points = edge_points[np.random.choice(len(edge_points), sample_size, replace=False)]
            points_xy = [(int(x), int(y)) for y, x in edge_points]
            # 全局归一化保底（如果只有轮廓提取，对某些生僻字、笔画粘连的字和部分特殊字符会有bug）
            all_x = [p[0] for p in points_xy]
            all_y = [p[1] for p in points_xy]
            gx_min, gx_max = min(all_x), max(all_x)
            gy_min, gy_max = min(all_y), max(all_y)
            gc_x, gc_y = (gx_min+gx_max)/2, (gy_min+gy_max)/2
            g_width, g_height = max(1, gx_max-gx_min), max(1, gy_max-gy_min)
            normalized = [
                ((x-gc_x)/(g_width/2), (gc_y-y)/(g_height/2)) 
                for x,y in points_xy
            ]
            final_points = self.adjust_point_count(normalized)
            return final_points
        
        # 过滤噪点轮廓
        valid_contours = []
        contour_lengths = []
        for contour in contours:
            if cv2.contourArea(contour) > 5:  
                contour_np = np.array([point[0] for point in contour], dtype=np.float32)
                valid_contours.append(contour_np)
                contour_lengths.append(self.calculate_contour_length(contour_np))
        
        print(f"   有效笔画轮廓数量: {len(valid_contours)} 个")
        print(f"   各轮廓原始点数: {[len(c) for c in valid_contours]}")
        
        # 计算字符全局边界（所有轮廓的点，保证相对位置）
        all_contour_points = []
        for contour in valid_contours:
            all_contour_points.extend([(p[0], p[1]) for p in contour])
        global_x = [p[0] for p in all_contour_points]
        global_y = [p[1] for p in all_contour_points]
        global_x_min, global_x_max = min(global_x), max(global_x)
        global_y_min, global_y_max = min(global_y), max(global_y)
        global_center_x = (global_x_min + global_x_max) / 2.0
        global_center_y = (global_y_min + global_y_max) / 2.0
        global_width = max(1, global_x_max - global_x_min)
        global_height = max(1, global_y_max - global_y_min)
        print(f"   字符全局边界: 宽{global_width:.1f} × 高{global_height:.1f}（保证姿态）")
        
        # 分配总点数（600-1200）
        total_length = sum(contour_lengths)
        target_total_points = np.clip(int(total_length * 0.8), self.min_points, self.max_points)  # 适配笔画密度
        print(f"   总点数分配目标: {target_total_points} 个")
        
        # 对每个轮廓单独重采样（至少30个，避免形状变异）
        processed_contours = []
        for i, (contour, length) in enumerate(zip(valid_contours, contour_lengths)):
            if total_length < 1e-6:
                contour_points = [(float(p[0]), float(p[1])) for p in contour]
            else:
                # 按长度比例分配，每个轮廓至少30点
                ratio = length / total_length
                contour_target_points = max(30, int(target_total_points * ratio))
                resampled = self.uniform_resample_contour(contour, contour_target_points)
                contour_points = [(float(p[0]), float(p[1])) for p in resampled]
            
            # 用全局边界归一化 （相对位置/大小/姿态）
            normalized = []
            for x, y in contour_points:
                norm_x = (x - global_center_x) / (global_width / 2.0)
                norm_y = (global_center_y - y) / (global_height / 2.0)  # Y轴向上
                normalized.append((float(f"{norm_x:.6f}"), float(f"{norm_y:.6f}")))
            processed_contours.append(normalized)
            print(f"   笔画{i+1}: 原始{len(contour)}个 → 重采样{len(contour_points)}个（足够点数防变异）")
        
        # 合并所有点（保持输出格式）
        all_points = []
        for contour_points in processed_contours:
            all_points.extend(contour_points)
        
        # 最终校验一下
        final_points = all_points
        if len(final_points) < self.min_points:
            shortage = self.min_points - len(final_points)
            for i, cnt in enumerate(processed_contours):
                add_cnt = max(2, int(shortage * len(cnt)/len(all_points)))
                resampled = self.uniform_resample_contour(np.array(cnt), len(cnt)+add_cnt)
                final_points = final_points[:len(final_points)-len(cnt)] + [(float(p[0]),float(p[1])) for p in resampled]
        elif len(final_points) > self.max_points:
            excess = len(final_points) - self.max_points
            for i, cnt in enumerate(processed_contours):
                reduce_cnt = max(2, int(excess * len(cnt)/len(all_points)))
                resampled = self.uniform_resample_contour(np.array(cnt), len(cnt)-reduce_cnt)
                final_points = final_points[:len(final_points)-len(cnt)] + [(float(p[0]),float(p[1])) for p in resampled]
        
        print(f"最终总点数: {len(final_points)} 个")
        return final_points
    
    def adjust_point_count(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """备用点数调整"""
        current_count = len(points)
        print(f"点数调整: 原始{current_count}个 → 目标{self.min_points}-{self.max_points}个")
        
        points_np = np.array(points, dtype=np.float32)
        target_count = np.clip(current_count, self.min_points, self.max_points)
        uniform_points = self.uniform_resample_contour(points_np, target_count)
        
        return [(float(p[0]), float(p[1])) for p in uniform_points]
    
    def convert_to_target_area(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """转换到ur5e作用区域，经测试，0.2-0.6范围内ur5e机械臂不会有奇异值"""
        if not points:
            print("空坐标列表，无法转换")
            return []
        
        print(f"转换坐标到第一象限[{self.target_min}, {self.target_max}]...")
        converted = []
        target_range = self.target_max - self.target_min
        for x, y in points:
            x_01 = (x + 1) / 2.0
            y_01 = (y + 1) / 2.0
            final_x = self.target_min + x_01 * target_range
            final_y = self.target_min + y_01 * target_range
            converted.append((float(f"{final_x:.6f}"), float(f"{final_y:.6f}")))
        
        # 验证范围
        xs = [p[0] for p in converted]
        ys = [p[1] for p in converted]
        print(f"   转换后X范围: [{min(xs):.6f}, {max(xs):.6f}]")
        print(f"   转换后Y范围: [{min(ys):.6f}, {max(ys):.6f}]")
        
        return converted
    
    def save_converted_file(self, char: str, points: List[Tuple[float, float]]) -> str:
        """保存坐标文件（格式不变）"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        safe_char = "".join([c if c.isalnum() else f"U{ord(c):04X}" for c in char])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"contour_{safe_char}_{timestamp}_黑体_converted.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("#"*80 + "\n")
            f.write(f"# 字符轮廓点数据 - {char} (转换到[{self.target_min},{self.target_max}]区域)\n")
            f.write(f"# 点数范围: {self.min_points}-{self.max_points} 个\n")
            f.write("#"*80 + "\n\n")
            
            f.write(f"字符: {char}\n")
            f.write(f"字体: 黑体\n")
            f.write(f"总点数: {len(points)} 个\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"坐标范围: 第一象限 [{self.target_min}, {self.target_max}]\n\n")
            
            f.write(f"坐标数据 (第一象限，范围: [{self.target_min}, {self.target_max}]):\n")
            f.write("格式: X坐标 Y坐标\n")
            f.write("-"*60 + "\n")
            for x, y in points:
                f.write(f"{x:.6f} {y:.6f}\n")
        
        print(f"轮廓文件已保存到: {filepath}")
        return filepath
    
    def visualize_results(self, char: str, char_image: np.ndarray, points: List[Tuple[float, float]]):
        """可视化（验证字体姿态）"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 1. 原始黑体字符
            axes[0].imshow(char_image, cmap='gray')
            axes[0].set_title(f"原始字符: {char}")
            axes[0].axis('off')
            
            # 2. 验证姿态
            height, width = char_image.shape
            point_img = np.ones((height, width), dtype=np.uint8)*255
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            center_x = (x_min + x_max)/2
            center_y = (y_min + y_max)/2
            display_width = max(1, x_max-x_min)*1.2
            display_height = max(1, y_max-y_min)*1.2
            
            for x, y in points:
                px = int((x - center_x + display_width/2) * (width/display_width))
                py = int((center_y - y + display_height/2) * (height/display_height))
                if 0<=px<width and 0<=py<height:
                    cv2.circle(point_img, (px, py), 2, 0, -1)
            
            axes[1].imshow(point_img, cmap='gray')
            axes[1].set_title(f"轮廓点 ({len(points)}个) - 原始姿态")
            axes[1].axis('off')
            
            # 3. 点云图
            ax3 = axes[2]
            ax3.scatter([p[0] for p in points], [p[1] for p in points], s=3, c='red', alpha=0.8)
            ax3.set_title("轮廓点云")
            ax3.set_xlabel("X坐标")
            ax3.set_ylabel("Y坐标")
            ax3.grid(alpha=0.3)
            ax3.axis('equal')
            
            plt.suptitle(f"字符 '{char}' 轮廓提取结果（保留原始姿态）", fontsize=14)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"可视化错误: {e}")
def process_character(char: str, 
                     font_path: str = "C://Windows//Fonts//simhei.ttf",  # 黑体默认路径
                     show_visualization: bool = True,
                     target_min: float = 0.2,
                     target_max: float = 0.6) -> Tuple[List[Tuple[float, float]], str]:
    """处理单个字符（逻辑不变）"""
    print("="*70)
    print(f"处理黑体字符: '{char}'")
    print(f"目标区域[{target_min},{target_max}]")
    print("="*70)
    
    try:
        extractor = CharacterContourExtractor(
            font_path=font_path,
            min_points=600,
            max_points=1200,
            target_min=target_min,
            target_max=target_max
        )
        
        print("创建高分辨率图像...")
        char_image = extractor.create_high_res_image(char)
        print(f"   图像尺寸: {char_image.shape[1]}×{char_image.shape[0]}")
        
        print("提取轮廓点（保留原始姿态）...")
        original_points = extractor.extract_contour_points(char_image)
        
        print("转换坐标到目标区域...")
        converted_points = extractor.convert_to_target_area(original_points)
        
        print("保存轮廓文件...")
        converted_filepath = extractor.save_converted_file(char, converted_points)
        
        if show_visualization:
            print("显示可视化结果...")
            extractor.visualize_results(char, char_image, converted_points)
        
        print("\n字符处理完成!")
        print(f"   字符: {char}")
        print(f"   最终点数: {len(converted_points)} 个")
        print(f"   文件位置: {converted_filepath}")
        
        if converted_points:
            print(f"\n前10个坐标示例:")
            print("-"*60)
            for i, (x,y) in enumerate(converted_points[:10],1):
                print(f"{i:3d}: ({x:10.6f}, {y:10.6f})")
            
            distances = [math.hypot(p1[0]-p2[0], p1[1]-p2[1]) for p1,p2 in zip(converted_points[:-1], converted_points[1:])]
            print(f"\n均匀性指标:")
            print(f"   平均距离: {np.mean(distances):.6f}")
            print(f"   距离标准差: {np.std(distances):.6f}")
        
        return converted_points, converted_filepath
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return [], None

def get_program_directory():
    """获取程序目录"""
    if hasattr(sys, 'frozen'):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(sys.argv[0]))

def main():
    """主函数"""
    program_dir = get_program_directory()
    contours_dir = os.path.join(program_dir, "character_contours")
    
    print("="*70)
    print("字符轮廓点提取工具")
    print("="*70)
    print(f"程序目录: {program_dir}")
    print(f"文件保存到: {contours_dir}")
    print(f"点数范围: 600-1200 个")
    print()
    
    while True:
        print("\n请选择操作:")
        print("1. 处理单个字符")
        print("2. 查看已保存的轮廓文件")
        print("3. 退出程序")
        
        choice = input("\n输入选择 (1-3): ").strip()
        
        if choice == '1':
            char = input("输入要处理的字符: ").strip()
            if char:
                try:
                    target_min = float(input("目标区域最小值（默认0.2）: ").strip() or 0.2)
                    target_max = float(input("目标区域最大值（默认0.6）: ").strip() or 0.6)
                except ValueError:
                    print("输入无效，使用默认值0.2/0.6")
                    target_min, target_max = 0.2, 0.6
                
                process_character(
                    char=char,
                    show_visualization=True,
                    target_min=target_min,
                    target_max=target_max
                )
        
        elif choice == '2':
            if os.path.exists(contours_dir):
                files = [f for f in os.listdir(contours_dir) if "黑体_converted.txt" in f]
                if files:
                    print(f"\n已保存的轮廓文件 ({len(files)}个):")
                    print("-"*80)
                    for i, f in enumerate(sorted(files),1):
                        print(f"{i:3d}. {f}")
                else:
                    print(f"\n目录 {contours_dir} 中无轮廓文件")
            else:
                print(f"\n目录 {contours_dir} 不存在")
        
        elif choice == '3':
            print("\n程序输出结束")
            break
        
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    # 检查依赖库
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageFont, ImageDraw
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        sys.exit(1)
    main()