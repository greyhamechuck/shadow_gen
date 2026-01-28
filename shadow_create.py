import sys
import os
import cv2
import numpy as np
import torch
from rembg import remove
from transformers import pipeline
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, 
                             QMessageBox, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

class ShadowProcessor:
    def __init__(self):
        self.fg_img = None  
        self.bg_img = None  
        self.depth_map = None 
        self.mask = None    
        # 针对 M4 Mac 优化显存和性能
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # 使用 'Base' 模型：平衡速度和细节 (比 Small 细致，但比 Large 快/小)
        self.depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", device=self.device)

    def load_foreground(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return False
        
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # Scale up the foreground by 1.6x
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 1.0), int(h * 1.0)), interpolation=cv2.INTER_CUBIC)
        print("Scaled loaded subject by 1.6x")
            
        print("Removing background... this might take a second.")
        self.fg_img = remove(img)
        self.mask = self.fg_img[:, :, 3]
        return True

    def load_background(self, path):
        img = cv2.imread(path)
        if img is None: return False
        self.bg_img = img
        return True

    def auto_generate_depth(self):
        if self.bg_img is None: return False
        rgb = cv2.cvtColor(self.bg_img, cv2.COLOR_BGR2RGB)
        result = self.depth_pipe(Image.fromarray(rgb))
        self.depth_map = np.array(result["depth"])
        return True

    def apply_ray_march(self, shadow_mask, angle, elevation, strength):
        """ 
        [修正版] 深度置换 (Depth Displacement)
        根据背景的深度（灰度）将影子“搬运”到物体表面。
        白色（高处）= 影子会向光源方向移动（看起来像爬墙）。
        """
        if self.depth_map is None or strength <= 0: return shadow_mask, None
        
        h, w = shadow_mask.shape[:2]
        depth_raw = cv2.resize(self.depth_map, (w, h)).astype(np.float32) / 255.0
        
        # 1. 计算梯度 & 脉冲 (Gradient & Impulse)
        # 我们寻找"Rising Slope" (Wall), 即 d(Depth)/dy < 0.
        gy = cv2.Sobel(depth_raw, cv2.CV_32F, 0, 1, ksize=5)
        rising_slope = np.clip(-gy, 0, None) # 只取正值部分
        
        # 定义"Shift Impulse": 每个像素贡献多少位移
        # 强度由 strength 控制. 
        # 用户需求: "每一个梯度变化的时候... x轴+1"
        # 这意味着位移是累积的 (Cumulative).
        
        # 脉冲强度
        impulse_per_pixel = rising_slope * (strength * 0.2) 
        
        # 2. 累积位移 (Integration)
        # 影子投射路径：从脚底 (Bottom, High Y) -> 远方 (Top, Low Y).
        # 所以我们需要从下往上累积.
        # Shift(y) = Sum(impulse(k)) for k from y to Bottom.
        
        # Flip vertically -> Cumsum -> Flip back
        # axis=0 is vertical (Y)
        acc_shift_x = np.flip(np.cumsum(np.flip(impulse_per_pixel, axis=0), axis=0), axis=0)
        
        # 限制最大位移防止溢出屏幕太远 (Optional)
        # acc_shift_x = np.clip(acc_shift_x, 0, w/2)
        
        # 3. 构建 Remap grids
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        
        # 应用累积位移到 X 轴
        # 应用累积位移到 X 轴
        # 用户需求: "模拟上墙... 保持垂直于墙面的感觉"
        # 解析: 也就是要"抵消"原本光照产生的水平切变 (Horizontal Shear).
        # 让墙上的影子看起来垂直向上 (Vertical).
        
        # 1. 计算几何切变因子 (Geometric Skew Factor)
        # Shadow Length Factor = 1.0 / tan(elevation)
        # Horizontal Drift per vertical unit = cos(angle) * shadow_len
        # (如果光从侧面来，Drift大；光从正面来，Drift 0)
        
        rad_a = np.radians(angle)
        rad_e = np.radians(max(elevation, 10)) # 避免除零
        
        shadow_len = 1.0 / np.tan(rad_e)
        
        # Correction Factor:
        # 我们需要抵消原本的投影偏移.
        # Angle=0 (Right Light) -> Shadow Left (Neg Drift) -> Need Pos Correction (Move Right).
        # cos(0) = 1. Positive. Matches.
        # Angle=180 (Left Light) -> Shadow Right (Pos Drift) -> Need Neg Correction (Move Left).
        # cos(180) = -1. Negative. Matches.
        
        skew_correction = np.cos(rad_a) * shadow_len
        
        # 2. 应用矫正
        # acc_shift_x 代表了"累积的墙面高度" (模拟值).
        # 我们用这个高度 * Skew Correction 来算出需要这一行需要回拉多少像素才能变直.
        # Magic Number 4.0: 将深度累积值映射到像素空间的系数.
        
        correction_offset = acc_shift_x * skew_correction * 4.0
        
        # Map = Grid - Offset
        # Angle 0 -> Skew Pos -> Offset Pos -> Map < Grid -> Sample Left -> Content Moves Right. Correct.
        map_x = grid_x - correction_offset
        map_y = grid_y # Y轴保持不变
        
        # Ensure maps are float32 for OpenCV remap
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # 4. 执行变形
        # 这是一个全局变形，应用于整个 shadow_mask
        warped_shadow = cv2.remap(shadow_mask, map_x, map_y, 
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
        
        # 5. 深度剔除
        visibility_mask = np.power(depth_raw, 1.5)
        visibility_mask[depth_raw < 0.05] = 0.0
        
        final_shadow = (warped_shadow.astype(np.float32) * visibility_mask).astype(np.uint8)
        
        # 生成一个用于Debug/Coloring的Mask
        # 指示哪些地方正在经历显著的"Shift" (即 Wall 区域)
        # 注意：这里的 Wall Mask 是"触发源"，而不是变形后的结果位置.
        # 如果想显示"变形后的墙上影子"，我们需要对 wall_mask 也做同样的 remap.
        wall_trigger_mask = (rising_slope > 0.002).astype(np.float32)
        warped_wall_mask = cv2.remap(wall_trigger_mask, map_x, map_y,
                                    interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        
        # 仅保留落在影子里的部分
        wall_hits_mask = (warped_wall_mask * (final_shadow/255.0)).astype(np.uint8)
        
        # 返回:
        # 1. 地面层 (Main Shadow)：也可以包含全部，然后在 Composite 里决定是否扣除
        # 为了兼容之前的 "蓝色显示" 逻辑:
        # Ground = All - Wall
        # Wall = Wall
        
        return final_shadow, wall_hits_mask

    def generate_composite(self, angle, elevation, softness, opacity, d_strength, pos_y, save_debug=False):
        if self.fg_img is None or self.bg_img is None: return None
        bg_h, bg_w = self.bg_img.shape[:2]
        fg_h, fg_w = self.fg_img.shape[:2]
        
        # 1. 锚点与放置
        y_idx, x_idx = np.where(self.mask > 0)
        feet_y, feet_x = np.max(y_idx), int(np.mean(x_idx))
        offset_x = (bg_w // 2) - (fg_w // 2)
        offset_y = int(bg_h * (pos_y / 100.0)) - feet_y
        
        # 2. 生成物理准确的几何投影
        rad_elev = np.radians(max(elevation, 5))
        shadow_len = 1.0 / np.tan(rad_elev)
        shift_x = -np.cos(np.radians(angle)) * shadow_len * 100
        shift_y = -np.sin(np.radians(angle)) * shadow_len * 100
        src = np.float32([[feet_x, feet_y], [feet_x, feet_y-100], [feet_x+100, feet_y]])
        dst = np.float32([[feet_x+offset_x, feet_y+offset_y], 
                          [feet_x+offset_x+shift_x, feet_y+offset_y+shift_y], 
                          [feet_x+offset_x+100, feet_y+offset_y]])
        
        raw_shadow = cv2.warpAffine(self.mask, cv2.getAffineTransform(src, dst), (bg_w, bg_h))
        
        # 应用深度感知的 累积 Ray Marching
        full_shadow, wall_shadow_part = self.apply_ray_march(raw_shadow, angle, elevation, d_strength)
        
        # 3. 模拟“接触硬化”物理模糊
        grid_y, grid_x = np.indices((bg_h, bg_w))
        dist = np.sqrt((grid_x - (feet_x+offset_x))**2 + (grid_y - (feet_y+offset_y))**2)
        alpha_mix = np.power(np.clip(dist / (200 * shadow_len), 0, 1), 0.6)
        
        def blur_mask(m):
            if m is None or np.max(m) == 0: return np.zeros((bg_h, bg_w), dtype=np.float32)
            s_far = cv2.GaussianBlur(m, (softness|1, softness|1), 0)
            s_near = cv2.GaussianBlur(m, (max(1, softness//15)|1, max(1, softness//15)|1), 0)
            return (s_near * (1 - alpha_mix) + s_far * alpha_mix).astype(np.float32) / 255.0

        # 对 Main Shadow 进行模糊
        final_shd = blur_mask(full_shadow)
        
        # 如果需要分别着色，我们可以单独模糊 Wall Part
        if wall_shadow_part is not None:
             final_wall_shd = blur_mask(wall_shadow_part)
        else:
             final_wall_shd = np.zeros_like(final_shd)
            
        # 4. 增强脚底 AO
        ao_mask = np.clip(1.0 - (dist / 40.0), 0, 1) * 0.4
        
        # 5. 合成逻辑
        result = self.bg_img.copy()
        
        # 主要阴影层 (黑色/变暗)
        # 用 Full Shadow 减去 Wall Shadow (以免重叠部分被画两次, 或者我们想保留 Wall Shadow 的黑色底色?)
        # 假设 Wall Shadow 是"蓝色高亮"。我们通常希望它是"有色阴影"。
        # 所以 Ground Shadow = Full - Wall
        
        wall_val = final_wall_shd if final_wall_shd is not None else 0
        ground_val = np.maximum(final_shd - wall_val, 0) # 简单的扣除
        
        # 渲染黑色地面阴影
        combined_ground = np.maximum(ground_val * (opacity / 100.0), ao_mask)
        for c in range(3): 
            result[:,:,c] = (result[:,:,c] * (1.0 - combined_ground)).astype(np.uint8)
        
        # 渲染彩色墙面阴影 (蓝色)
        if np.max(wall_val) > 0:
            w_alpha = np.expand_dims(wall_val * (opacity / 100.0), axis=-1)
            color_layer = np.zeros_like(result, dtype=np.float32)
            color_layer[:,:,0] = 255 # Blue
            
            # 混合: Dest = Src * (1-a) + Color * a
            # 这里背景已经被 Ground Shadow 变暗了(如果没有重叠)。
            # 如果是独立的层，直接混合
            target = result.astype(np.float32)
            # Wall shadow 还是要有遮蔽效果(变暗) + 蓝色
            # Darkened = Target * 0.4
            # Blue = Color * 0.6
            colored = target * 0.4 + color_layer * 0.6
            result = (colored * w_alpha + target * (1.0 - w_alpha)).astype(np.uint8)
        
        # 6. 正确叠加主体
        y1, y2, x1, x2 = offset_y, offset_y + fg_h, offset_x, offset_x + fg_w
        y1c, y2c, x1c, x2c = max(0, y1), min(bg_h, y2), max(0, x1), min(bg_w, x2)
        if y1c < y2c and x1c < x2c:
            alpha_c = np.expand_dims((self.fg_img[:,:,3]/255.0)[y1c-y1:y2c-y1, x1c-x1:x2c-x1], axis=-1)
            rgb_c = self.fg_img[y1c-y1:y2c-y1, x1c-x1:x2c-x1, :3]
            target = result[y1c:y2c, x1c:x2c].astype(float)
            result[y1c:y2c, x1c:x2c] = (rgb_c.astype(float) * alpha_c + target * (1.0 - alpha_c)).astype(np.uint8)

        if save_debug:
            print(f"Saving output to {os.getcwd()}...")
            cv2.imwrite("composite.png", result)
            cv2.imwrite("shadow_only.png", (final_shd * 255).astype(np.uint8))
            cv2.imwrite("mask_debug.png", self.mask)
            if self.depth_map is not None:
                cv2.imwrite("depth_map.png", self.depth_map)
            print("💾 Files Saved Successfully!")

        return result


# ==========================================
# 🖥️ UI: PyQt6
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = ShadowProcessor()
        self.setWindowTitle("AI-Powered Shadow Engine (Ray Tracing)")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()

    def initUI(self):
        main = QWidget(); self.setCentralWidget(main); layout = QHBoxLayout(main)
        
        # Controls
        sidebar = QFrame(); sidebar.setFixedWidth(300); side_ly = QVBoxLayout(sidebar)
        
        self.btn_fg = QPushButton("1. Load Subject"); self.btn_fg.clicked.connect(self.load_fg)
        self.btn_bg = QPushButton("2. Load Background"); self.btn_bg.clicked.connect(self.load_bg)
        self.btn_ai = QPushButton("✨ AI Auto-Depth"); self.btn_ai.clicked.connect(self.run_ai)
        self.btn_ai.setStyleSheet("background-color: #6200EE; color: white; font-weight: bold;")
        self.btn_save = QPushButton("💾 Save Layers"); self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setStyleSheet("background-color: #00C853; color: white; margin-top: 10px;")
        
        def sld(name, mi, ma, iv):
            l = QLabel(f"{name}: {iv}"); s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(mi, ma); s.setValue(iv); s.valueChanged.connect(lambda v: (l.setText(f"{name}: {v}"), self.draw()))
            return l, s

        self.l_a, self.s_a = sld("Angle", 0, 360, 45)
        self.l_e, self.s_e = sld("Elevation", 10, 80, 45)
        self.l_b, self.s_b = sld("Softness", 1, 100, 20)
        self.l_o, self.s_o = sld("Opacity", 0, 100, 70)
        self.l_d, self.s_d = sld("Ray Distance", 0, 200, 50)
        self.l_p, self.s_p = sld("Position Y%", 0, 120, 90)

        for w in [self.btn_fg, self.btn_bg, self.btn_ai, self.btn_save, self.l_a, self.s_a, self.l_e, self.s_e, 
                  self.l_b, self.s_b, self.l_o, self.s_o, self.l_d, self.s_d, self.l_p, self.s_p]:
            side_ly.addWidget(w)
        side_ly.addStretch()

        self.view = QLabel("Waiting for images..."); self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setStyleSheet("background: #111; border: 1px solid #333;")
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        layout.addWidget(sidebar); layout.addWidget(self.view, 1)

    def load_fg(self):
        f, _ = QFileDialog.getOpenFileName(self, "Person Image"); 
        if f and self.processor.load_foreground(f): self.btn_fg.setText("Subject ✅"); self.draw()

    def load_bg(self):
        f, _ = QFileDialog.getOpenFileName(self, "Background"); 
        if f and self.processor.load_background(f): self.btn_bg.setText("Background ✅"); self.draw()

    def run_ai(self):
        self.btn_ai.setText("Processing..."); QApplication.processEvents()
        if self.processor.auto_generate_depth():
            self.btn_ai.setText("✨ AI Depth Ready"); self.draw()

    def draw(self):
        res = self.processor.generate_composite(self.s_a.value(), self.s_e.value(), self.s_b.value(), 
                                               self.s_o.value(), self.s_d.value(), self.s_p.value())
        if res is not None:
            h, w, _ = res.shape; q = QImage(res.data, w, h, 3*w, QImage.Format.Format_BGR888)
            self.view.setPixmap(QPixmap.fromImage(q).scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def save_results(self):
        self.processor.generate_composite(self.s_a.value(), self.s_e.value(), self.s_b.value(), 
                                         self.s_o.value(), self.s_d.value(), self.s_p.value(), save_debug=True)
        QMessageBox.information(self, "Success", "Saved composite.png, shadow_only.png, and mask_debug.png (and depth_map.png if available)")

if __name__ == "__main__":
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())