import matplotlib
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

matplotlib.use('TkAgg')

# 设置国内镜像加速下载 (可选，如果需要重新下载模型的话)
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class SDFeatureMatcher:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda", layer_idx=2):
        """
        初始化 Stable Diffusion 特征匹配器
        :param model_id: 本地模型路径 或 HuggingFace 模型 ID
        :param device: 运行设备 ('cuda' or 'cpu')
        :param layer_idx: 提取 UNet decoder 的第几层特征 (通常 1, 2, 3 效果较好，RoboTwin 常用 up_blocks[1])
        """
        self.device = device
        print(f"Loading SD model from: {model_id}...")
        # 加载本地模型时，torch_dtype 需要根据下载的模型实际精度设置，如果报错改为 torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
        print("Model loaded.")

        # 冻结模型参数，只做推断
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)

        # 注册 Hook 来捕获中间层特征
        self.features = {}
        self.layer_idx = layer_idx
        self._register_hooks()

    def _register_hooks(self):
        """注册 forward hook 到指定的 UNet 层"""

        def hook_fn(module, input, output):
            self.features["feature_map"] = output

        # 这里的层级路径可能随 diffusers 版本微调，通常 up_blocks[1] 是包含丰富语义的高分辨率层
        # up_blocks[1] 的输出通常是 64x64 (对于 512x512 输入)
        # 如果你的 diffusers 版本较新，且报错找不到 up_blocks，可能需要调整为 down_blocks 或 mid_block
        self.pipe.unet.up_blocks[self.layer_idx].register_forward_hook(hook_fn)

    def extract_features(self, image, prompt="a photo of an object"):
        """
        从单张图像提取特征
        :param image: PIL Image 对象
        :return: 归一化后的特征图 [1, C, H, W]
        """
        # 1. 预处理图像 (记录原始尺寸以便后续恢复坐标)
        original_size = image.size  # (W, H)
        image_resized = image.resize((512, 512), resample=Image.BILINEAR)

        # 将图像编码到 Latent Space
        latents = self.pipe.vae.encode(
            self.pipe.image_processor.preprocess(image_resized).to(self.device, dtype=torch.float16)
        ).latent_dist.sample() * self.pipe.vae.config.scaling_factor

        # 2. 准备文本嵌入 (即使是空提示词也需要)
        prompt_embeds = self.pipe.encode_prompt(prompt, self.device, 1, False)[0]

        # 3. 设定时间步 (Timestep)
        # 论文指出较小的 t (如 t=100~200) 能保留更多纹理细节，较大的 t (如 t=500) 语义更强
        t = torch.tensor([400], device=self.device, dtype=torch.long)

        # 4. 前向传播 (Forward Pass)
        self.pipe.unet(latents, t, encoder_hidden_states=prompt_embeds)

        # 5. 获取 Hook 捕获的特征
        feat_map = self.features["feature_map"]  # [B, C, h_small, w_small]

        # 上采样回 512x512 (特征提取的标准空间)
        feat_map = F.interpolate(feat_map.float(), size=(512, 512), mode='bilinear', align_corners=False)

        # L2 归一化，方便计算余弦相似度
        feat_map = F.normalize(feat_map, dim=1)

        return feat_map, original_size

    def match_points(self, src_img, tgt_img, src_points_xy):
        """
        计算源图像关键点在目标图像中的对应位置
        :param src_img: 源图像 (PIL Image)
        :param tgt_img: 目标图像 (PIL Image)
        :param src_points_xy: 源图像上的关键点坐标列表 [[x1, y1], [x2, y2], ...]
        :return: 目标图像上的关键点坐标列表 [[x1', y1'], ...]
        """
        # 1. 提取两张图的特征
        # 注意：特征是在 512x512 空间下计算的
        src_feat, src_orig_size = self.extract_features(src_img)  # [1, C, 512, 512]
        tgt_feat, tgt_orig_size = self.extract_features(tgt_img)  # [1, C, 512, 512]

        results = []
        # 特征图的空间尺寸固定为 512 (因为我们在 extract_features 里 interpolate 到了 512)
        Feat_H, Feat_W = 512, 512
        src_W, src_H = src_orig_size
        tgt_W, tgt_H = tgt_orig_size

        for x_orig, y_orig in src_points_xy:
            # --- 坐标映射：原始尺寸 -> 512特征空间 ---
            # 需要将原始图片上的坐标映射到 512x512 的特征图空间上
            x_512 = int(x_orig * (Feat_W / src_W))
            y_512 = int(y_orig * (Feat_H / src_H))

            # 确保坐标在范围内
            x_512 = np.clip(x_512, 0, Feat_W - 1)
            y_512 = np.clip(y_512, 0, Feat_H - 1)

            # 2. 获取源点的特征向量
            target_vec = src_feat[0, :, y_512, x_512]  # [C]

            # 3. 计算与目标图所有像素的余弦相似度
            # einsum: 计算 target_vec 与 tgt_feat 每一个像素的点积 (因为已经归一化，所以点积=余弦相似度)
            sim_map = torch.einsum('chw, c -> hw', tgt_feat[0], target_vec)

            # 4. 找到相似度最大的位置 (argmax) 在 512 特征空间中
            # np.unravel_index：把一个“扁平的编号”还原成“二维坐标”。
            y_max_512, x_max_512 = np.unravel_index(torch.argmax(sim_map).cpu().numpy(), (Feat_H, Feat_W))

            # --- 坐标映射：512特征空间 -> 目标图像原始尺寸 ---
            x_tgt_orig = int(x_max_512 * (tgt_W / Feat_W))
            y_tgt_orig = int(y_max_512 * (tgt_H / Feat_H))

            results.append([x_tgt_orig, y_tgt_orig])

        return results


def visualize_matches(src_img, tgt_img, src_points, tgt_points, point_radius=8):
    """
    可视化匹配结果，在两张图上绘制红点
    """
    # 复制图像以免修改原图
    src_draw_img = src_img.copy()
    tgt_draw_img = tgt_img.copy()

    # 创建绘图对象
    draw_src = ImageDraw.Draw(src_draw_img)
    draw_tgt = ImageDraw.Draw(tgt_draw_img)

    # 设置颜色 (红色)
    point_color = (255, 0, 0)

    # 遍历所有点对并绘制
    # 使用 zip 同时遍历源点和目标点列表
    for i, ((sx, sy), (tx, ty)) in enumerate(zip(src_points, tgt_points)):
        # 在源图上画实心圆
        # ellipse 接受边界框坐标 [left, top, right, bottom]
        draw_src.ellipse(
            (sx - point_radius, sy - point_radius, sx + point_radius, sy + point_radius),
            fill=point_color, outline=point_color
        )
        # 在目标图上画实心圆
        draw_tgt.ellipse(
            (tx - point_radius, ty - point_radius, tx + point_radius, ty + point_radius),
            fill=point_color, outline=point_color
        )
        print(f"Point {i}: Source({sx}, {sy}) -> Target({tx}, {ty})")

    # 使用 matplotlib 显示图像
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 显示源图
    axes[0].imshow(src_draw_img)
    axes[0].set_title("Source Image (Reference)")
    axes[0].axis('off')  # 关闭坐标轴显示

    # 显示目标图
    axes[1].imshow(tgt_draw_img)
    axes[1].set_title("Target Image (Matched Result)")
    axes[1].axis('off')

    plt.tight_layout()
    print("Displaying visualization result...")
    plt.show()


# ================= 使用示例 =================
if __name__ == "__main__":
    # --- 配置 ---
    # 1. 设置你的本地模型路径 (请替换为你实际下载的路径)
    # 例如 Windows: r"D:\AI_Models\stable-diffusion-v1-5"
    # 例如 Linux/Mac: "/home/user/models/stable-diffusion-v1-5"
    local_model_path = "/home/benson/projects/second_work/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 准备图像路径
    img1_path = "pic.jpg"
    img2_path = "pic2.jpg"

    # 检查文件是否存在，不存在则创建假数据用于测试运行
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Warning: Image files not found. Creating dummy images for testing.")
        Image.new('RGB', (600, 800), color='white').save(img1_path)
        Image.new('RGB', (700, 600), color='gray').save(img2_path)
        source_points = [[554, 306]]
    else:
        # 3. 假设已知 Source 图上的像素坐标 (请根据你的 pic.jpg 实际情况修改)
        # 例如：杯把的位置 [x, y]
        source_points = [[554, 306]]
        # 可以添加多个点： source_points = [[554, 306], [100, 200]]

    # --- 主流程 ---
    try:
        # 初始化匹配器，传入本地路径
        # 注意：如果你的本地模型不是 fp16 的，请在初始化里去掉 torch_dtype=torch.float16
        matcher = SDFeatureMatcher(model_id=local_model_path, device=device)

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        print(f"Source Image Size: {img1.size}")
        print(f"Target Image Size: {img2.size}")

        print("Calculating matches...")
        # 计算 Target 图上对应的位置
        target_points = matcher.match_points(img1, img2, source_points)

        # --- 可视化 ---
        visualize_matches(img1, img2, source_points, target_points)

    except OSError as e:
        print(f"\nError: Loading model failed. \nDetails: {e}")
        print(
            "Please ensure 'local_model_path' points to a valid Stable Diffusion directory containing model_index.json.")
    except Exception as e:
        print(f"An error occurred: {e}")