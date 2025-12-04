import sys
import os
import torch
import torch.nn.functional as F
import matplotlib
import numpy as np

# 强制使用非交互式后端
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file

# TODO: Benson dinov3的代码有点问题，还可以调一调
class DINOv3Matcher:
    def __init__(self,
                 repo_dir,
                 model_path,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 img_size=512):  # 显式指定高分辨率
        """
        DINOv3 语义匹配器 (融合优化版)
        优点:
        1. 支持 512x512 高分辨率 (通过 Pos Embed 插值)
        2. 使用中间层特征融合 (解决特征坍缩，提高几何精度)
        3. 显存优化 (CPU 处理特征图)
        """
        self.device = device
        self.img_size = img_size
        self.patch_size = 16

        print(f"🚀 初始化... 设备: {self.device}, 分辨率: {img_size}x{img_size}")

        # 1. 加载架构 (引用你的新写法，更稳健)
        self.model = self._load_architecture(repo_dir)
        self.model.to(self.device)
        self.model.eval()

        # 2. 加载权重 (包含 Pos Embed 插值)
        self._load_weights(model_path)

        # 3. 预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=3),  # Bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _load_architecture(self, repo_dir):
        if repo_dir not in sys.path:
            sys.path.append(repo_dir)

        # 动态导入
        try:
            from dinov3.models.vision_transformer import vit_base
        except ImportError:
            print(f"❌ 无法在 {repo_dir} 找到 dinov3 模块，请检查路径。")
            raise

        print("🏗️ 创建模型架构...")
        # 尝试开启 registers (DINOv3 默认配置)
        try:
            model = vit_base(
                img_size=self.img_size,
                patch_size=16,
                num_register_tokens=4
            )
            self.has_registers = True
        except TypeError:
            print("⚠️ 警告: 当前代码库不支持 register_tokens，使用标准 ViT。")
            model = vit_base(img_size=self.img_size, patch_size=16)
            self.has_registers = False

        return model

    def _load_weights(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到权重: {model_path}")

        print(f"📦 加载并适配权重: {os.path.basename(model_path)}")
        state_dict = load_file(model_path)
        new_dict = {}
        model_params = dict(self.model.named_parameters())

        for k, v in state_dict.items():
            # === 键名清洗映射 (来自你的新代码) ===
            k = k.replace('module.', '').replace('backbone.', '')
            k = k.replace('embeddings.patch_embeddings.weight', 'patch_embed.proj.weight')
            k = k.replace('embeddings.patch_embeddings.bias', 'patch_embed.proj.bias')
            k = k.replace('embeddings.cls_token', 'cls_token')
            k = k.replace('embeddings.mask_token', 'mask_token')
            k = k.replace('embeddings.position_embeddings', 'pos_embed')
            k = k.replace('embeddings.register_tokens', 'register_tokens')
            k = k.replace('encoder.layers.', 'blocks.')
            k = k.replace('encoder.norm.', 'norm.')

            # 处理 QKV 权重融合 (如果 safetensors 里是分开的，需要合并)
            # 注意：ModelScope 的 lvd1689m 权重通常已经是融合好的 qkv，
            # 如果报错尺寸不匹配，可能需要在这里加 QKV 合并逻辑。
            # 暂时假设你的权重文件格式与新代码假设的一致。

            # === 维度适配 ===
            if k in model_params:
                target_shape = model_params[k].shape
                if v.ndim != len(target_shape):
                    if v.ndim == 2 and len(target_shape) == 3:
                        v = v.unsqueeze(1)
                    elif v.ndim == 3 and len(target_shape) == 2:
                        v = v.squeeze(1)

            # === 关键：位置编码插值 ===
            if k == 'pos_embed' and v.shape != model_params[k].shape:
                print(f"🔄 Resizing pos_embed: {v.shape} -> {model_params[k].shape}")
                v = self._resize_pos_embed(v, model_params[k].shape)

            new_dict[k] = v

        msg = self.model.load_state_dict(new_dict, strict=False)
        print(f"✅ 权重加载完毕. Missing keys (可忽略 head/rope): {len(msg.missing_keys)}")

    def _resize_pos_embed(self, pos_embed, expected_shape):
        """位置编码双线性插值 (核心修复逻辑)"""
        n_special = 1 + (4 if self.has_registers else 0)  # CLS + Registers

        # 分离特殊 Token 和 Patch Token
        cls_tokens = pos_embed[:, :n_special, :]
        patch_tokens = pos_embed[:, n_special:, :]

        # Reshape 成 2D 网格
        orig_num_patches = patch_tokens.shape[1]
        orig_size = int(orig_num_patches ** 0.5)  # e.g. 14
        dim = patch_tokens.shape[-1]

        patch_tokens = patch_tokens.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)

        # 计算目标尺寸
        target_num_patches = expected_shape[1] - n_special
        target_size = int(target_num_patches ** 0.5)  # e.g. 32

        # 插值
        patch_tokens = F.interpolate(
            patch_tokens, size=(target_size, target_size),
            mode='bicubic', align_corners=False
        )

        # 展平回 [1, N, D]
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        return torch.cat((cls_tokens, patch_tokens), dim=1)

    def extract_features(self, tensor):
        """
        改进版特征提取：使用中间层 + CPU Offload
        """
        with torch.inference_mode():
            # === 核心修改：使用 get_intermediate_layers ===
            # n=4: 提取最后4层，增加几何特征丰富度
            # reshape=True: 自动变为 [B, C, H, W]
            features_list = self.model.get_intermediate_layers(
                tensor,
                n=4,
                reshape=True
            )

            # 拼接多层特征 [1, 768*4, 32, 32]
            # 立即转到 CPU 防止 OOM
            feature_map = torch.cat(features_list, dim=1).cpu()

        # 上采样回 512x512
        feature_map = F.interpolate(
            feature_map.float(),
            size=(self.img_size, self.img_size),
            mode='bicubic',
            align_corners=False
        )

        # 归一化
        feature_map = F.normalize(feature_map, dim=1)

        return feature_map

    def find_correspondence(self, img1_path, img2_path, query_point):
        # 1. 预处理
        img1_pil, tensor1 = self.preprocess(img1_path)
        img2_pil, tensor2 = self.preprocess(img2_path)

        # 2. 提取特征 (此时已在 CPU)
        print("🧠 提取特征中...")
        feat1 = self.extract_features(tensor1)
        feat2 = self.extract_features(tensor2)

        # 3. 坐标转换
        orig_w, orig_h = img1_pil.size
        qx, qy = query_point

        # 映射到 512x512 空间
        x_512 = int(qx / orig_w * self.img_size)
        y_512 = int(qy / orig_h * self.img_size)

        # 边界保护
        x_512 = min(max(x_512, 0), self.img_size - 1)
        y_512 = min(max(y_512, 0), self.img_size - 1)

        # 4. 匹配计算 (CPU)
        # 获取源点特征向量 [C]
        target_vec = feat1[0, :, y_512, x_512]

        # 计算余弦相似度
        # [1, C, H, W] * [C] -> [1, H, W]
        sim_map = torch.einsum('bchw, c -> bhw', feat2, target_vec)

        sim_map_np = sim_map[0].numpy()
        best_idx_flat = sim_map_np.argmax()
        y_max_512, x_max_512 = np.unravel_index(best_idx_flat, sim_map_np.shape)
        max_sim = sim_map_np[y_max_512, x_max_512]

        # 5. 还原坐标
        target_orig_w, target_orig_h = img2_pil.size
        target_x = int(x_max_512 / self.img_size * target_orig_w)
        target_y = int(y_max_512 / self.img_size * target_orig_h)

        print(f"🎯 匹配成功: ({target_x}, {target_y}), 相似度: {max_sim:.3f}")
        self._visualize(img1_pil, img2_pil, query_point, (target_x, target_y))

    def preprocess(self, img_path):
        image = Image.open(img_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, tensor

    def _visualize(self, img1, img2, pt1, pt2):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(img1)
        axes[0].add_patch(Circle(pt1, 8, color='red', fill=True))
        axes[0].set_title(f"Source {pt1}")

        axes[1].imshow(img2)
        axes[1].add_patch(Circle(pt2, 8, color='red', fill=True))
        axes[1].set_title(f"Target {pt2}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 配置区
    REPO_DIR = "/home/benson/projects/dinov3"
    # 请确保这里是你的 safetensors 权重路径
    MODEL_PATH = "/home/benson/projects/second_work/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m/model.safetensors"

    IMG1 = "pic.jpg"
    IMG2 = "pic2.jpg"
    QUERY_POINT = (554, 306)  # 源图上的点

    if not os.path.exists(IMG1):
        print("⚠️ 生成测试图片...")
        Image.new('RGB', (600, 800), 'white').save(IMG1)
        Image.new('RGB', (700, 600), 'gray').save(IMG2)

    matcher = DINOv3Matcher(REPO_DIR, MODEL_PATH)
    matcher.find_correspondence(IMG1, IMG2, QUERY_POINT)