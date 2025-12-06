import sys
import os
import torch
import torch.nn.functional as F
import matplotlib

# 强制使用非交互式后端，防止 AttributeError
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file


class DINOv3Matcher:
    def __init__(self,
                 repo_dir,
                 model_path,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 img_size=512):  # 【关键修改】改为 512 (16*32)，解决坐标漂移问题
        """
        初始化 DINOv3 匹配器
        """
        self.device = device
        self.img_size = img_size
        self.patch_size = 16  # ViT-B-16 固定为 16

        print(f"🚀 初始化... 设备: {self.device}")

        # 1. 加载架构
        self.model = self._load_architecture(repo_dir)
        self.model.to(self.device)
        self.model.eval()

        # 2. 加载权重
        self._load_weights(model_path)

        # 3. 预处理 (使用 img_size=512)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=3),  # Bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _load_architecture(self, repo_dir):
        if repo_dir not in sys.path:
            sys.path.append(repo_dir)

        # 尝试导入 ViT 定义
        try:
            from dinov3.models.vision_transformer import vit_base
        except ImportError:
            try:
                from models.vision_transformer import vit_base
            except ImportError as e:
                raise ImportError(f"无法找到模型定义，请检查路径: {repo_dir}")

        # 初始化模型 (开启 registers)
        try:
            model = vit_base(
                img_size=self.img_size,
                patch_size=16,
                num_register_tokens=4  # DINOv3 默认有4个 registers
            )
            self.has_registers = True
            print("✅ 模型架构创建成功 (含 Register Tokens)")
        except TypeError:
            print("⚠️ 本地代码不支持 Register Tokens，降级加载。")
            model = vit_base(img_size=self.img_size, patch_size=16)
            self.has_registers = False

        return model

    def _load_weights(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到权重: {model_path}")

        print(f"📦 加载权重: {model_path}")
        state_dict = load_file(model_path)
        new_dict = {}
        model_params = dict(self.model.named_parameters())

        for k, v in state_dict.items():
            # 1. 键名清洗
            k = k.replace('module.', '').replace('backbone.', '')
            k = k.replace('embeddings.patch_embeddings.weight', 'patch_embed.proj.weight')
            k = k.replace('embeddings.patch_embeddings.bias', 'patch_embed.proj.bias')
            k = k.replace('embeddings.cls_token', 'cls_token')
            k = k.replace('embeddings.mask_token', 'mask_token')
            k = k.replace('embeddings.position_embeddings', 'pos_embed')
            k = k.replace('embeddings.register_tokens', 'register_tokens')
            k = k.replace('encoder.layers.', 'blocks.')
            k = k.replace('encoder.norm.', 'norm.')

            # 2. 智能维度适配
            if k in model_params:
                target_shape = model_params[k].shape
                # 修复 [1, 768] vs [1, 1, 768]
                if v.ndim != len(target_shape):
                    if v.ndim == 2 and len(target_shape) == 3:
                        v = v.unsqueeze(1)
                    elif v.ndim == 3 and len(target_shape) == 2:
                        v = v.squeeze(1)

            # 3. Pos Embed 插值 (关键：适配 512 分辨率)
            if k == 'pos_embed' and v.shape != model_params[k].shape:
                print(f"🔄 自动调整 pos_embed: {v.shape} -> {model_params[k].shape}")
                v = self._resize_pos_embed(v, model_params[k].shape)

            new_dict[k] = v

        self.model.load_state_dict(new_dict, strict=False)
        print("✅ 权重加载完成")

    def _resize_pos_embed(self, pos_embed, expected_shape):
        """对位置编码进行双线性插值"""
        # pos_embed: [1, Total_Source_Tokens, D]
        n_special = 1 + (4 if self.has_registers else 0)  # CLS + Registers

        cls_tokens = pos_embed[:, :n_special, :]
        patch_tokens = pos_embed[:, n_special:, :]

        # Reshape to grid
        orig_size = int(patch_tokens.shape[1] ** 0.5)
        dim = patch_tokens.shape[-1]
        patch_tokens = patch_tokens.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)

        # Interpolate
        target_count = expected_shape[1] - n_special
        target_size = int(target_count ** 0.5)

        patch_tokens = F.interpolate(
            patch_tokens, size=(target_size, target_size),
            mode='bicubic', align_corners=False
        )

        # Flatten back
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((cls_tokens, patch_tokens), dim=1)

    def extract_features(self, tensor):
        """
        使用 DINOv3 原生 API 提取特征
        """
        with torch.inference_mode():
            # 【修正】forward_features 直接返回字典 (当输入为单张图时)
            out_dict = self.model.forward_features(tensor)

            # 检查是否意外返回了列表 (防御性编程)
            if isinstance(out_dict, list):
                out_dict = out_dict[0]

            # 直接获取 patch tokens
            if 'x_norm_patchtokens' in out_dict:
                patch_tokens = out_dict['x_norm_patchtokens']  # [B, N_Patches, D]
            else:
                # 打印可用键以帮助调试
                raise RuntimeError(f"未找到特征键，可用键: {out_dict.keys()}")

            # 计算网格大小
            # 强制了 img_size=512, patch_size=16 -> 32x32
            h = w = self.img_size // self.patch_size

            return patch_tokens, (h, w)

    def find_correspondence(self, img1_path, img2_path, query_point):
        # 1. 预处理
        img1_pil, tensor1 = self.preprocess(img1_path)
        img2_pil, tensor2 = self.preprocess(img2_path)

        # 2. 提取特征
        feat1, (h1, w1) = self.extract_features(tensor1)
        feat2, (h2, w2) = self.extract_features(tensor2)

        print(f"📊 特征网格大小: {h1}x{w1} (Token数: {feat1.shape[1]})")

        # 3. 坐标映射 (Pixel -> Grid)
        orig_w, orig_h = img1_pil.size
        qx, qy = query_point

        # 使用 img_size (512) 进行归一化
        grid_x = int(qx / orig_w * w1)
        grid_y = int(qy / orig_h * h1)
        grid_x = min(max(grid_x, 0), w1 - 1)
        grid_y = min(max(grid_y, 0), h1 - 1)

        query_idx = grid_y * w1 + grid_x

        # 4. 计算相似度
        q_feat = F.normalize(feat1[0, query_idx, :].unsqueeze(0), p=2, dim=-1)
        k_feat = F.normalize(feat2[0], p=2, dim=-1)

        sim = torch.mm(q_feat, k_feat.t())
        best_idx = torch.argmax(sim).item()
        max_sim = sim[0, best_idx].item()

        # 5. 坐标还原 (Grid -> Pixel)
        target_grid_y = best_idx // w2
        target_grid_x = best_idx % w2

        target_orig_w, target_orig_h = img2_pil.size
        # 映射回原图中心点
        target_x = int((target_grid_x + 0.5) / w2 * target_orig_w)
        target_y = int((target_grid_y + 0.5) / h2 * target_orig_h)

        print(f"✅ 匹配成功: 相似度 {max_sim:.3f}")
        self._visualize(img1_pil, img2_pil, query_point, (target_x, target_y))

    def preprocess(self, img_path):
        image = Image.open(img_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, tensor

    def _visualize(self, img1, img2, pt1, pt2):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(img1)
        axes[0].add_patch(Circle(pt1, 10, color='red', fill=True))
        axes[0].add_patch(Circle(pt1, 30, color='red', fill=False, lw=2))
        axes[0].set_title("Source")

        axes[1].imshow(img2)
        axes[1].add_patch(Circle(pt2, 10, color='red', fill=True))
        axes[1].add_patch(Circle(pt2, 30, color='red', fill=False, lw=2))
        axes[1].set_title("Target Match")

        plt.tight_layout()
        plt.savefig("result_final.png")
        print("🖼️ 结果已保存: result_final.png")


if __name__ == "__main__":
    # 请修改为你的实际路径
    REPO_DIR = "/home/benson/projects/dinov3"
    MODEL_PATH = "/home/benson/projects/second_work/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m/model.safetensors"

    IMG1 = "pic.jpg"
    IMG2 = "pic2.jpg"
    QUERY_POINT = (554, 306)

    matcher = DINOv3Matcher(REPO_DIR, MODEL_PATH)
    matcher.find_correspondence(IMG1, IMG2, QUERY_POINT)