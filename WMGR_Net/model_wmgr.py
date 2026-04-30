import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ACA(nn.Module):
    """Axial Context Attention"""

    def __init__(self, channels, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.h_conv = nn.Conv2d(channels, channels, (1, h_kernel_size), 1,
                                (0, h_kernel_size // 2), groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, (v_kernel_size, 1), 1,
                                (v_kernel_size // 2, 0), groups=channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class PMRF(nn.Module):
    """Parallel Multi-Receptive Field"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7, 9, 11)):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        # 多尺度深度卷积
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, ks, 1, ks // 2,
                      groups=out_channels) for ks in kernel_sizes
        ])

        self.pw_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.pre_conv(x)

        # 保存原始特征用于残差连接
        identity = x

        # 多尺度特征提取和累加
        multi_scale_feat = x
        for dw_conv in self.dw_convs:
            multi_scale_feat = multi_scale_feat + dw_conv(multi_scale_feat)

        x = self.pw_conv(multi_scale_feat)
        x = x + identity  # 残差连接
        x = self.post_conv(x)

        return x


class LocalLatentExtractor(nn.Module):
    """局部特征与潜在类别特征提取器 (Local & Latent Category Branch)"""

    def __init__(self, feature_dim=1024, local_dim=256):
        super().__init__()

        # PMRF 多尺度卷积提取局部特征
        self.pmrf = PMRF(feature_dim, local_dim)

        # ACA 轴向上下文注意力
        self.aca = ACA(local_dim)

        # LCA 潜在类别聚合 (Latent Category Aggregation)
        self.lca = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 7, 1)  # 7个语义类别
        )

        # 潜在类别权重
        self.register_buffer('lc_weights',
                             torch.tensor([2.0, 1.5, 1.8, 1.2, 0.8, 1.0, 0.5]))

    def align_features(self, sat_feat, ground_feat):
        """对齐不同尺寸的特征图"""
        b, c, h1, w1 = sat_feat.shape
        _, _, h2, w2 = ground_feat.shape

        if h1 != h2 or w1 != w2:
            # 将ground_feat调整到与sat_feat相同的尺寸
            ground_feat = F.interpolate(ground_feat, size=(h1, w1), mode='bilinear', align_corners=False)

        return sat_feat, ground_feat

    def forward(self, sat_feat, ground_feat):
        # 保存原始特征图尺寸用于语义分析
        orig_sat_feat = sat_feat
        orig_ground_feat = ground_feat

        # 1. PMRF多尺度特征提取
        sat_local = self.pmrf(sat_feat)
        ground_local = self.pmrf(ground_feat)

        # 2. ACA上下文增强
        sat_context = self.aca(sat_local)
        ground_context = self.aca(ground_local)

        sat_enhanced = sat_local * sat_context
        ground_enhanced = ground_local * ground_context

        # 3. 特征对齐
        sat_aligned, ground_aligned = self.align_features(sat_enhanced, ground_enhanced)

        # 4. 潜在类别(LCA)特征提取 - 使用原始特征图
        sat_lc = self.lca(orig_sat_feat)
        ground_lc = self.lca(orig_ground_feat)

        lc_weights_sat = F.softmax(sat_lc, dim=1)
        lc_weights_ground = F.softmax(ground_lc, dim=1)

        # 为潜在类别权重对齐尺寸 - 调整到与enhanced特征相同尺寸
        b, c, h_enh, w_enh = sat_aligned.shape
        _, _, h_sat, w_sat = lc_weights_sat.shape
        _, _, h_ground, w_ground = lc_weights_ground.shape

        if h_sat != h_enh or w_sat != w_enh:
            lc_weights_sat = F.interpolate(lc_weights_sat, size=(h_enh, w_enh), mode='bilinear',
                                                 align_corners=False)

        if h_ground != h_enh or w_ground != w_enh:
            lc_weights_ground = F.interpolate(lc_weights_ground, size=(h_enh, w_enh), mode='bilinear',
                                                    align_corners=False)

        # LCA感知的特征聚合 (GAP)
        lc_features_sat = []
        lc_features_ground = []

        for i in range(7):
            # 每个潜在类别的加权特征
            sat_class_weight = lc_weights_sat[:, i:i + 1]
            ground_class_weight = lc_weights_ground[:, i:i + 1]

            sat_class_feat = F.adaptive_avg_pool2d(sat_aligned * sat_class_weight, 1).flatten(1)
            ground_class_feat = F.adaptive_avg_pool2d(ground_aligned * ground_class_weight, 1).flatten(1)

            lc_features_sat.append(sat_class_feat)
            lc_features_ground.append(ground_class_feat)

        lc_features_sat = torch.stack(lc_features_sat, dim=2)
        lc_features_ground = torch.stack(lc_features_ground, dim=2)

        return {
            'local_sat': sat_aligned,
            'local_ground': ground_aligned,
            'lc_sat': lc_features_sat,
            'lc_ground': lc_features_ground,
            'lc_weights_sat': lc_weights_sat,
            'lc_weights_ground': lc_weights_ground
        }


class WeatherAwareModulation(nn.Module):
    def __init__(self, in_dim=768, num_classes=4):  # ConvNeXt-B 维度通常是 768 或 1024
        super().__init__()

        # 1. 天气预测器 (Predictor)
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # 2. 修正因子生成器 (Modulation)
        self.factor_generator = nn.Sequential(
            nn.Linear(num_classes, in_dim),
            nn.Sigmoid()  # 生成 0-1 之间的权重
        )

    def forward(self, global_feat):
        # global_feat: [B, C]

        # Step 1: 显式预测天气 logits
        weather_logits = self.predictor(global_feat)  # [B, 4] -> W'

        # Step 2: 生成修正因子
        factor = self.factor_generator(weather_logits)  # [B, C]

        # Step 3: 特征修正 (Modulation)
        refined_feat = global_feat * (1.0 - factor) + global_feat

        return refined_feat, weather_logits

class WMGR_Net(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=True,
                 img_size=384,
                 enable_local_features=True):

        super(WMGR_Net, self).__init__()

        self.img_size = img_size
        self.enable_local_features = enable_local_features

        if "vit" in model_name:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
            self.is_vit = True
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.is_vit = False

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 获取特征维度
        if hasattr(self.model, 'num_features'):
            self.feature_dim = self.model.num_features
        else:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, img_size, img_size)
                dummy_output = self.model.forward_features(dummy_input)
                if len(dummy_output.shape) == 4:
                    self.feature_dim = dummy_output.shape[1]
                elif len(dummy_output.shape) == 3:
                    self.feature_dim = dummy_output.shape[2]
                else:
                    self.feature_dim = dummy_output.shape[1]

        # 局部特征提取器
        if self.enable_local_features and not self.is_vit:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, img_size, img_size)
                dummy_output = self.model.forward_features(dummy_input)
                if len(dummy_output.shape) == 4:
                    self.spatial_size = dummy_output.shape[2:]
                else:
                    self.spatial_size = None

            if self.spatial_size is not None:
                self.local_extractor = LocalLatentExtractor(self.feature_dim, 256)

        self.weather_decoupler = WeatherAwareModulation(in_dim=self.feature_dim)

    def get_config(self):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def extract_features_with_local(self, x):
        """提取特征图用于局部特征分析"""
        if not self.enable_local_features or self.is_vit:
            return self.model(x), None

        # 获取特征图
        features = self.model.forward_features(x)

        if len(features.shape) != 4:
            return self.model(x), None

        # 全局池化得到全局特征 (GAP)
        global_feat = F.adaptive_avg_pool2d(features, 1).flatten(1)

        return global_feat, features

    def forward(self, img1, img2=None, return_local=False):

        if img2 is not None:
            # 双图像输入
            if not self.enable_local_features or self.is_vit:
                # 无法提取局部特征的情况 (如 ViT)
                global_feat1 = self.model(img1)
                global_feat2 = self.model(img2)
                
                # 依然尝试预测天气并进行特征解耦
                refined_global1, weather_logits = self.weather_decoupler(global_feat1)

                if return_local:
                    return (refined_global1, None), (global_feat2, None), weather_logits
                else:
                    return refined_global1, global_feat2
            else:
                # 局部特征模式
                global_feat1, feat_map1 = self.extract_features_with_local(img1)
                global_feat2, feat_map2 = self.extract_features_with_local(img2)
                # 预测并解耦
                refined_global1, weather_logits = self.weather_decoupler(global_feat1)

                if feat_map1 is not None and feat_map2 is not None:
                    # 提取局部特征
                    local_dict = self.local_extractor(feat_map1, feat_map2)

                    local_info1 = {
                        'local': local_dict['local_sat'],
                        'lc': local_dict['lc_sat'],
                        'lc_weights': local_dict['lc_weights_sat']
                    }

                    local_info2 = {
                        'local': local_dict['local_ground'],
                        'lc': local_dict['lc_ground'],
                        'lc_weights': local_dict['lc_weights_ground']
                    }

                    if return_local:
                        return (refined_global1, local_info1), (global_feat2, local_info2), weather_logits
                    else:
                        return refined_global1, global_feat2
                else:
                    if return_local:
                        return (refined_global1, None), (global_feat2, None), weather_logits
                    else:
                        return refined_global1, global_feat2

        else:
            # 单图像输入
            if return_local and self.enable_local_features and not self.is_vit:
                global_feat, feat_map = self.extract_features_with_local(img1)
                return global_feat, {'feature_map': feat_map}
            else:
                return self.model(img1)


class WMGRLoss(nn.Module):
    def __init__(self, loss_function, device='cuda', local_weight=0.2, lc_weight=0.1):
        super().__init__()
        self.loss_function = loss_function
        self.device = device
        self.local_weight = local_weight
        self.lc_weight = lc_weight

        # 记录各部分损失
        self.last_global_loss = 0.0
        self.last_local_loss = 0.0
        self.last_lc_loss = 0.0

    def forward(self, output1, output2, logit_scale):
        if isinstance(output1, tuple) and output1[1] is not None:
            # 有局部特征
            global_feat1, local_dict1 = output1
            global_feat2, local_dict2 = output2

            # 全局InfoNCE损失
            global_loss = self.global_infonce(global_feat1, global_feat2, logit_scale)
            # 安全地获取loss值
            if hasattr(global_loss, 'item'):
                self.last_global_loss = global_loss.item()
            else:
                self.last_global_loss = float(global_loss)

            total_loss = global_loss

            # 局部对齐损失 (Alignment)
            if 'local' in local_dict1 and self.local_weight > 0:
                local_loss = self.local_alignment_loss(
                    local_dict1['local'], local_dict2['local']
                )
                # 安全地获取loss值
                if hasattr(local_loss, 'item'):
                    self.last_local_loss = local_loss.item()
                else:
                    self.last_local_loss = float(local_loss)
                total_loss = total_loss + self.local_weight * local_loss
            else:
                self.last_local_loss = 0.0

            # 潜在类别对比损失 (Latent Category Contrastive)
            if ('lc' in local_dict1 and self.lc_weight > 0):
                lc_loss = self.lc_contrastive_loss(
                    local_dict1['lc'], local_dict2['lc'],
                    local_dict1['lc_weights'], local_dict2['lc_weights'],
                    logit_scale
                )
                # 安全地获取loss值
                if hasattr(lc_loss, 'item'):
                    self.last_lc_loss = lc_loss.item()
                else:
                    self.last_lc_loss = float(lc_loss)
                total_loss = total_loss + self.lc_weight * lc_loss
            else:
                self.last_lc_loss = 0.0

            return total_loss
        else:
            # 只有全局特征
            if isinstance(output1, tuple):
                global_feat1, global_feat2 = output1[0], output2[0]
            else:
                global_feat1, global_feat2 = output1, output2

            global_loss = self.global_infonce(global_feat1, global_feat2, logit_scale)
            if hasattr(global_loss, 'item'):
                self.last_global_loss = global_loss.item()
            else:
                self.last_global_loss = float(global_loss)
            self.last_local_loss = 0.0
            self.last_lc_loss = 0.0

            return global_loss

    def global_infonce(self, feat1, feat2, logit_scale):
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        logits_per_image1 = logit_scale * feat1 @ feat2.T
        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss = (self.loss_function(logits_per_image1, labels) +
                self.loss_function(logits_per_image2, labels)) / 2
        return loss

    def local_alignment_loss(self, local_sat, local_ground):
        if local_sat is None or local_ground is None:
            return torch.tensor(0.0, device=self.device)

        if torch.isnan(local_sat).any() or torch.isnan(local_ground).any():
            return torch.tensor(0.0, device=self.device)

        local_sat = F.normalize(local_sat, dim=1, eps=1e-8)
        local_ground = F.normalize(local_ground, dim=1, eps=1e-8)

        similarity = (local_sat * local_ground).sum(dim=1, keepdim=True)

        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            return torch.tensor(0.0, device=self.device)

        loss = (1.0 - similarity).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device)

        return loss

    def lc_contrastive_loss(self, lc_feat1, lc_feat2, weights1, weights2, logit_scale):
        if lc_feat1 is None or lc_feat2 is None:
            return torch.tensor(0.0, device=self.device)

        if torch.isnan(lc_feat1).any() or torch.isnan(lc_feat2).any():
            return torch.tensor(0.0, device=self.device)

        B, C, num_cls = lc_feat1.shape
        lc_weights = torch.tensor([2.0, 1.5, 1.8, 1.2, 0.8, 1.0, 0.5], device=lc_feat1.device)

        loss = 0
        valid_classes = 0

        for cls_idx in range(num_cls):
            try:
                cls_weight1 = weights1[:, cls_idx].mean(dim=[1, 2])
                cls_weight2 = weights2[:, cls_idx].mean(dim=[1, 2])

                if torch.isnan(cls_weight1).any() or torch.isnan(cls_weight2).any():
                    continue

                valid_mask = (cls_weight1 > 0.05) & (cls_weight2 > 0.05)

                if valid_mask.sum() > 1:
                    cls_feat1 = lc_feat1[valid_mask, :, cls_idx]
                    cls_feat2 = lc_feat2[valid_mask, :, cls_idx]

                    if torch.isnan(cls_feat1).any() or torch.isnan(cls_feat2).any():
                        continue

                    cls_loss = self.global_infonce(cls_feat1, cls_feat2, logit_scale)

                    if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                        continue

                    loss += lc_weights[cls_idx] * cls_loss
                    valid_classes += 1
            except Exception as e:
                continue

        if valid_classes == 0:
            return torch.tensor(0.0, device=self.device)

        final_loss = loss / valid_classes

        if torch.isnan(final_loss) or torch.isinf(final_loss):
            return torch.tensor(0.0, device=self.device)

        return final_loss
