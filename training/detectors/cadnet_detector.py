import math
import logging
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import CLIPModel
from detectors import DETECTOR
from training.networks.DRA import DynamicRankProjection
from training.networks.MSA import HierarchicalAttention

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='cadnet')
class CADNetDetector(nn.Module):
    def __init__(self, config=None):
        super(CADNetDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.register_buffer('current_epoch', torch.tensor(0))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.DynamicRankProjection = DynamicRankProjection(1024, 64, 1024)
        self.HierarchicalAttention = HierarchicalAttention(1024)

    def build_backbone(self, config):
        torch.cuda.empty_cache()
        gc.collect()
        clip_model = CLIPModel.from_pretrained(
        "/home/changcun/myself/DeepfakeBench_changcun/preprocessing/clip-vit-large-patch14",
        local_files_only=True,
        cache_dir=None
        )
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=1024-1)
        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['last_hidden_state']
        B, N, C = feat.shape
        H = W = int((N - 1)**0.5)
        vit_feat = feat[:, 1:, :]
        vit_feat = vit_feat.permute(0, 2, 1)
        vit_feat = vit_feat.view(B, C, H, W)
        feat_hrp = self.DynamicRankProjection(vit_feat)
        feat = self.HierarchicalAttention(feat_hrp)
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def compute_weight_loss(self):
        weight_sum_dict = {}
        num_weight_dict = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, SVDResidualLinear):
                weight_curr = module.compute_current_weight()
                if str(weight_curr.size()) not in weight_sum_dict.keys():
                    weight_sum_dict[str(weight_curr.size())] = weight_curr
                    num_weight_dict[str(weight_curr.size())] = 1
                else:
                    weight_sum_dict[str(weight_curr.size())] += weight_curr
                    num_weight_dict[str(weight_curr.size())] += 1
        loss2 = 0.0
        for k in weight_sum_dict.keys():
            _, S_sum, _ = torch.linalg.svd(weight_sum_dict[k], full_matrices=False)
            loss2 += -torch.mean(S_sum)
        loss2 /= len(weight_sum_dict.keys())
        return loss2

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss_cls = self.loss_func(pred, label)
        mask_real = label == 0
        mask_fake = label == 1
        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
            loss_real = torch.tensor(0.0, device=pred.device)
        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)
        loss = loss_cls
        loss_dict = {
            'overall': loss,
            'cls_loss': loss_cls,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pool = nn.AdaptiveAvgPool2d((1, 1))
        features = pool(features)
        features = features.view(features.size(0), -1)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict


class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)

    def compute_orthogonal_loss(self):
        if self.S_residual is not None:
            UUT = torch.cat((self.U_r, self.U_residual), dim=1) @ torch.cat((self.U_r, self.U_residual), dim=1).t()
            VVT = torch.cat((self.V_r, self.V_residual), dim=0) @ torch.cat((self.V_r, self.V_residual), dim=0).t()
            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0
        return loss

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            loss = torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
        else:
            loss = 0.0
        return loss

    def compute_fn_loss(self):
        if (self.S_residual is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            loss = weight_current_fnorm ** 2
        else:
            loss = 0.0
        return loss

def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            apply_svd_residual_to_self_attn(module, r)
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
        r = min(r, len(S))
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')
        new_module.weight_main.data.copy_(weight_main)
        U_residual = U[:, r:]
        S_residual = S[r:]
        Vh_residual = Vh[r:, :]
        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None
        return new_module
    else:
        return module