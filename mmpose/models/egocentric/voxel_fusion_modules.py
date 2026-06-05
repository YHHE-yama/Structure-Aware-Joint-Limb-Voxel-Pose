# voxel_fusion_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class JointHeatmapEncoder_before(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Flatten(),
            nn.Linear(32, feature_dim)
        )

    def forward(self, joint_heatmap):  # [B, J, D, H, W]
        B, J, D, H, W = joint_heatmap.shape
        joint_feat = self.encoder(joint_heatmap.reshape(B * J, 1, D, H, W)).view(B, J, -1)

        return joint_feat


class LimbHeatmapEncoder_before(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Flatten(),
            nn.Linear(32, feature_dim)
        )


    def forward(self, limb_heatmap):  # [B, L, D, H, W]
        
        B, J, D, H, W = limb_heatmap.shape
        limb_feat = self.encoder(limb_heatmap.reshape(B * J, 1, D, H, W)).view(B, J, -1)

        return limb_feat

class JointHeatmapEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),  # 64^3 -> 32^3
            nn.Conv3d(in_channels, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(16, feature_dim),
        )

    def forward(self, joint_heatmap):  # [B, J, D, H, W]
        B, J, D, H, W = joint_heatmap.shape
        joint_feat = self.encoder(joint_heatmap.reshape(B * J, 1, D, H, W)).view(B, J, -1)
        return joint_feat


class LimbHeatmapEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),  # 64^3 -> 32^3
            nn.Conv3d(in_channels, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(16, feature_dim),
        )

    def forward(self, limb_heatmap):  # [B, L, D, H, W]
        
        B, J, D, H, W = limb_heatmap.shape
        limb_feat = self.encoder(limb_heatmap.reshape(B * J, 1, D, H, W)).view(B, J, -1)
        return limb_feat


class PropagationFusionUnit(nn.Module):
    def __init__(self, feature_dim=128, dropout=0.1):
        super().__init__()
        self.gate_joint = nn.Linear(feature_dim, feature_dim)
        self.gate_limb = nn.Linear(feature_dim, feature_dim)
        self.update = nn.Linear(feature_dim * 3, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)
        nn.init.constant_(self.gate_joint.bias, -2.0)
        nn.init.constant_(self.gate_limb.bias, -2.0)


    def forward(self, parent_feat, child_feat, limb_feat, return_debug=False):

        g_joint = torch.sigmoid(self.gate_joint(child_feat))  # [B, J, C]
        g_limb = torch.sigmoid(self.gate_limb(child_feat))    # [B, J, C]

        parent_gated = parent_feat * g_joint
        limb_gated = limb_feat * g_limb

        fusion = torch.cat([child_feat, parent_gated, limb_gated], dim=-1)  # [B, J, 3C]
        updated_child = self.update(fusion)  # [B, J, C]
        updated_child = updated_child*0.1 + child_feat
        updated_child = self.norm(updated_child)                         
        updated_child = self.dropout(updated_child)                       

        if return_debug:
            debug_info = {
                "g_joint_mean": g_joint.mean().item(),
                "g_joint_std": g_joint.std().item(),
                "g_limb_mean": g_limb.mean().item(),
                "g_limb_std": g_limb.std().item(),
                "child_feat_std": child_feat.std().item(),
                "updated_child_std": updated_child.std().item()
            }
            return updated_child, debug_info
        else:
            return updated_child


class JointVoxelDecoder(nn.Module):
    def __init__(self, feature_dim=128, output_shape=(64, 64, 64)):
        super().__init__()
        self.init_size = (8, 8, 8)  
        self.output_shape = output_shape

        self.fc = nn.Linear(feature_dim, 64 * self.init_size[0] * self.init_size[1] * self.init_size[2])

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),   
        )

        nn.init.zeros_(self.fc.weight);  nn.init.zeros_(self.fc.bias)
        last = self.deconv[-1]
        nn.init.zeros_(last.weight);     nn.init.zeros_(last.bias)


    def forward(self, x):  # x: [B, J, C]
        B, J, C = x.shape
        #print('input x shape:', x.shape)
        x = x.view(B * J, C)                         # [B*J, C]
        #print('flattened x shape:', x.shape)
        x = self.fc(x)                               # [B*J, 64 * 8 * 8 * 8]
        #print('fc output shape:', x.shape)
        x = x.view(B * J, 64, *self.init_size)       # [B*J, 64, 8, 8, 8]
        x = self.deconv(x)                           # [B*J, 1, 64, 64, 64]
        x = x.view(B, J, *self.output_shape)         # [B, J, 64, 64, 64]
        return x

class DynamicLimbSelector(nn.Module):
    def __init__(self, C=128, prior_dim=0, use_prior=False):
        super().__init__()
        in_dim = C * 2 + (prior_dim if use_prior else 0)
        self.score = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        nn.init.constant_(self.score[-1].bias, -2.0)
        self.use_prior = use_prior
        self.prior_dim = prior_dim

    @torch.no_grad()
    def _build_neighbors(self, limb_map, J):

        neighbors = [[] for _ in range(J)]
        for (a, b), idx in limb_map.items():
            neighbors[a].append(idx)
            neighbors[b].append(idx)
        return neighbors

    def forward(self, f_joint, f_limb, limb_map):
        B, J, C = f_joint.shape
        device = f_joint.device
        hat = torch.zeros(B, J, C, device=device)

        neighbors = self._build_neighbors(limb_map, J)  # list[J] of list[l_idx]

        for j in range(J):
            cand = neighbors[j]
            if len(cand) == 0:
                continue
            fj = f_joint[:, j].unsqueeze(1).expand(-1, len(cand), -1)  # [B,K,C]
            fl = f_limb[:, cand]                                       # [B,K,C]
            x = torch.cat([fj, fl], dim=-1)                            # [B,K,2C]
            s = self.score(x).squeeze(-1)                              # [B,K]
            a = torch.softmax(s, dim=-1)                               # [B,K]
            hat[:, j] = torch.einsum('bk,bkc->bc', a, fl)              # [B,C]
        return hat






