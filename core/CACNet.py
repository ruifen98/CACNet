#注意sampling_rate=0.6,DGCNN=6
import torch
import torch.nn as nn
from loss import batch_episym
import torch.nn.functional as F
def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1, 1)
    if len(idx.shape) == 3:
        batch_indices = batch_indices.repeat(1, idx.shape[1], idx.shape[2])
        new_points = points[batch_indices, idx, :]  # [B, S, K, C]
    else:
        batch_indices = batch_indices.repeat(1, idx.shape[1])
        new_points = points[batch_indices, idx, :]  # [B, S, C]
    return new_points
class SplitFeatureEncoder(nn.Module):
    def __init__(self, out_channel=128, use_geo=True, use_tri=True):
        super().__init__()
        self.out_channel = out_channel
        self.half_channel = out_channel // 2
        self.use_geo = use_geo
        self.use_tri = use_tri
        self.conv_A = nn.Sequential(
            nn.Conv2d(2, self.half_channel, kernel_size=1),
            nn.BatchNorm2d(self.half_channel),
            nn.ReLU(inplace=True),
            ResNet_Block(self.half_channel, self.half_channel)
        )
        self.conv_B = nn.Sequential(
            nn.Conv2d(2, self.half_channel, kernel_size=1),
            nn.BatchNorm2d(self.half_channel),
            nn.ReLU(inplace=True),
            ResNet_Block(self.half_channel, self.half_channel)
        )
        if self.use_geo:
            self.geo_encoder = nn.Sequential(
                nn.Conv2d(3, self.half_channel // 2, kernel_size=1),
                nn.BatchNorm2d(self.half_channel // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.half_channel // 2, self.half_channel, kernel_size=1)
            )
        if self.use_tri:
            self.triangle_conv = nn.Sequential(
                nn.Conv2d(5, self.half_channel, kernel_size=1),
                nn.BatchNorm2d(self.half_channel),
                nn.ReLU(inplace=True),
                ResNet_Block(self.half_channel, self.half_channel)
            )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def build_triangle_feat(self, x):
        B, _, N, _ = x.shape
        x_flat = x.squeeze(3).transpose(1, 2)
        dist = torch.cdist(x_flat, x_flat)
        dist = dist + 1e-6 * torch.randn_like(dist)
        knn_idx = dist.topk(k=3, largest=False)[1][:, :, 1:3]
        neighbor = index_points(x_flat, knn_idx)
        center = x_flat.unsqueeze(2)
        v1 = neighbor[:, :, 0, :] - center.squeeze(2)
        v2 = neighbor[:, :, 1, :] - center.squeeze(2)
        len1 = torch.norm(v1, dim=-1)
        len2 = torch.norm(v2, dim=-1)
        scale = (len1 + len2) / 2 + 1e-6
        len1 = len1 / scale
        len2 = len2 / scale
        cos_angle = F.cosine_similarity(v1, v2, dim=-1)
        ratio = len1 / (len2 + 1e-6)
        area = torch.abs(v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0])
        area = area / (scale**2 + 1e-6)
        feat = torch.stack([len1, len2, cos_angle, ratio, area], dim=1)
        return feat.unsqueeze(3)
    def forward(self, x):
        coords_raw = x[:, :4, :, :]
        x_A_raw = coords_raw[:, :2, :, :]
        x_B_raw = coords_raw[:, 2:, :, :]
        feat_A = self.conv_A(x_A_raw)
        feat_B = self.conv_B(x_B_raw)
        tri_A, tri_B = None, None
        if self.use_tri:
            tri_A = self.triangle_conv(self.build_triangle_feat(x_A_raw))
            tri_B = self.triangle_conv(self.build_triangle_feat(x_B_raw))
            feat_A = feat_A + 0.5 * tri_A
            feat_B = feat_B + 0.5 * tri_B
        if self.use_geo:
            delta_coord = x_A_raw - x_B_raw
            dist = torch.norm(delta_coord, dim=1, keepdim=True)
            geo_feat = self.geo_encoder(torch.cat([delta_coord, dist], dim=1))
            feat_A = feat_A + geo_feat
            feat_B = feat_B + geo_feat
        split_feat = torch.cat([feat_A, feat_B], dim=1)
        split_feat = self.fuse_conv(split_feat)
        return split_feat, tri_A, tri_B

class DeepBidirectionalConsistency(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=32, top_k=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.desc_refiner = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim, 1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim, 1),
            nn.BatchNorm2d(self.feat_dim),
        )
        in_channel = self.feat_dim * 2 + 3
        self.sim_encoder_main = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
        )
        self.sim_encoder_shortcut = nn.Conv2d(in_channel, hidden_dim, 1)
        nn.init.kaiming_normal_(self.sim_encoder_shortcut.weight)
        nn.init.zeros_(self.sim_encoder_shortcut.bias)
        self.attention_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        self.consist_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        self.temp = nn.Parameter(torch.tensor(0.1))
    def forward(self, desc_A, desc_B, tri_A, tri_B):
        B, C, N, _ = desc_A.shape
        feat_A = desc_A + self.desc_refiner(desc_A)
        feat_B = desc_B + self.desc_refiner(desc_B)
        feat_diff = torch.abs(feat_A - feat_B)
        cross_interaction = feat_A * feat_B
        feat_A_flat = feat_A.squeeze(3).transpose(1, 2)
        feat_B_flat = feat_B.squeeze(3).transpose(1, 2)
        temp = torch.clamp(self.temp, min=1e-3)
        sim_matrix = torch.bmm(feat_A_flat, feat_B_flat.transpose(1, 2)) / temp
        top_sim_A2B = torch.topk(sim_matrix, k=self.top_k, dim=-1)[0].mean(dim=-1) + 1e-6
        top_sim_B2A = torch.topk(sim_matrix.transpose(1,2), k=self.top_k, dim=-1)[0].mean(dim=-1) + 1e-6
        diag_sim = torch.diagonal(sim_matrix, dim1=1, dim2=2)
        diff_A2B = torch.tanh(diag_sim - top_sim_A2B).unsqueeze(1).unsqueeze(3)
        diff_B2A = torch.tanh(diag_sim - top_sim_B2A).unsqueeze(1).unsqueeze(3)
        if tri_A is None or tri_B is None:
            tri_feat = torch.zeros_like(diff_A2B)
        else:
            tri_diff = torch.abs(tri_A - tri_B).mean(1, keepdim=True)
            tri_sim = F.cosine_similarity(tri_A.squeeze(3), tri_B.squeeze(3), dim=1).unsqueeze(1).unsqueeze(3)
            tri_feat = (tri_diff + tri_sim) / 2
        fused_feat = torch.cat([feat_diff, cross_interaction, diff_A2B, diff_B2A, tri_feat], dim=1)
        sim_feat = F.relu(self.sim_encoder_main(fused_feat) + self.sim_encoder_shortcut(fused_feat))
        attn = self.attention_gate(sim_feat)
        sim_feat = sim_feat * attn + sim_feat * (1 - attn) * 0.5
        consist_prob = self.consist_predictor(sim_feat)
        return consist_prob, sim_feat
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
            trans(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x

        return out
class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1)
        )
    def forward(self, x):
        embed = self.conv(x)  # b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out

class OABlock(nn.Module):
    def __init__(self, net_channels, depth=6, clusters=250):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size=1)
    def forward(self, data):
        x1_1 = data
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = torch.cat([x1_1, x_up], dim=1)
        return self.shot_cut(out)

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx[:, :, :]
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature
class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)
class LGDAFN(nn.Module):
    def __init__(self, in_channels=128, reduction=4, use_residual=True):
        super(LGDAFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        inter_channels = int(self.in_channels // reduction)
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.global_att_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.global_local_diff = nn.Sequential(
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.use_residual = use_residual
    def forward(self, x):
        batch_size, C, N, _ = x.shape
        input_conv = self.conv_in(x)
        local_scale = self.local_att(input_conv)
        global_avg_scale = self.global_att_avg(input_conv)
        global_mean = torch.mean(input_conv, dim=2, keepdim=True)
        diff_feat = input_conv - global_mean
        diff_scale = self.global_local_diff(diff_feat)
        scale_out = local_scale + global_avg_scale + diff_scale
        scale_out = self.sigmoid(scale_out)
        weighted_feat = input_conv * scale_out
        output = self.conv_out(weighted_feat)
        output = output + input_conv
        if self.use_residual:
            output = output + x
        return output
def batch_symeig(X):
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv
def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]  # [32,500] logits的第一层
    weights = logits[:, 1, :, 0]  # [32,500] logits的第二层
    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    x_shp = x_in.shape
    x_in = x_in.squeeze(1)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)
    v = batch_symeig(XwX)#v[32,9,9]
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat
class DGCNN_MAX_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_MAX_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
    def forward(self, features):
        # feature[32,128,2000,1]
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.unsqueeze(3)
        return out
class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm( w.transpose(1, 2), w)
        return A
    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad():
            A = self.attention(w)
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()
        return out
    def forward(self, x, w):
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out
class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5, use_geo=True, use_tri=True):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.half_channel = out_channel // 2
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.bidirectional_consist = DeepBidirectionalConsistency(
            feat_dim=out_channel // 2,
            hidden_dim=out_channel // 4,
            top_k=10
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        self.gcn = GCN_Block(self.out_channel)
        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            LGDAFN(in_channels=self.out_channel),
            DGCNN_MAX_Block(int(self.k_num * 2), self.out_channel),
            LGDAFN(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            LGDAFN(in_channels=self.out_channel),
            OABlock(self.out_channel, clusters=256),
            LGDAFN(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.embed_0_1 = nn.Sequential(
            ResNet_Block(self.half_channel, self.half_channel, pre=False),
            OABlock(self.half_channel, clusters=128),
            LGDAFN(in_channels=self.half_channel),
            ResNet_Block(self.half_channel, self.half_channel, pre=False),
        )
        self.embed_0_2 = nn.Sequential(
            ResNet_Block(self.half_channel, self.half_channel, pre=False),
            OABlock(self.half_channel, clusters=128),
            LGDAFN(in_channels=self.half_channel),
            ResNet_Block(self.half_channel, self.half_channel, pre=False),
        )
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            OABlock(self.out_channel, clusters=128),
            LGDAFN(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_MAX_Block(self.k_num, self.out_channel),
            LGDAFN(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.split_encoder = SplitFeatureEncoder(
            out_channel=out_channel,
            use_geo=use_geo,
            use_tri=use_tri
        )
        self.attn_fuse = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        self.sim_feat_proj = nn.Conv2d(32, self.out_channel, kernel_size=1)
        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))
            self.cross_key_fuse = nn.Sequential(
                nn.Conv2d(2 * self.out_channel, self.out_channel, kernel_size=1),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(inplace=True)
            )
    def down_sampling(self, x, y, weights, indices, features=None, predict=False, key_outs=None):
        B, _, N, _ = x.size()
        keep_num = int(N * self.sr)
        indices = indices[:, :keep_num]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)
        key_outs_ds = []
        if key_outs is not None:
            for out in key_outs:
                out_indices = indices.repeat(1, self.out_channel, 1, 1)
                out_ds = torch.gather(out, dim=2, index=out_indices)
                key_outs_ds.append(out_ds)
        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out, indices.squeeze(1).squeeze(-1), key_outs_ds
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out, indices.squeeze(1).squeeze(-1), key_outs_ds
    def forward(self, x, y, cross_key_outs=None):
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()
        out_split, tri_A, tri_B = self.split_encoder(out)
        out = self.conv(out)
        self.out1 = out
        if cross_key_outs is not None:
            cross_key_cat = torch.cat([out, cross_key_outs], dim=1)
            out = out + self.cross_key_fuse(cross_key_cat)
        out = self.embed_0(out)
        feat_A_deep = out_split[:, :64, :, :]
        feat_B_deep = out_split[:, 64:, :, :]
        out_split_A = self.embed_0_1(feat_A_deep)
        out_split_B = self.embed_0_2(feat_B_deep)
        consist_prob, sim_feat = self.bidirectional_consist(
            desc_A=out_split_A, desc_B=out_split_B, tri_A=tri_A, tri_B=tri_B
        )
        w0 = self.linear_0(out).view(B, -1)
        out_g = self.gcn(out, w0.detach())
        out = out_g + out
        self.out2 = out
        sim_feat_128 = self.sim_feat_proj(sim_feat)
        sim_feat_weighted = sim_feat_128 * consist_prob
        out = out + sim_feat_weighted
        out = self.embed_1(out)
        self.out3 = out
        w1 = self.linear_1(out).view(B, -1)
        consist_prob_flat = consist_prob.squeeze(1).squeeze(-1)
        consist_prob_clamped = torch.clamp(consist_prob_flat, min=0.2, max=1.0)
        w1 = w1 * consist_prob_clamped
        key_outs = [self.out1, self.out2, self.out3]
        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, ds_indices, key_outs_ds = self.down_sampling(
                x, y, w0, indices, None, self.predict, key_outs=key_outs
            )
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], ds_indices, key_outs_ds
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out, indices_ds1, key_outs_ds = self.down_sampling(
                x, y, w0, indices, out, self.predict, key_outs=key_outs
            )
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)
            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat, key_outs_ds
class PLSNet(nn.Module):
    def __init__(self, config, ablate_geo=True, ablate_tri=True):
        super(PLSNet, self).__init__()
        self.use_geo = not ablate_geo
        self.use_tri = not ablate_tri
        self.ds_0 = DS_Block(
            initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=0.6,
            use_geo=self.use_geo, use_tri=self.use_tri
        )
        self.ds_1 = DS_Block(
            initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr,
            use_geo=self.use_geo, use_tri=self.use_tri
        )
        self.ds0_conv_adjust = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.num_substages = 3
        self.substage_att = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for _ in range(self.num_substages)
        ])
    def forward(self, x, y):
        B, _, N, _ = x.shape
        x1, y1, ws0, w_ds0, ds0_indices, key_outs_ds0 = self.ds_0(x, y)
        ds0_out1, ds0_out2, ds0_out3= key_outs_ds0
        ds0_substages = [ds0_out1, ds0_out2, ds0_out3]
        weights = [att(feat) for att, feat in zip(self.substage_att, ds0_substages)]
        weights_sum = sum(weights) + 1e-8
        norm_weights = [w / weights_sum for w in weights]
        ds0_multistage_feat = sum(feat * w for feat, w in zip(ds0_substages, norm_weights))
        ds0_all_feat = self.ds0_conv_adjust(ds0_multistage_feat)
        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)
        x2, y2, ws1, w_ds1, e_hat,key_outs_ds1 = self.ds_1(
            x_, y1, cross_key_outs=ds0_all_feat
        )
        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat
CACNet = PLSNet