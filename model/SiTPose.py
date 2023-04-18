import torch
from torchvision import models as pre_model
import torch.nn as nn
from torch.nn import functional as F
import math
from model.CCT import CCT

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out + residual
        return F.relu(out)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
      
class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0. ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x1, x2):
        B, N, C = x1.shape

        qkv1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]   # make torchscript happy (cannot use tensor as tuple)

        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]   # make torchscript happy (cannot use tensor as tuple)

        # q2, k1, v1
        attn_1 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn_1 = attn_1.softmax(dim=-1)
        attn_1 = self.attn_drop(attn_1)

        x1 = (attn_1 @ v1).transpose(1, 2).reshape(B, N, C)

        # q1, k2, v2
        attn_2 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn_2 = attn_2.softmax(dim=-1)
        attn_2 = self.attn_drop(attn_2)

        x2 = (attn_2 @ v2).transpose(1, 2).reshape(B, N, C)
        
        x1 = self.proj(x1)
        x2 = self.proj(x2)

        x1 = self.proj_drop(x1)
        x2 = self.proj_drop(x2)

        return x2, x1 

class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.5, act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop,

                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, feat1,feat2):
        x1, x2 = self.cross_attn(self.norm1(feat1), self.norm1(feat2))
        feat1 = feat1 + self.drop_path(x1) 
        res_feat1 =  self.drop_path(self.mlp(self.norm2(feat1)))
        feat2 = feat2 + self.drop_path(x2) 
        res_feat2 =  self.drop_path(self.mlp(self.norm2(feat2)))

        res_feat1 = (res_feat1.permute(0,-1,-2)).reshape(res_feat1.shape[0],res_feat1.shape[2],int(math.sqrt(res_feat1.shape[1])),int(math.sqrt(res_feat1.shape[1])))
        res_feat2 = (res_feat2.permute(0,-1,-2)).reshape(res_feat2.shape[0],res_feat2.shape[2],int(math.sqrt(res_feat2.shape[1])),int(math.sqrt(res_feat2.shape[1])))
        
        return res_feat1,res_feat2


class SiTPose(nn.Module):
    def __init__(self,emb_dim,rsl,num_head_cross_attn = 8,num_head_attn = 6,feature_encoder_layer = 14):
        super(SiTPose, self).__init__()
        self.drop_path = DropPath(0.5) 
        self.resnet_layer1 = nn.Sequential( *list(pre_model.resnet34(pretrained=True).children() )[0:6])
        self.cross_block1 =CrossBlock(dim=128, num_heads=num_head_cross_attn)
        self.resnet_layer2 = nn.Sequential( *list(pre_model.resnet34(pretrained=True).children() )[6:7])
        self.cross_block2 =CrossBlock(dim=256, num_heads=num_head_cross_attn)
        self.resnet_layer3 = nn.Sequential( *list(pre_model.resnet34(pretrained=True).children() )[7:8])
        self.main_stream = self._make_layer(1024, emb_dim, 3)

        self.reshape_fused = nn.Flatten(2,3)
        self.reshape_feat = nn.Flatten(1,3)
        self.featureEncoder = CCT(
                                        embedding_dim = emb_dim,
                                        num_layers = feature_encoder_layer,
                                        num_heads = num_head_attn,
                                        mlp_radio = 3.,
                                        num_classes = 512,
                                        positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
                                        resscale = rsl,
                                        use_dualSP = True,
                                    )
        self.norm1 = nn.LayerNorm(64)
        self.fc_q = nn.Linear(512,4)
        self.fc_t = nn.Linear(512,3)
        self.dropout = nn.Dropout(0.3)


    def _make_layer(self, inchannel, outchannel, block_num, stride=2):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def siamese_extractor(self,x1,x2):
        feat1 = self.resnet_layer1(x1)
        feat2 = self.resnet_layer1(x2)
        contra_feat1 = self.reshape_feat(feat1)
        contra_feat2 = self.reshape_feat(feat2)
        cross_feat1,cross_feat2 = self.cross_block1(  self.reshape_fused(feat1).permute(0,-1,-2)  ,   self.reshape_fused(feat2).permute(0,-1,-2)   )

        feat1 = feat1 + self.drop_path(cross_feat1) 
        feat2 = feat2 + self.drop_path(cross_feat2) 

        feat1 = self.resnet_layer2(feat1)
        feat2 = self.resnet_layer2(feat2)

        cross_feat1,cross_feat2 = self.cross_block2(  self.reshape_fused(feat1).permute(0,-1,-2)  ,   self.reshape_fused(feat2).permute(0,-1,-2)   )
        feat1 = feat1 + self.drop_path(cross_feat1) 
        feat2 = feat2 + self.drop_path(cross_feat2) 

        feat1 = self.resnet_layer3(feat1)
        feat2 = self.resnet_layer3(feat2)

        feat = torch.cat( (feat1,feat2),1 )

        feat = self.main_stream(feat)
        return feat,self.reshape_feat(feat1),self.reshape_feat(feat2)
    
    def norm_Quaternion(self,quat):
        normalized = quat.norm(dim=-1).unsqueeze(-1)
        eps = torch.ones_like(normalized) * .01
        pred_q = quat / torch.max(normalized, eps)
        return pred_q

    def forward(self,x1,x2):

        feat,feat1,feat2 = self.siamese_extractor(x1,x2)
        feat = self.reshape_fused(feat).permute(0,-1,-2)
        feat = self.featureEncoder(feat)
        
        pred_q = self.fc_q(self.dropout(feat))
        
        pred_t = self.fc_t(self.dropout(feat))

        return  pred_q,pred_t,feat1,feat2

def SiTPose_1():

    return SiTPose(emb_dim = 384,rsl = 0.9,num_head_cross_attn = 8, num_head_attn = 6,feature_encoder_layer = 14)
