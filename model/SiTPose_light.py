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
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

        
class CrossBlock_conv(nn.Module):
    def __init__(self, dim,input_channel,output_channel,num_heads = 4,drop_rate = .1):
        super().__init__()
        
        self.num_heads =num_heads
        head_dim = dim // (num_heads*num_heads)
        self.scale = head_dim ** -0.5
        self.norm1 = nn.BatchNorm2d(input_channel)
        self.norm2 = nn.BatchNorm2d(input_channel)
        self.attn_drop = nn.Dropout(drop_rate)

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.qkv_feat1 = nn.Sequential(nn.Conv2d(input_channel, input_channel*3,kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm2d(input_channel*3) )
        self.qkv_feat2 = nn.Sequential(nn.Conv2d(input_channel, input_channel*3,kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm2d(input_channel*3) )
        
        self.proj = nn.Sequential( nn.ReLU(),
                                  nn.Conv2d(input_channel, input_channel, 1),
                                  nn.BatchNorm2d(input_channel) )

    def forward(self,feat1,feat2):
        B,N,W,H = feat1.shape
        #cross attention
        qkv1 = self.qkv_feat1(feat1).reshape(B, N, 3, self.num_heads*self.num_heads, W // self.num_heads, H // self.num_heads).permute(2, 0, 3, 1, 4, 5)
        qkv2 = self.qkv_feat2(feat2).reshape(B, N, 3, self.num_heads*self.num_heads, W // self.num_heads, H // self.num_heads).permute(2, 0, 3, 1, 4, 5)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        attn_1 = ((q2 @ k1.transpose(-2, -1)) * self.scale).flatten(2)
        attn_2 = ((q1 @ k2.transpose(-2, -1)) * self.scale).flatten(2)
        BB,NN,CC = attn_1.shape

        attn_1 =  self.attn_drop(attn_1.softmax(dim=-1)).reshape(B, NN,N, int((CC/N)** 0.5),int((CC/N)** 0.5))
        x1 = (attn_1 @ v1).reshape(B,N,W,H)
        attn_2 =  self.attn_drop(attn_2.softmax(dim=-1)).reshape(B, NN,N, int((CC/N)** 0.5),int((CC/N)** 0.5))
        x2 = (attn_2 @ v2).reshape(B,N,W,H)
        x1 =  self.proj(x1)
        x2 =  self.proj(x2)
        #fusion
        feat1 = self.drop_path(self.norm1( feat1 + self.drop_path(x1) ))
        feat2 = self.drop_path(self.norm2( feat2 + self.drop_path(x2) ))

        return feat1,feat2

        
class SiTPose(nn.Module):
    def __init__(self,emb_dim,rsl,num_head_attn = 6,feature_encoder_layer = 2):
        super(SiTPose, self).__init__()
        self.drop_path = DropPath(0.1) 

        self.resnet_layer1 = nn.Sequential( *list(pre_model.resnet18(pretrained=True).children() )[0:6])
        self.resnet_layer2 = nn.Sequential( *list(pre_model.resnet18(pretrained=True).children() )[6:7])
        self.resnet_layer3 = nn.Sequential( *list(pre_model.resnet18(pretrained=True).children() )[7:8])
        
        self.ctb = CrossBlock_conv(dim=128,input_channel = 128,output_channel=128,num_heads =4)
        self.ctb2 = CrossBlock_conv(dim=256,input_channel = 256,output_channel=256,num_heads =2)
        self.reshape_fused = nn.Flatten(2,3)
        self.reshape_feat = nn.Flatten(1,3)
        self.featureEncoder = CCT(      embedding_dim = emb_dim,
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
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(1024, 384, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

    def siamese_extractor(self,x1,x2):
        feat1 = self.resnet_layer1(x1)
        feat2 = self.resnet_layer1(x2)

        contra_feat1 = self.reshape_feat(feat1)
        contra_feat2 = self.reshape_feat(feat2)
        cross_feat1,cross_feat2 = self.ctb(feat1,feat2)

        feat1 = feat1 + self.drop_path(cross_feat1) 
        feat2 = feat2 + self.drop_path(cross_feat2) 

        feat1 = self.resnet_layer2(feat1)
        feat2 = self.resnet_layer2(feat2)

        cross_feat1,cross_feat2 = self.ctb2(feat1,feat2)
        feat1 = feat1 + self.drop_path(cross_feat1) 
        feat2 = feat2 + self.drop_path(cross_feat2) 
        
        
        feat1 = self.resnet_layer3(feat1)
        feat2 = self.resnet_layer3(feat2)

        feat = torch.cat( (feat1,feat2),1 )

        feat = self.fusion_layer(feat)

        return feat,self.reshape_feat(feat1),self.reshape_feat(feat2)
    
    def norm_Quaternion(self,quat):
        normalized = quat.norm(dim=-1).unsqueeze(-1)

        eps = torch.ones_like(normalized) * .01
        pred_q = quat / torch.max(normalized, eps)
        return pred_q

    def forward(self,x1,x2):
        #孪生网络特征提取并融合
        feat,feat1,feat2 = self.siamese_extractor(x1,x2)
        feat = self.reshape_fused(feat).permute(0,-1,-2)
        feat = self.featureEncoder(feat)
        
        pred_q = self.fc_q(self.dropout(feat))
        
        pred_t = self.fc_t(self.dropout(feat))

        return  pred_q,pred_t,feat1,feat2

def SiTPose_1():

    return SiTPose(emb_dim = 384,rsl = 0.9 , num_head_attn = 6,feature_encoder_layer = 14)
