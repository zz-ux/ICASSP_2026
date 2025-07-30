import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from collections import OrderedDict
from einops import rearrange
from torch.nn import Dropout, Softmax, Linear, LayerNorm
import math


class Encoder(nn.Module):
    def __init__(self, c_in):
        super(Encoder, self).__init__()
        self.encoder_norm = LayerNorm(c_in, eps=1e-6)
        self.layer = Block(c_in)

    def forward(self, hidden_states):
        hidden_states = self.layer(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Attention(nn.Module):
    def __init__(self, c_in):
        super(Attention, self).__init__()
        self.num_attention_heads = 4
        self.attention_head_size = int(c_in / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(c_in, self.all_head_size)
        self.key = Linear(c_in, self.all_head_size)
        self.value = Linear(c_in, self.all_head_size)
        self.out = Linear(c_in, c_in)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        return attention_output


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Mlp(nn.Module):
    def __init__(self, c_in):
        super(Mlp, self).__init__()
        self.fc1 = Linear(c_in, c_in)
        self.fc2 = Linear(c_in, c_in)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, c_in):
        super(Block, self).__init__()
        self.hidden_size = c_in
        self.attention_norm = LayerNorm(c_in, eps=1e-6)
        self.ffn_norm = LayerNorm(c_in, eps=1e-6)
        self.ffn = Mlp(c_in)
        self.attn = Attention(c_in)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Transformer(nn.Module):
    def __init__(self, c_in):
        super(Transformer, self).__init__()
        self.encoder = Encoder(c_in)

    def forward(self, x):
        input_shape = x.shape
        input = rearrange(x, 'b c n h w -> (b h w) n c')
        out = self.encoder(input)
        out = rearrange(out, '(b h w) n c -> b c n h w', b=input_shape[0], h=input_shape[3], w=input_shape[4])
        return out


'---------------------------------------------------------------------------------------------------------------------'


class LightExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=9, other={}):
        super(LightExtractor, self).__init__()
        self.extractor_1 = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            Transformer(32),
            Transformer(32),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            Transformer(64),
        )
        self.extractor_2 = nn.Sequential(
            Transformer(64),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            Transformer(32),
        )

    def forward(self, x):
        feature_4 = self.extractor_1(x)
        feature_2 = self.extractor_2(feature_4)
        return feature_4, feature_2


class ImageExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=9, other={}):
        super(ImageExtractor, self).__init__()
        self.extractor_1 = nn.Sequential(
            nn.Conv3d(6, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.extractor_2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        feature_4 = self.extractor_1(x)
        feature_2 = self.extractor_2(feature_4)
        return feature_4, feature_2


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}, in_channel=32):
        super(Regressor, self).__init__()
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel * 2, in_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True), )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), )

    def forward(self, feat_4, feat_2):
        feat = self.conv_4(feat_4) + feat_2
        normal = F.normalize(self.conv(feat), p=2, dim=1)
        return normal


'---------------------------------------------------------------------------------------------------------------------'


class MyMethod(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=6, other={}):
        super(MyMethod, self).__init__()
        self.light_extractor = LightExtractor(batchNorm=batchNorm, c_in=c_in, other=other)
        self.image_extractor = ImageExtractor(batchNorm=batchNorm, c_in=c_in, other=other)
        self.regressor_aux_light = Regressor(batchNorm=batchNorm, other=other, in_channel=32)
        self.regressor_aux_image = Regressor(batchNorm=batchNorm, other=other, in_channel=32)
        self.regressor = Regressor(batchNorm=batchNorm, other=other, in_channel=64)

    def DataPreprocessing(self, x):
        with torch.no_grad():
            img, light = x
            Batchsize, NC, H, W = light.shape
            N = NC // 3
            # ImgReshape, LightReshape = torch.reshape(img, (Batchsize, N, 3, H, W)), torch.reshape(light,
            #                                                                                       (Batchsize, N, 3,
            #                                                                                        H, W))[:, :, :, 0, 0]
            # Img_gray = 0.2989 * ImgReshape[:, :, 0, :, :] + 0.5870 * ImgReshape[:, :, 1, :, :] + 0.1140 * ImgReshape[:,
            #                                                                                               :, 2, :, :]
            # '''利用L2的方法求解一下表面法向量L2Normals、表面反射率L2_Albedo'''
            # TransferConv = nn.Conv2d(in_channels=N, out_channels=3, kernel_size=1, stride=1, padding=0,
            #                          bias=False).to(torch.device(img.device))
            # L2Normals = []
            # for i in range(Batchsize):
            #     TransferConv.weight.data = torch.linalg.pinv(LightReshape[i]).unsqueeze(2).unsqueeze(3)
            #     L2Normals.append(TransferConv(Img_gray[i].unsqueeze(0)))
            # L2Normals = torch.cat(L2Normals, dim=0)
            # L2Normals = F.normalize(L2Normals, p=2, dim=1)
            # LightReshape = torch.reshape(light, (Batchsize, N, 3, H, W))
            # ReLU = nn.ReLU()
            # L2_Albedo = (torch.sum(ImgReshape, 1)) / (
            #         torch.sum(ReLU(torch.sum(L2Normals.unsqueeze(1) * LightReshape, dim=2)).unsqueeze(2),
            #                   dim=1) + 1e-6)
            # '''利用像素亮度最大值的方式，找到能让像素点最亮时的光照方向'''
            # Img_gray_reshape = Img_gray.unsqueeze(2)
            # _, Img_gray_Max_Indice = torch.max(Img_gray_reshape, dim=1, keepdim=True)
            # Img_Light_Max_Value = torch.gather(torch.reshape(light, (Batchsize, N, 3, H, W)), dim=1,
            #                                    index=Img_gray_Max_Indice.expand(-1, -1, 3, -1, -1))
            # Img_Light_Max_Value = Img_Light_Max_Value.squeeze(1)
            # Img_Max_Value, _ = torch.max(ImgReshape, dim=1, keepdim=True)
            # Img_Max_Value = Img_Max_Value.squeeze(1)
            # '''计算输入图像信息的平均亮度'''
            # Img_Mean_Value = torch.mean(ImgReshape, dim=1, keepdim=True).squeeze(1)
            '''获取每一组的输入信息'''
            img_split, light_split = torch.split(img, 3, 1), torch.split(light, 3, 1)  # 每一个都是列表，包含有N个[B, 3, H, W]数据
            NonShadow = []
            for i in range(len(img_split)):
                img_gray_i = 0.2989 * img_split[i][:, 0, :, :] + 0.5870 * img_split[i][:, 1, :, :] + 0.1140 * \
                             img_split[i][:, 2, :, :]
                img_gray_i = img_gray_i.unsqueeze(1)
                NonShadow.append(torch.tensor(img_gray_i > 0.05, dtype=torch.float32))  # 像素值太低，可能意味着有遮挡
            NonShadow = torch.sum(torch.stack(NonShadow, dim=1), dim=1) / N
            NonShadow = (NonShadow > 0.20)

            return NonShadow, img_split, light_split

    def forward(self, x):
        with torch.no_grad():
            NonShadow, img_split, light_split = self.DataPreprocessing(
                x)
            input_shadings = []
            for i in range(len(img_split)):
                shading_net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
                input_shadings.append(shading_net_in)
            input_shadings = torch.stack(input_shadings, 2)
            '----------------------------------------------------------------------------------------------------------'
        light_4, light_2 = self.light_extractor(input_shadings)
        image_4, image_2 = self.image_extractor(input_shadings)
        light_4_max, light_2_max = light_4.max(2)[0], light_2.max(2)[0]
        image_4_max, image_2_max = image_4.max(2)[0], image_2.max(2)[0]
        Aux_light = self.regressor_aux_light(light_4_max, light_2_max)
        Aux_image = self.regressor_aux_image(image_4_max, image_2_max)
        PredNormal = self.regressor(torch.cat([light_4_max, image_4_max], dim=1),
                                    torch.cat([light_2_max, image_2_max], dim=1))
        return NonShadow, Aux_light, Aux_image, PredNormal


if __name__ == '__main__':
    device = torch.device('cuda:0')
    B, N, C, H, W = 1, 10, 3, 512, 512
    rand_x = [torch.rand(size=(B, N * C, H, W)).to(device), torch.rand(size=(B, N * C, H, W)).to(device)]
    model = MyMethod(c_in=6).to(device)
    model.load_state_dict(
        torch.load('F:/py_code/pythonProject/PhotometricStereometry/ICASSP_2026/ICASSP_2026.pth.tar')['state_dict'])

    with torch.no_grad():
        NonShadow, Normal_1, Normal_2, Normal_3 = model(rand_x)
    # print(Normal_1.shape, torch.max(Normal_1), torch.min(Normal_1), torch.linalg.norm(Normal_1[0, :, 0, 0]))
    # print(Normal_2.shape, torch.max(Normal_2), torch.min(Normal_2), torch.linalg.norm(Normal_2[0, :, 0, 0]))
    # print(Normal_3.shape, torch.max(Normal_3), torch.min(Normal_3), torch.linalg.norm(Normal_3[0, :, 0, 0]))
