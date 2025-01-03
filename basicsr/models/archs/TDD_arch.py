import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2



class AdaIn(nn.Module):
    def __init__(self):
        super(AdaIn, self).__init__()
        self.eps = 1e-5

    def forward(self, x, mean_style, std_style):
        b, c, h, w = x.shape
        feature = x.view(b, c, -1)
        # print (mean_feat.shape, std_feat.shape, mean_style.shape, std_style.shape)
        std_style = std_style.view(b, c, 1)
        mean_style = mean_style.view(b, c, 1)
        adain = std_style * (feature) + mean_style
        adain = adain.view(b, c, h, w)
        return adain

class CustomAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size=(1, 1)):
        super(CustomAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        # 根据输入的形状动态调整池化尺寸
        return F.adaptive_avg_pool2d(x, self.output_size)


class NCS_Block(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.in_channels = c
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # R gamma and beta
        self.conv_gamma_r = nn.Conv2d(c // 2, c, kernel_size=3, padding=1)
        self.conv_beta_r = nn.Conv2d(c // 2, c, kernel_size=3, padding=1)
        self.conv_gamma_t = nn.Conv2d(c // 2, c, kernel_size=3, padding=1)
        self.conv_beta_t = nn.Conv2d(c // 2, c, kernel_size=3, padding=1)
        self.conv_att_A = nn.Conv2d(c, 1, kernel_size=3, padding=1)
        self.conv_att_B = nn.Conv2d(c, 1, kernel_size=3, padding=1)
        self.global_feat = CustomAdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_1x1_style = nn.Conv2d(c, c // 2, kernel_size=3, padding=1)
        self.conv_1x1_style_rt = nn.Conv2d(c, c // 2, kernel_size=3, padding=1)
        self.style = nn.Linear(c // 2, c * 2)
        self.adaIn = AdaIn()

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x1 = self.dropout1(x)
        style_feat_1 = F.relu(self.conv_1x1_style(x1))  # 1,8,1,256*256
        style_feat_2 = F.relu(self.conv_1x1_style_rt(x1))  # 1,8,1,256*256
        style_feat = self.global_feat(style_feat_1)
        style_feat = torch.flatten(style_feat, start_dim=1)  # 1,8
        style_feat = self.style(style_feat)  # 1,32  #Fs
        # self-guieded learn mean，std，gamma，and beta
        # mean, std
        style_mean = style_feat[:, :self.in_channels]  # mean and std is shape 1*1*2 each channel    #1,16,256,256
        style_std = style_feat[:, self.in_channels:]  # 1,16,256,256
        gamma_r = self.conv_gamma_r(style_feat_1)  #
        beta_r = self.conv_beta_r(style_feat_1)  #
        gamma_t = self.conv_gamma_t(style_feat_2)
        beta_t = self.conv_beta_t(style_feat_2)
        out_new_gamma_T = x1 * (1 + gamma_t) + beta_t  # T
        out_new_style = self.adaIn(inp, style_mean, style_std) #自
        out_att_B = torch.sigmoid(self.conv_att_B(x1))
        step1 = (1 - out_att_B) * out_new_gamma_T + out_att_B * out_new_style

        y = step1 + inp
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        out_att_A = torch.sigmoid(self.conv_att_A(x))
        out_new_gamma_R = x * (1 + gamma_r) + beta_r  # R
        step2 = out_att_A * out_new_gamma_R + (1 - out_att_A) * step1
        out = step2 + y
        return out


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CFFB(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(CFFB, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = CustomAdaptiveAvgPool2d(output_size=(1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class TDDCR_Multiscale(nn.Module):

    def __init__(self, img_channel=4, width=16, middle_blk_num=2, blk_nums=[2, 3],
                 ):
        super().__init__()

        chan = width
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.Fusion1= CFFB(chan)
        self.Fusion2= CFFB(chan)
        self.Fusion3 = CFFB(chan)
        self.encoders_small = nn.ModuleList()
        self.downs_small = nn.ModuleList()
        self.latents_small = nn.ModuleList()
        self.ups_small = nn.ModuleList()
        self.decoders_small = nn.ModuleList()

        self.encoders_mid = nn.ModuleList()
        self.downs_mid = nn.ModuleList()
        self.latents_mid = nn.ModuleList()
        self.ups_mid = nn.ModuleList()
        self.decoders_mid = nn.ModuleList()

        self.encoders_max = nn.ModuleList()
        self.downs_max = nn.ModuleList()
        self.latents_mid = nn.ModuleList()
        self.ups_max = nn.ModuleList()
        self.decoders_max = nn.ModuleList()

        for num in blk_nums:
            self.encoders_small.append(
                nn.Sequential(
                    *[NCS_Block(chan) for _ in range(num)]
                )
            )

            self.encoders_mid.append(
                nn.Sequential(
                    *[NCS_Block(chan) for _ in range(num)]
                )
            )

            self.encoders_max.append(
                nn.Sequential(
                    *[NCS_Block(chan) for _ in range(num)]
                )
            )


            self.downs_small.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )

            self.downs_mid.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)

            )

            self.downs_max.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
          )
            chan = chan * 2

        self.middle_blks_small = \
            nn.Sequential(
                *[NCS_Block(chan) for _ in range(middle_blk_num)]
            )
        self.middle_blks_mid = \
            nn.Sequential(
                *[NCS_Block(chan) for _ in range(middle_blk_num)]
            )

        self.middle_blks_max = \
            nn.Sequential(
                *[NCS_Block(chan) for _ in range(middle_blk_num)]
            )

        for num in reversed(blk_nums):
            self.ups_small.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)  # 像素重排层

                )
            )
            self.ups_mid.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)  # 像素重排层   #nn.PixelUnShuffle(2)缩小二百
                )
            )
            self.ups_max.append(
                nn.Sequential(
                    #ResidualUpSample(chan)
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)  # 像素重排层
                )
            )
            chan = chan // 2

            self.decoders_small.append(
                nn.Sequential(
                    *[NCS_Block(chan) for _ in range(num)]
                )
            )

            self.decoders_mid.append(
                nn.Sequential(
                    *[NCS_Block(chan) for _ in range(num)]
                )
            )

            self.decoders_max.append(
                nn.Sequential(
                    *[NCS_Block(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders_small)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        inp_img_max = inp
        inp_img_mid = F.interpolate(inp, scale_factor=0.5)
        inp_img_small = F.interpolate(inp, scale_factor=0.25)

        inp_img_max = self.intro(inp_img_max)  #to_dim
        inp_img_mid = self.intro(inp_img_mid)  #to_dim
        inp_img_small = self.intro(inp_img_small)#to_dim

        encs_small = []
        encs_mid = []
        encs_max = []

        x1 = inp_img_small
        for encoder, down in zip(self.encoders_small, self.downs_small):
            x1 = encoder(x1)
            encs_small.append(x1)
            x1 = down(x1)
        x1 = self.middle_blks_small(x1)
        for decoder, up, enc_skip, in zip(self.decoders_small, self.ups_small, encs_small[::-1]):
            x1 = up(x1)
            x1 = x1 + enc_skip
            x1 = decoder(x1)
        out_small= self.Fusion1([x1, inp_img_small])
        out_small = F.interpolate(out_small, scale_factor=2)  #1,64,64,64
        inp_mid = out_small+ inp_img_mid

        x2 = inp_mid
        for encoder, down in zip(self.encoders_mid, self.downs_mid):
            x2 = encoder(x2)
            encs_mid.append(x2)
            x2 = down(x2)
        x2 = self.middle_blks_mid(x2)  #256,16,16
        for decoder, up, enc_skip, in zip(self.decoders_mid, self.ups_mid, encs_mid[::-1]):
            x2 = up(x2)
            x2 = x2 + enc_skip
            x2 = decoder(x2)

        out_mid = self.Fusion2([x2, inp_img_mid])
        out_mid = F.interpolate(out_mid, scale_factor=2) #64,128,128
        inp_max = out_mid + inp_img_max

        x3 =inp_max
        for encoder, down in zip(self.encoders_max, self.downs_max):
            x3 = encoder(x3)
            encs_max.append(x3)
            x3 = down(x3)
        x3 = self.middle_blks_max(x3)

        for decoder3, up3, enc_skip3, in zip(self.decoders_max, self.ups_max, encs_max[::-1]):
            x3 = up3(x3)
            x3 = x3 + enc_skip3
            x3 = decoder3(x3)

        out_max = self.Fusion3([x3, inp_img_max])
        out  = self.ending(out_max)
        out = out + inp
        return out[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class TDDLocal(Local_Base, TDDCR_Multiscale):
    def __init__(self, *args, train_size=(1, 4, 128, 128), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        TDDCR_Multiscale.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)




