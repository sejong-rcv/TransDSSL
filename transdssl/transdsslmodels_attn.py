import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    SoftAttDepth,
    Interpolate,
    _make_scratch,
)

from transdssl.swintf import SwinTransformer

def _make_fusion_block(features, use_norm, onlyATT=False):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        layer_norm=use_norm,
        expand=False,
        align_corners=True,
    )


class TransDSSL(BaseModel):
    def __init__(
        self,
        head,
        infer=False,
        features=256,
        backbone="S",
        channels_last=False,
        use_norm=False,
    ):
        non_negative = True
        super(TransDSSL, self).__init__()
        self.infer=infer
        self.channels_last = channels_last
        if backbone=="L":
            #############Swin-L#############################
            self.scratch = _make_scratch(
                [192, 384, 768, 1536], 256, groups=1, expand=False
            ) 
            ##Swin Large
            self.Swin=SwinTransformer(embed_dim=192, \
                                    depths=[2, 2, 18, 2], \
                                    num_heads=[6, 12, 24, 48], \
                                    window_size=7, \
                                    ape=False, \
                                    drop_path_rate=0.5, \
                                    patch_norm=True, \
                                    use_checkpoint=False)

            ch = torch.load("pretrained_model/swin_large_patch4_window7_224_22kto1k.pth")
        ##############Swin-B#############################
        elif backbone=="B":
            self.scratch = _make_scratch(
                [128, 256, 512, 1024], 256, groups=1, expand=False
            ) 

            ##Swin Base
            self.Swin=SwinTransformer(embed_dim=128, \
                                    depths=[2, 2, 18, 2], \
                                    num_heads=[4, 8, 16, 32], \
                                    window_size=7, \
                                    ape=False, \
                                    drop_path_rate=0.3, \
                                    patch_norm=True, \
                                    use_checkpoint=False)

            ch=torch.load("pretrained_model/swin_base_patch4_window7_224.pth")
        ###########Swin-S#######################
        elif backbone=="S":
            self.scratch = _make_scratch(
                [96, 192, 384, 768], features, groups=1, expand=False
            )
            ##Swin Small
            self.Swin=SwinTransformer(
                                    embed_dim=96,
                                    depths=[2, 2, 18, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=7,
                                    ape=False,
                                    drop_path_rate=0.3,
                                    patch_norm=True,
                                    use_checkpoint=False
                                )
            ch=torch.load("pretrained_model/swin_small_patch4_window7_224.pth")
        ##########Swin-T#######################
        elif backbone=="T":
            self.scratch = _make_scratch(
                [96, 192, 384, 768, 48], features, groups=1, expand=False
            ) 
            ##Swin Tiny
            self.Swin=SwinTransformer(
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=7,
                                    ape=False,
                                    drop_path_rate=0.2,
                                    patch_norm=True,
                                    use_checkpoint=False
                                )
            ch=torch.load("pretrained_model/swin_tiny_patch4_window7_224.pth")

        st=self.Swin.state_dict()
        st_keys=list(st.keys()) 

        if 'model' in ch.keys():
            keys=ch['model'].keys()

            ch_keys=[]
            for i in st_keys:
                if i in keys:
                    ch_keys.append(i) 
                else:
                    print(i)

            for i in range(len(ch_keys)):
                st[st_keys[i]]=ch["model"][ch_keys[i]]
        elif 'state_dict' in ch.keys():
            keys=ch["state_dict"].keys()

            ch_keys=[]
            for i in keys:
                if "backbone" in i:
                    ch_keys.append(i)
                else:
                    print(i)

            for i in range(len(ch_keys)):
                st[st_keys[i]]=ch["state_dict"][ch_keys[i]]

        else:
            print("Can't find Keys")
        
        
        print("################### Load State_dict [Swin_transformer] ############################## Swin-",backbone)

        self.Swin.load_state_dict(st)

        self.upsample = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        self.scratch.refinenet0 = _make_fusion_block(features, use_norm)
        self.scratch.refinenet1 = _make_fusion_block(features, use_norm)
        self.scratch.refinenet2 = _make_fusion_block(features, use_norm)
        self.scratch.refinenet3 = _make_fusion_block(features, use_norm)
        self.scratch.refinenet4 = _make_fusion_block(features, use_norm)
        self.attn_depth=SoftAttDepth()

        self.scratch.output_conv4 =nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )
        self.scratch.output_conv3 = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )

        self.scratch.output_conv = head

    def forward(self, x, epoch=0):


        _, layer_1, layer_2, layer_3, layer_4 = self.Swin(x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn)        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        if not self.infer:
            disp_4 = self.scratch.output_conv4(path_3)
            disp_4=self.attn_depth(disp_4)
        
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        if not self.infer:        
            disp_3 = self.scratch.output_conv3(path_2)
            disp_3=self.attn_depth(disp_3)
        
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)        
        
        disp_2 = self.scratch.output_conv2(path_1)
        disp_2 = self.attn_depth(disp_2)
        
        if not self.infer:
            layer_0_rn = self.upsample(layer_1_rn)
            path_0 = self.scratch.refinenet0(path_1, layer_0_rn)
            out = self.scratch.output_conv(path_0)

            out=self.attn_depth(out)
            return [out,disp_2,disp_3,disp_4]
        else:
            
            return [disp_2] 

class TRANSDSSLDepthModel(TransDSSL):
    def __init__(
        self, path=None,infer=False,   **kwargs    ):
        features = kwargs["features"] if "features" in kwargs else 256
        self.infer=infer
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )

        super().__init__(head,infer, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, rgb, epoch=0):
        x=rgb
        inv_depth = super().forward(x, epoch=epoch)        
        output={}
       
        output["inv_depths"]=inv_depth
        return output
