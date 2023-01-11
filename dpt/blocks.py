import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape0 = out_shape
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape

    if expand == True:
        # out_shape0 = out_shape
        # out_shape1 = out_shape * 2
        # out_shape2 = out_shape * 4
        # out_shape3 = out_shape * 8
        # out_shape4 = out_shape * 16

        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    # scratch.layer0_rn = nn.Conv2d(
    #     in_shape[0],
    #     out_shape1,
    #     kernel_size=1,
    #     stride=1,
    #     padding=0,
    #     bias=False,
    #     groups=groups,
    # )

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        groups=groups,
    )

    return scratch

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class SoftAttn(nn.Module):
    def __init__(self, alpha=0.01, beta=1.0, dim=1, discretization='UD', convatten='False',ch=256, onlyATT=False):
        super(SoftAttn, self).__init__()
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.discretization = discretization
        self.convatten = convatten
        self.onlyATT = onlyATT

        self.en_atten = nn.Sequential(
                                        nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1, padding=0),
                                        # nn.BatchNorm2d(ch),
                                        # nn.LeakyReLU(True)
                                        )
        self.en_feature = nn.Sequential(
                                    nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1, padding=0),
                                    # nn.BatchNorm2d(ch),
                                    # nn.LeakyReLU(True)
                                    )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, output_t, input_t, eps=1e-6, isADAT=False):
        
        info_map = self.en_feature(input_t)
        z = self.skip_add.add(info_map, output_t)
        atten_map = F.softmax(self.en_atten(input_t), dim=self.dim)
        z = z * atten_map

        return z

class SoftAttDepth(nn.Module):
    def __init__(self, alpha=0.01, beta=1.0, dim=1, discretization='UD'):
        super(SoftAttDepth, self).__init__()
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.discretization = discretization

    def get_depth_sid(self, depth_labels):
        alpha_ = torch.FloatTensor([self.alpha])
        beta_ = torch.FloatTensor([self.beta])
        t = []
        for K in range(depth_labels):
            K_ = torch.FloatTensor([K])
            t.append(torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * K_ / depth_labels))
        t = torch.FloatTensor(t)
        return t

    def forward(self, input_t, eps=1e-6):
        batch_size, depth, height, width = input_t.shape
        if self.discretization == 'SID':
            grid = self.get_depth_sid(depth).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            grid = torch.linspace(
                self.alpha, self.beta, depth,
                requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        grid = grid.repeat(batch_size, 1, height, width).float()

        z = F.softmax(input_t, dim=self.dim)
        z = z * (grid.to(z.device))
        z = torch.sum(z, dim=1, keepdim=True)

        return z

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, layer_norm, isFlow=False):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.layer_norm = layer_norm
        self.isFlow = isFlow
        self.groups = 1
        

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.layer_norm,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.layer_norm,
            groups=self.groups,
        )

        if self.layer_norm == True:
            self.layer_norm1 = nn.BatchNorm2d(features)
            self.layer_norm2 = nn.BatchNorm2d(features)

        print(f'self.conv1.weight : {self.conv1.weight.sum()}, self.conv2.weight : {self.conv2.weight.sum()}')
        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.layer_norm == True:
            out = self.layer_norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.layer_norm == True:
            out = self.layer_norm2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        layer_norm=False,
        expand=False,
        align_corners=True,
        scale=1
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.scale = scale
        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            if features==256:
                out_features=features
            else:
                out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )
        self.dim = 1
        print(f'self.out_conv.weight : {self.out_conv.weight.sum()}')
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, layer_norm)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, layer_norm)
        self.en_atten = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, stride=1, padding=0)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        df = xs[0]#11763

        if len(xs) == 2:      
            if self.scale==1:
                res = self.skip_add.add(df, xs[1]) #16557
            else:
                import pdb;pdb.set_trace()
                res = df
            #resatten verison
            # ef = nn.functional.interpolate(
            #      self.resConfUnit1(xs[1]), scale_factor=self.scale, mode="bilinear", align_corners=self.align_corners
            #      )

            att = F.softmax(self.en_atten(self.resConfUnit1(xs[1])), dim=self.dim)
            out = res*att #84
            output = self.skip_add.add(self.resConfUnit2(out), res)

            # output = self.resConfUnit2(res)
        else:
            output = self.resConfUnit2(df)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output