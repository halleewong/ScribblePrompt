from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------

class Conv2d(nn.Module):
    """ Perform a 2D convolution

    inputs are [b, c, h, w] where 
        b is the batch size
        c is the number of channels 
        h is the height
        w is the width
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 padding: int,
                 do_activation: bool = True, 
                 ):
        super(Conv2d, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]

        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x):
        # x is [B, C, H, W]
        return self.conv(x)
    
# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

class _UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: List[int] = [64, 64, 64, 64, 64],
                 conv_kernel_size: int = 3,
                 conv: Optional[nn.Module] = None,
                 conv_kwargs: Dict[str,Any] = {}
                 ):
        """
        UNet (but can switch out the Conv)
        """
        super(_UNet, self).__init__()

        self.in_channels = in_channels

        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feat in features:
            self.downs.append(
                conv(
                    in_channels, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                )
            )
            in_channels = feat

        # Up part of U-Net
        for feat in reversed(features):
            self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.ups.append(
                conv(
                    # Factor of 2 is for the skip connections
                    feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
                )
            )

        self.bottleneck = conv(
            features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, **conv_kwargs
            )
        self.final_conv = conv(
            features[0], out_channels, kernel_size=1, padding=0, do_activation=False, **conv_kwargs
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
    

class UNet(_UNet):
    """
    Unet with normal conv blocks

    input shape: B x C x H x W
    output shape: B x C x H x W 
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(conv=Conv2d, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
        