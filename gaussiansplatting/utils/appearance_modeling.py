import torch
import torch.nn as nn
import torch.nn.functional as F

class AppearanceCNN(nn.Module):
    def __init__(self, upsample_num ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        upscale_factor = 2
        def get_upsample_block(channels): 
            return nn.Sequential(
                *[
                    nn.PixelShuffle(upscale_factor=upscale_factor),
                    nn.Conv2d(in_channels=int(channels/2),out_channels=channels,kernel_size=3,padding=1,padding_mode="replicate"),
                    nn.ReLU()
                ]
            )
        self.first_conv = nn.Conv2d(67,256,3,padding=1,padding_mode="replicate")
        self.upsample_blocks = nn.ModuleList([get_upsample_block(int(256 / pow(2,i+1))) for i in range(upsample_num)])
        # self.bi_interp = F.interpolate
        self.final_conv = nn.Sequential(
            *[
                nn.Conv2d(16,3,3,padding=1,padding_mode="replicate"),
                nn.ReLU(),
                nn.Conv2d(3,3,3,padding=1,padding_mode="replicate")
            ]
        )

    def forward(self, image:torch.Tensor, embedding: torch.Tensor):
        # image: 3, height, weight
        # embedding: (64,)
        downsample_factor = 32
        _,orig_h,orig_w = image.shape
        downsampled_image = F.interpolate(image.unsqueeze(0),size=(int(orig_h/downsample_factor),int(orig_w/downsample_factor)),mode="bilinear") 

        _,_,H,W = downsampled_image.shape
        expanded_embedding = embedding[None,...,None,None].repeat(1,1,H,W)
        D = torch.concat([downsampled_image,expanded_embedding],dim=1)

        output = self.first_conv(D)
        for upsample_block in self.upsample_blocks:
            output = upsample_block(output)
        output = self.final_conv(F.interpolate(output,size=(orig_h,orig_w),mode="bilinear")).squeeze(0)
        return output