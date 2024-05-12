from gaussiansplatting.utils.sh_utils import RGB2SH, eval_sh
import torch

class SkyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sh_degree = 3
        # blue
        default_diffuse = RGB2SH(torch.Tensor([0,0,1])).float().cuda()
        shs = (
            torch.zeros(( 3, (self.sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        shs[:3,0] = default_diffuse
        self.shs = torch.nn.Parameter(shs)
    
    def forward(self,dirs):
        return eval_sh(self.sh_degree,self.shs,dirs)