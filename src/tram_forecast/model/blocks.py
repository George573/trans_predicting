"""All convolution stages use independent multiscale paths."""
import torch
from torch import nn


class MultiscaleConv1d(nn.Module):
    def __init__(self, channels, paths):
        super().__init__()
        self.paths=nn.ModuleList(nn.Conv1d(channels,c,k,padding=d*(k-1)//2,dilation=d) for c,k,d in paths)

    def forward(self,x,mask=None):
        if mask is not None: x=x*mask
        out=torch.cat([torch.nn.functional.gelu(path(x),approximate='none') for path in self.paths],dim=1)
        return out if mask is None else out*mask
