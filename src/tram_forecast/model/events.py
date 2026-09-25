"""Event embedding and masked convolution reference implementation."""
import torch
from torch import nn
from .blocks import MultiscaleConv1d


class EventEncoder(nn.Module):
    def __init__(self,vocab_sizes,config):
        super().__init__()
        if len(vocab_sizes)!=5 or any(type(v) is not int or v<3 or v>cap+3 for v,cap in zip(vocab_sizes,config.category_caps)):
            raise ValueError('five vocabulary sizes, including three reserved IDs, must fit configured caps')
        self.embeddings=nn.ModuleList(nn.Embedding(v,d,padding_idx=0) for v,d in zip(vocab_sizes,config.embedding_dims))
        self.e1=MultiscaleConv1d(sum(config.embedding_dims),config.event1)
        self.e2=MultiscaleConv1d(32,config.event2)
        self.config=config

    def features(self,ids,lengths):
        """ids [H,N,5], lengths [H]; includes zero-length reference rows."""
        if ids.ndim!=3 or ids.shape[2]!=5 or ids.shape[1]<1 or lengths.shape!=(ids.shape[0],):
            raise ValueError('expected nonzero padded length and IDs [H,N,5], lengths [H]')
        if (lengths<0).any() or (lengths>ids.shape[1]).any():
            raise ValueError('invalid lengths')
        mask=(torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None])[:,None,:]
        x=torch.cat([embedding(ids[:,:,i]) for i,embedding in enumerate(self.embeddings)],dim=-1).transpose(1,2)
        return self.e2(self.e1(x,mask),mask),mask

    def forward(self,ids,lengths):
        features,mask=self.features(ids,lengths)
        # All-empty rows have a finite zero reduction; do not backprop through -inf.
        safe=mask | (lengths==0)[:,None,None]
        return features.masked_fill(~safe,float('-inf')).max(dim=-1).values
