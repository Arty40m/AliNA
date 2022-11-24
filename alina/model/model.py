import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

torch.manual_seed(42)



def pos_enc(seq, dim):
      
    pos_enc = np.array([
    [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] 
        if pos != 0 else np.zeros(dim) 
        for pos in range(seq)
    ], dtype=np.float32)

    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1

    return pos_enc


class ConvLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b = torch.nn.parameter.Parameter(torch.zeros((dim,1,1)), requires_grad=True)
        self.a = torch.nn.parameter.Parameter(torch.ones((dim,1,1)), requires_grad=True)
        self.eps = 1e-6
        
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        dif = x - mean
        var = dif.pow(2).mean(1, keepdim=True)
        x = dif / torch.sqrt(var + self.eps)
        x = self.a*x + self.b
        
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, do):
        super().__init__()
        
        self.H = nn.Conv2d(in_ch, out_ch, kernel_size=(1,5), stride=1, padding=(2,0))
        self.W = nn.Conv2d(out_ch, out_ch, kernel_size=(5,1), stride=1, padding=(0,2))
        torch.nn.init.kaiming_uniform_(self.H.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.W.weight, nonlinearity='relu')
        
        self.l1 = nn.Conv2d(out_ch, out_ch*4, 
                            kernel_size=(1,1), stride=1, padding='valid')
        self.l2 = nn.Conv2d(out_ch*4, out_ch, 
                            kernel_size=(1,1), stride=1, padding='valid')
        
        torch.nn.init.kaiming_uniform_(self.l1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=1.0)
        self.drop = nn.Dropout(do)
        self.norm = ConvLayerNorm(out_ch)
        
        
    def forward(self, x):
        x = self.H(x)
        x = F.relu(x)

        x = self.W(x)
        a = F.relu(x)

        x = self.l1(a)
        x = F.relu(x)
        x = self.l2(x)
        
        x = self.drop(x)
        x = x+a
        x = self.norm(x)
        
        return x
        
        
class MHAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        
        self.heads = heads
        self.dim = dim
        self.depth = dim//heads
        self.norm = np.sqrt(self.depth)
        
        self.Q = torch.nn.parameter.Parameter(torch.empty(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.Q, gain=1.0)
        
        self.K = torch.nn.parameter.Parameter(torch.empty(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.K, gain=1.0)
        
        self.V = torch.nn.parameter.Parameter(torch.empty(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.V, gain=1.0)
        
        self.O = torch.nn.parameter.Parameter(torch.empty(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.O, gain=1.0)
        
        
    def forward(self, args):
        q, k, v, mask = args
        
        q = torch.matmul(q, self.Q)
        k = torch.matmul(k, self.K)
        v = torch.matmul(v, self.V)

        # batch, seq, heads, dim
        q = q.view(-1, q.shape[-2], self.heads, self.depth)
        k = k.view(-1, k.shape[-2], self.heads, self.depth)
        v = v.view(-1, v.shape[-2], self.heads, self.depth)

        # batch, heads, seq, dim
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        
        #att
        g = torch.matmul(q, k)
        g /= self.norm
        if mask is not None:
            g -= (mask*1e9)
        A = F.softmax(g, dim=-1)

        att = torch.matmul(A, v)# b,h,s,d

        att = att.permute(0, 2, 1, 3)# b,s,h,d
        att = torch.reshape(att, (att.shape[0], att.shape[-3], self.dim))
        att = torch.matmul(att, self.O)
        
        return att
    
    
class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, do):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.do = do

        self.Att = MHAttention(self.dim, self.heads) 

        self.drop1 = nn.Dropout(self.do)
        self.drop2 = nn.Dropout(self.do)

        self.LN1 = nn.LayerNorm(normalized_shape=dim)
        self.LN2 = nn.LayerNorm(normalized_shape=dim)

        self.FC1 = nn.Linear(dim, dim*4)
        self.FC2 = nn.Linear(dim*4, dim)
        torch.nn.init.kaiming_uniform_(self.FC1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.FC2.weight, gain=1.0)


    def forward(self, args):
        q, k, v, mask = args

        att = self.Att([q, k, v, mask])
        att = self.drop1(att)
        
        x = att + q
        x = self.LN1(x)

        d = F.relu(self.FC1(x))
        d = self.FC2(d)
        d = self.drop2(d)

        x = d + x
        x = self.LN2(x)

        return x


class Alina(nn.Module):
    def __init__(self, vocab, emb, dim, layers, heads, channels, convdrop):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab, emb)
        
        # ENCODER
        self.pool = nn.MaxPool2d(2, stride=2)
        self.EncCVBlock1 = ConvBlock(emb, channels[0], convdrop)
        self.EncCVBlock2 = ConvBlock(channels[0], channels[1], convdrop)
        self.EncCVBlock3 = ConvBlock(channels[1], channels[2], convdrop)
        self.EncCVBlock4 = ConvBlock(channels[2], channels[3], convdrop)
        
        # MIDDLE
        self.postpool = nn.Linear(channels[3], dim)
        torch.nn.init.xavier_uniform_(self.postpool.weight, gain=1.0)
        
        self.PE = torch.nn.parameter.Parameter(torch.from_numpy(pos_enc(seq=256, dim=dim)), 
                                                requires_grad=False)
        self.encoders_list = nn.ModuleList()
        for i in range(layers):
            self.encoders_list.append(EncoderLayer(dim=dim, heads=heads, do=0.1))

        # DECODER
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.DecCVBlock1 = ConvBlock((dim+channels[3]), channels[-1], convdrop)
        self.DecCVBlock2 = ConvBlock(channels[-2]+channels[-1], channels[-2], convdrop)
        self.DecCVBlock3 = ConvBlock(channels[-3]+channels[-2], channels[-3], convdrop)
        self.DecCVBlock4 = ConvBlock(channels[-4]+channels[-3], channels[-4], convdrop)
        
        # OUT    
        self.out = nn.Conv2d(channels[-4], 1, kernel_size=(1,1), stride=1, padding='valid')
        torch.nn.init.xavier_uniform_(self.out.weight, gain=1.0)
        
        
    def get_pad_mask(self, x):
        x = x.view(-1, 16, 16, 256)
        x = torch.permute(x, (0, 1, 3, 2)).contiguous() # b, Hpatch, patch_dim, W -> b, Hpatch, W, patch_dim
        x = x.view(-1, 16**2, 16**2) # b, Hpatch, W, patch_dim -> b, Hpatch*Wpatch, patch_dim*patch_dim
        
        x = torch.sum(x, dim=-1, keepdim=False) # b, Hpatch*Wpatch
        return (x==0.).float()
        
            
    def forward(self, seq):
        mask = self.get_pad_mask(seq) # b, seq
        att_mask = torch.unsqueeze(torch.unsqueeze(mask, 1), 1) # b, seq(1), heads(1), seq
        
        # EMB
        x = self.embedding(seq) # b, 256H, 256W, emb
        x = torch.permute(x, (0, 3, 1, 2)).contiguous() # b, dim, 256H, 256W
        
        # ENCODER
        x1 = self.EncCVBlock1(x) # b, 16, 256, 256 -> b, 32, 256, 256
        x = self.pool(x1) # b, 32, 256, 256 -> b, 32, 128, 128

        x2 = self.EncCVBlock2(x) # b, 32, 128, 128 -> b, 64, 128, 128
        x = self.pool(x2) # b, 64, 128, 128 -> b, 64, 64, 64

        x3 = self.EncCVBlock3(x) # b, 64, 64, 64 -> b, 128, 64, 64
        x = self.pool(x3) # b, 128, 64, 64 -> b, 128, 32, 32

        x4 = self.EncCVBlock4(x) # b, 128, 32, 32 -> b, 256, 32, 32
        x = self.pool(x4) # b, 256, 32, 32 -> b, 256, 16, 16
        
        # MIDDLE
        x = torch.permute(x, (0, 2, 3, 1)).contiguous() # b, dim, 16, 16 -> b, 16, 16, dim
        x = x.view(-1, 256, x.shape[-1]) # b, 16, 16, dim -> b, seq(256), dim
        x = self.postpool(x)
        x += self.PE
        
        for l in self.encoders_list:
            x = l([x, x, x, att_mask])
            
        x = x.view(-1, 16, 16, x.shape[-1])
        x = torch.permute(x, (0, 3, 1, 2)).contiguous() # b, H, W, dim -> b, dim, H, W
            
        # DECODER
        x = self.upsample(x) # b, 256, 16, 16 -> # b, 256, 32, 32
        x = torch.cat((x, x4), 1) # b, 256, 32, 32 + b, 256, 32, 32 -> b, 512, 32, 32
        x = self.DecCVBlock1(x) # b, 512, 32, 32 -> b, 256, 32, 32

        x = self.upsample(x) # b, 256, 32, 32 -> b, 256, 64, 64
        x = torch.cat((x, x3), 1) # b, 256, 64, 64 + b, 128, 64, 64 -> b, 384, 64, 64
        x = self.DecCVBlock2(x) # b, 384, 64, 64 -> b, 128, 64, 64

        x = self.upsample(x) # b, 128, 64, 64 -> b, 128, 128, 128
        x = torch.cat((x, x2), 1) # b, 128, 128, 128 + b, 64, 128, 128 -> b, 192, 128, 128
        x = self.DecCVBlock3(x) # b, 192, 128, 128 -> b, 64, 128, 128

        x = self.upsample(x) # b, 64, 128, 128 -> b, 64, 256, 256
        x = torch.cat((x, x1), 1) # b, 64, 256, 256 + b, 32, 256, 256 -> b, 96, 256, 256
        x = self.DecCVBlock4(x) # b, 96, 256, 256 -> b, 32, 256, 256
        
        # HEAD
        x = torch.squeeze(self.out(x)) # b, 32, 256, 256 -> b, 256, 256
        x = torch.sigmoid(x)
        
        return x
        