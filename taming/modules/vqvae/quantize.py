import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q



class VectorQuantizer2(nn.Module):
    """
    GPT:
        输入 latent z:         (B, 1600, 64, 64)
        ↓ rearrange + flatten
        z_flattened:           (B*4096, 1600)
        ↓ 与 codebook (8192, 1600) 计算距离
        ↓ argmin → 最相似 codebook index
        ↓ embedding lookup
        z_q:                   (B, 1600, 64, 64)
        🧠 Bonus: sane_index_shape=True 的作用
        如果你设置了：

        sane_index_shape=True
        那么返回的 min_encoding_indices 会 reshape 成：
        (B, 64, 64)
        这样你就能把每个像素位置的编码 index 可视化为一个“离散图”。

        🌟 一句话总结
        你提供的量化器 VectorQuantizer2 会将形状为 (B, 1600, 64, 64) 的张量，
        在每个位置查表替换为 codebook 中最接近的 1600 维向量，
        从而输出一个同样形状的量化结果，并带有可训练的 codebook。
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim) # !codebook.shape = (8192, 1600)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # 1、 reshape z -> (batch, height, width, channel) and flatten 
        # 第一步： 为了方便后续与 codebook 中的每个嵌入做距离计算
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        # 第二步：计算 L2 距离（欧氏距离）
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        # 第三步：选取最小距离的 code
        min_encoding_indices = torch.argmin(d, dim=1)
        
        
        # 第四步：量化，这就得到了量化后的张量，形状与输入相同，但内容是从 codebook 中查表而来的
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # 第五步：计算 VQ 损失，compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # 第六步：gradient trick（保留梯度），preserve gradients
        # 这一步是为了让 z_q 对输入 z 可导，但对 codebook 不可导
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)   


# class EMAVectorQuantizer(nn.Module):
#     def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5,
#                 remap=None, unknown_index="random"):
#         super().__init__()
#         self.codebook_dim = codebook_dim
#         self.num_tokens = num_tokens
#         self.beta = beta
#         self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

#         self.remap = remap
#         if self.remap is not None:
#             self.register_buffer("used", torch.tensor(np.load(self.remap)))
#             self.re_embed = self.used.shape[0]
#             self.unknown_index = unknown_index # "random" or "extra" or integer
#             if self.unknown_index == "extra":
#                 self.unknown_index = self.re_embed
#                 self.re_embed = self.re_embed+1
#             print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
#                   f"Using {self.unknown_index} for unknown indices.")
#         else:
#             self.re_embed = n_embed

#     def remap_to_used(self, inds):
#         ishape = inds.shape
#         assert len(ishape)>1
#         inds = inds.reshape(ishape[0],-1)
#         used = self.used.to(inds)
#         match = (inds[:,:,None]==used[None,None,...]).long()
#         new = match.argmax(-1)
#         unknown = match.sum(2)<1
#         if self.unknown_index == "random":
#             new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
#         else:
#             new[unknown] = self.unknown_index
#         return new.reshape(ishape)

#     def unmap_to_all(self, inds):
#         ishape = inds.shape
#         assert len(ishape)>1
#         inds = inds.reshape(ishape[0],-1)
#         used = self.used.to(inds)
#         if self.re_embed > self.used.shape[0]: # extra token
#             inds[inds>=self.used.shape[0]] = 0 # simply set to zero
#         back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
#         return back.reshape(ishape)

#     def forward(self, z):
#         # reshape z -> (batch, height, width, channel) and flatten
#         #z, 'b c h w -> b h w c'
#         z = rearrange(z, 'b c h w -> b h w c')
#         z_flattened = z.reshape(-1, self.codebook_dim)
        
#         # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
#         d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
#             self.embedding.weight.pow(2).sum(dim=1) - 2 * \
#             torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'


#         encoding_indices = torch.argmin(d, dim=1)

#         z_q = self.embedding(encoding_indices).view(z.shape)
#         encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

#         if self.training and self.embedding.update:
#             #EMA cluster size
#             encodings_sum = encodings.sum(0)            
#             self.embedding.cluster_size_ema_update(encodings_sum)
#             #EMA embedding average
#             embed_sum = encodings.transpose(0,1) @ z_flattened            
#             self.embedding.embed_avg_ema_update(embed_sum)
#             #normalize embed_avg and update weight
#             self.embedding.weight_update(self.num_tokens)

#         # compute loss for embedding
#         loss = self.beta * F.mse_loss(z_q.detach(), z) 

#         # preserve gradients
#         z_q = z + (z_q - z).detach()

#         # reshape back to match original input shape
#         #z_q, 'b h w c -> b c h w'
#         z_q = rearrange(z_q, 'b h w c -> b c h w')
#         return z_q, loss, (perplexity, encodings, encoding_indices)

