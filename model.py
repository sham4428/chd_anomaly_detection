import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        if batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model] if batch_first else [seq_len, batch_size, d_model]
        """
        if self.batch_first:
            return x + self.pe[:, :x.size(1), :]
        else:
            return x + self.pe[:x.size(0), :]


class ECGAutoencoder(nn.Module):
    """基于Transformer的ECG自编码器"""
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=1000)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码器（简单线性层）
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim(12), seq_len(1000)]
        """
        batch_size = x.shape[0]
        
        # 转置为 [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        z = self.transformer_encoder(x)
        
        # 解码重建
        recon = self.decoder(z)
        
        # 转置回原始格式 [batch_size, input_dim, seq_len]
        return recon.transpose(1, 2)


class PCGAutoencoder(nn.Module):
    """基于Transformer的PCG自编码器"""
    def __init__(self, input_dim: int = 13, hidden_dim: int = 32, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=200)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码器
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim(13), seq_len(200)]
        """
        batch_size = x.shape[0]
        
        # 转置为 [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        z = self.transformer_encoder(x)
        
        # 解码重建
        recon = self.decoder(z)
        
        # 转置回原始格式 [batch_size, input_dim, seq_len]
        return recon.transpose(1, 2)


class PatchEmbedding(nn.Module):
    """图像Patch嵌入模块"""
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 将patch展平并投影
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, in_channels, img_size, img_size]
        output: [batch_size, n_patches, embed_dim]
        """
        x = self.proj(x)  # [batch_size, embed_dim, n_patches^0.5, n_patches^0.5]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x


class CXRAutoencoder(nn.Module):
    """基于Vision Transformer的CXR自编码器"""
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 1,
                 embed_dim: int = 128, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=self.n_patches)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码器：通过反卷积重建图像
        self.decoder = nn.Sequential(
            # 先将embed_dim映射回去
            nn.Linear(embed_dim, patch_size * patch_size),
            # 重组成图像
            nn.Unflatten(2, (1, patch_size, patch_size)),
            # 逐步上采样
            nn.ConvTranspose2d(1, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, in_channels(1), img_size(256), img_size(256)]
        """
        batch_size = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        z = self.transformer_encoder(x)
        
        # 解码重建
        # [batch_size, n_patches, embed_dim] -> [batch_size, n_patches, patch_size^2]
        recon = self.decoder[0](z)
        
        # [batch_size, n_patches, patch_size^2] -> [batch_size, n_patches, 1, patch_size, patch_size]
        recon = recon.reshape(batch_size, -1, 1, self.patch_size, self.patch_size)
        
        # [batch_size, n_patches, 1, patch_size, patch_size] -> [batch_size, 1, img_size, img_size]
        recon = recon.permute(0, 2, 1, 3, 4)
        recon = recon.reshape(batch_size, 1, 
                              self.img_size // self.patch_size, self.patch_size,
                              self.img_size // self.patch_size, self.patch_size)
        recon = recon.permute(0, 1, 2, 4, 3, 5)
        recon = recon.reshape(batch_size, 1, self.img_size, self.img_size)
        
        return recon
