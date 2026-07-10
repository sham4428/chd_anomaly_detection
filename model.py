import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
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

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model] if batch_first else [seq_len, batch_size, d_model]
        """
        if self.batch_first:
            return x + self.pe[:, :x.size(1), :]
        return x + self.pe[:x.size(0), :]


class ECGAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, bottleneck_dim: int = 16,
                 num_layers: int = 2, num_heads: int = 4, use_bottleneck: bool = True):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.bottleneck = (
            nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.GELU(),
            )
            if use_bottleneck
            else nn.Identity()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim(12), seq_len(1000)]
        """
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        z = self.transformer_encoder(x)
        z = self.bottleneck(z)
        recon = self.decoder(z)
        return recon.transpose(1, 2)


class PCGAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 32, bottleneck_dim: int = 8,
                 num_layers: int = 2, num_heads: int = 4, use_bottleneck: bool = True):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=200)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.bottleneck = (
            nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.GELU(),
            )
            if use_bottleneck
            else nn.Identity()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim(13), seq_len(200)]
        """
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        z = self.transformer_encoder(x)
        z = self.bottleneck(z)
        recon = self.decoder(z)
        return recon.transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, in_channels, img_size, img_size]
        output: [batch_size, n_patches, embed_dim]
        """
        x = self.proj(x)
        x = x.flatten(2)
        return x.transpose(1, 2)


class CXRAutoencoder(nn.Module):
    """Vision Transformer encoder with a CNN decoder for CXR reconstruction."""

    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 1,
                 embed_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 decoder_mode: str = "cnn"):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2
        self.decoder_mode = decoder_mode

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=self.n_patches)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if decoder_mode == "cnn":
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        elif decoder_mode == "legacy_patch":
            if in_channels != 1:
                raise ValueError("The legacy CXR checkpoint supports one input channel only.")
            # This layout matches the historical checkpoint exactly. Only the
            # first layer was used by its original patch reconstruction path.
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, patch_size * patch_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(1, 32, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unknown CXR decoder mode: {decoder_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, in_channels(1), img_size(256), img_size(256)]
        """
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        x = self.pos_encoder(x)
        z = self.transformer_encoder(x)

        if self.decoder_mode == "legacy_patch":
            patches = self.decoder[0](z)
            patches = patches.reshape(
                batch_size,
                self.grid_size,
                self.grid_size,
                self.patch_size,
                self.patch_size,
            )
            return patches.permute(0, 1, 3, 2, 4).reshape(
                batch_size, 1, self.img_size, self.img_size
            )

        z = z.transpose(1, 2).reshape(batch_size, -1, self.grid_size, self.grid_size)
        return self.decoder(z)


def _load_checkpoint_state(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a state dictionary.")

    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    return checkpoint


def load_autoencoder(model_cls, checkpoint_path, device):
    """Load current and historical checkpoints without mixing architectures."""
    state_dict = _load_checkpoint_state(checkpoint_path, device)

    if model_cls is ECGAutoencoder:
        has_bottleneck = any(key.startswith("bottleneck.") for key in state_dict)
        model = ECGAutoencoder(use_bottleneck=has_bottleneck)
        architecture = "ecg_bottleneck_v2" if has_bottleneck else "ecg_legacy_no_bottleneck"
    elif model_cls is PCGAutoencoder:
        has_bottleneck = any(key.startswith("bottleneck.") for key in state_dict)
        model = PCGAutoencoder(use_bottleneck=has_bottleneck)
        architecture = "pcg_bottleneck_v2" if has_bottleneck else "pcg_legacy_no_bottleneck"
    elif model_cls is CXRAutoencoder:
        first_decoder_weight = state_dict.get("decoder.0.weight")
        is_legacy = first_decoder_weight is not None and first_decoder_weight.ndim == 2
        decoder_mode = "legacy_patch" if is_legacy else "cnn"
        model = CXRAutoencoder(decoder_mode=decoder_mode)
        architecture = "cxr_legacy_patch_v1" if is_legacy else "cxr_cnn_decoder_v2"
    else:
        raise TypeError(f"Unsupported model class: {model_cls!r}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with inferred architecture {architecture}."
        ) from exc

    model.checkpoint_architecture = architecture
    model.to(device)
    model.eval()
    return model
