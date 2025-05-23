import torch
from torch import nn, Tensor
from torch.nn.functional import scaled_dot_product_attention, mse_loss
import numpy as np
from einops import rearrange
from lightning import LightningModule
from tqdm import tqdm


def modulate(x: Tensor, scale: Tensor, shift: Tensor):
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.std(dim=-1, keepdim=True)
    x = x * (1 + scale) + shift
    return x


class Modulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, y: Tensor):
        modulation = self.linear(self.act(y)).unsqueeze(-2)
        return torch.split(modulation, y.shape[-1], dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv_proj_x = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_proj_c = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj_x = nn.Linear(dim, dim, bias=False)
        self.out_proj_c = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, c: Tensor):
        h = torch.concat([self.qkv_proj_x(x), self.qkv_proj_c(c)], dim=-2)
        qkv = rearrange(h, "B N (H D) -> B H N D", H=self.num_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        h = scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "B H N D -> B N (H D)")
        x, c = torch.split(h, [x.shape[-2], c.shape[-2]], dim=-2)
        x = self.out_proj_x(x)
        c = self.out_proj_c(c)
        return x, c


class MMDiTBlock(nn.Module):
    """
    implementation inspired by SD 3.5
    https://openreview.net/forum?id=FPnUhsQJ5B
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.modulation_x = Modulation(dim)
        self.modulation_c = Modulation(dim)
        self.attention = CrossAttention(dim, num_heads)
        self.mlp_x = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: Tensor, c: Tensor, y: Tensor):
        # get modulation parameters
        alpha_x, beta_x, gamma_x, delta_x, epsilon_x, zeta_x = self.modulation_x(y)
        alpha_c, beta_c, gamma_c, delta_c, epsilon_c, zeta_c = self.modulation_c(y)

        # cross attention block
        hx = modulate(x, alpha_x, beta_x)
        hc = modulate(c, alpha_c, beta_c)
        hx, hc = self.attention(hx, hc)
        x = x + hx * gamma_x
        c = c + hc * gamma_c

        # feed forward blocks
        hx = modulate(x, delta_x, epsilon_x)
        hc = modulate(c, delta_c, epsilon_c)
        hx = self.mlp_x(modulate(x, delta_x, epsilon_x))
        hc = self.mlp_c(modulate(c, delta_c, epsilon_c))
        x = x + hx * zeta_x
        c = c + hc * zeta_c
        return x, c


class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int, period: float = 2 * np.pi):
        super().__init__()
        self.dim = dim
        self.period = period
        self.embed = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def __call__(self, t: Tensor):
        freqs = torch.exp(
            -np.log(self.period) * torch.linspace(0, 1, self.dim, device=t.device)
        )
        angles = 2 * np.pi * freqs * t[..., None]
        x = torch.concat([torch.sin(angles), torch.cos(angles)], dim=-1)
        x = self.embed(x)
        return x


class MMDiT(LightningModule):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.x_pos_embed = SinusoidalEmbed(hidden_dim)
        self.x_embed = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.c_pos_embed = SinusoidalEmbed(hidden_dim)
        self.c_embed = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.y_embed = nn.Sequential(
            SinusoidalEmbed(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [MMDiTBlock(hidden_dim, num_heads) for _ in range(num_blocks)]
        )

        self.out_modulation = Modulation(hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, x_dim)

    def forward(self, x: Tensor, t: Tensor, c: Tensor):
        # embeddings
        x_pos = torch.linspace(0, 1, x.shape[-2], device=x.device)
        x = self.x_embed(x) + self.x_pos_embed(x_pos)
        c_pos = torch.linspace(0, 1, c.shape[-2], device=c.device)
        c = self.c_embed(c) + self.c_pos_embed(c_pos)
        y = self.y_embed(t)

        # cross attention blocks
        for block in self.blocks:
            x, c = block(x, c, y)

        # final projection
        alpha, beta, _, _, _, _ = self.out_modulation(y)
        x = modulate(x, alpha, beta)
        x = self.out_projection(x)
        return x

    def push(self, x: Tensor, c: Tensor, n_steps: int = 16):
        dt = 1.0 / n_steps
        t = torch.zeros_like(x[..., 0, 0])

        # integration with runge-kutta
        for _ in tqdm(range(n_steps), desc="pushing", leave=False):
            k1 = self(x, t, c)
            k2 = self(x + k1 * dt / 2, t + dt / 2, c)
            k3 = self(x + k2 * dt / 2, t + dt / 2, c)
            k4 = self(x + k3 * dt, t + dt, c)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            t = t + dt
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        (x1, c) = batch
        t = torch.sigmoid(torch.randn_like(x1[..., 0, 0]))
        x0 = torch.randn_like(x1)
        xt = x1 * t[..., None, None] + x0 * (1 - t[..., None, None])

        loss = mse_loss(self(xt, t, c), x1 - x0)
        self.log("train/loss", loss, prog_bar=True)
        return loss