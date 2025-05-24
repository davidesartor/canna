import numpy as np
import scipy
from torch.utils.data import DataLoader, Dataset
from einops import rearrange


class Sinusoids(Dataset):
    def __init__(
        self,
        batch_size: int,
        n_sources: int,
        n_times: int,
        size: int = 1024_000,
    ):
        super().__init__()
        self.size = size
        self.batch_size = batch_size

        self.n_sources = n_sources
        self.n_params = 3
        self.n_times = n_times
        self.n_channels = 2

        self.amplitude_range = (1, 10)
        self.frequency_range = (1, 10)
        self.times = np.linspace(0, 1, n_times)
        self.noise_std = 1.0

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, idx):
        params, datastream = self.sample_params_and_datastream()
        return params.astype(np.float32), datastream.astype(np.float32)

    def dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, batch_size=None, **kwargs)

    def sample_params_and_datastream(self) -> tuple[np.ndarray, np.ndarray]:
        # sample parameters
        shape = (self.batch_size, self.n_sources)
        a_min, a_max = self.amplitude_range
        f_min, f_max = self.frequency_range
        log_a = np.random.uniform(np.log(a_min), np.log(a_max), shape)
        log_f = np.random.uniform(np.log(f_min), np.log(f_max), shape)
        phi = np.random.uniform(0, 2 * np.pi, shape)
        params = np.stack([log_a, log_f, phi], axis=-1)

        # sample signal
        a = np.exp(log_a)
        f = np.exp(log_f)
        theta = 2 * np.pi * f[..., None] * self.times + phi[..., None]
        h_plus = (a[..., None] * np.sin(theta)).sum(-2)
        h_cross = (a[..., None] * np.cos(theta)).sum(-2)
        signal = np.stack([h_plus, h_cross], axis=-1)

        # sample noise
        noise = np.random.normal(0, self.noise_std, signal.shape)
        datastream = signal + noise
        return params, datastream

    def log_posterior(
        self,
        flat_params: np.ndarray,  # (..., n_sources * n_params)
        datastream: np.ndarray,  # (..., n_times, n_channels)
    ) -> np.ndarray:  # (...)
        # separate parameters
        params = rearrange(flat_params, "... (N P) -> ... N P", P=self.n_params)
        log_a = params[..., 0]
        log_f = params[..., 1]
        phi = params[..., 2]

        # log likelihood
        a = np.exp(log_a)
        f = np.exp(log_f)
        theta = 2 * np.pi * f[..., None] * self.times + phi[..., None]
        h_plus = (a[..., None] * np.sin(theta)).sum(-2)
        h_cross = (a[..., None] * np.cos(theta)).sum(-2)
        signal = np.stack([h_plus, h_cross], axis=-1)

        residual = datastream - signal
        log_likelihood = -0.5 * np.sum(residual**2) / self.noise_std**2

        # log prior
        mask_a = (self.amplitude_range[0] < a) * (a < self.amplitude_range[1])
        mask_f = (self.frequency_range[0] < f) * (f < self.frequency_range[1])
        mask_phi = (0 < phi) * (phi < 2 * np.pi)
        log_prior = (
            +np.where(mask_a, np.exp(-log_a.sum(-1)), -np.inf)
            + np.where(mask_f, np.exp(-log_f.sum(-1)), -np.inf)
            + np.where(mask_phi, np.exp(-phi.sum(-1)), -np.inf)
        ).sum(-1)  # sum over sources
        return log_likelihood + log_prior
