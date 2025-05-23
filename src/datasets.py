import numpy as np
from torch.utils.data import DataLoader, Dataset
from einops import rearrange


class Sinusoids(Dataset):
    def __init__(
        self, batch_size: int, n_sources: int, n_times: int, size: int = 1024_000
    ):
        super().__init__()
        self.size = size

        self.batch_size = batch_size
        self.n_sources = n_sources
        self.n_params = 3
        self.n_times = n_times
        self.n_channels = 2

        self.times = np.linspace(0, 1, n_times)
        self.noise_std = 1.0

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, idx):
        params = self.sample_params()
        datastream = self.sample_datastream(params)
        return params.astype(np.float32), datastream.astype(np.float32)

    def dataloader(self) -> DataLoader:
        return DataLoader(self, batch_size=None, num_workers=1, persistent_workers=True)

    def sample_params(self) -> np.ndarray:  # (batch_size, n_sources, n_params)
        log_amplitude = np.random.uniform(0, 1, (self.batch_size, self.n_sources))
        log_frequency = np.random.uniform(0, 1, (self.batch_size, self.n_sources))
        phase = np.random.uniform(0, 2 * np.pi, (self.batch_size, self.n_sources))
        params = np.stack([log_amplitude, log_frequency, phase], axis=-1)
        return params

    def sample_datastream(
        self, params: np.ndarray  # (..., n_sources, n_params)
    ) -> np.ndarray:  # (..., n_times, n_channels)
        signal = self.clean_signal(params)
        noise = np.random.normal(0, self.noise_std, signal.shape)
        return signal + noise

    def clean_signal(
        self, params: np.ndarray  # (..., n_sources, n_params)
    ) -> np.ndarray:  # (..., n_times, n_channels)
        amplitude = 10 ** params[..., 0, None]
        frequency = 10 ** params[..., 1, None]
        phase = params[..., 2, None]
        angles = 2 * np.pi * frequency * self.times + phase
        h_plus = (amplitude * np.sin(angles)).sum(-2)
        h_cross = (amplitude * np.cos(angles)).sum(-2)
        h = np.stack([h_plus, h_cross], axis=-1)
        return h

    def log_posterior(
        self,
        flat_params: np.ndarray,  # (..., n_sources * n_params)
        datastream: np.ndarray,  # (..., n_times, n_channels)
    ) -> np.ndarray:  # (...)
        params = rearrange(flat_params, "... -> ... N P", P=self.n_params)
        residual = datastream - self.clean_signal(params)
        log_likelihood = -0.5 * np.sum(residual**2) / self.noise_std**2
        log_prior = 0.0
        return log_likelihood + log_prior
