import numpy as np
import torch
from emcee import EnsembleSampler
from corner import corner
from matplotlib import pyplot as plt
import argparse

from datasets import Sinusoids
from networks import MMDiT

RUNS = 10
SAMPLES = 1024
DIFFUSIONSTEPS = 16
MCMCWALKERS = 32
MCMCDISCARD = 300

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script")
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device

    dataset = Sinusoids(n_sources=2, n_times=128, batch_size=1, size=RUNS)
    model = MMDiT.load_from_checkpoint(args.ckpt, map_location=device)

    for run in range(RUNS):
        true_parameters, datastream = dataset.sample_params_and_datastream()

        # generate samples using the model
        c = torch.from_numpy(datastream).to(device=device, dtype=torch.float32)
        c = torch.broadcast_to(c, (SAMPLES, *c.shape[1:]))
        x0 = torch.randn((SAMPLES, *true_parameters.shape[1:]), device=device)
        with torch.no_grad():
            x1 = model.push(x0, c, n_steps=16)
        generated_samples = x1.flatten(1).cpu().numpy()

        # generate samples using mcmc
        flat_parameters = true_parameters.flatten()
        p0 = flat_parameters + 1e-4 * np.random.randn(MCMCWALKERS, len(flat_parameters))
        sampler = EnsembleSampler(
            MCMCWALKERS, len(flat_parameters), dataset.log_posterior, args=(datastream,)
        )
        sampler.run_mcmc(p0, nsteps=SAMPLES // MCMCWALKERS + MCMCDISCARD, progress=True)
        mcmc_samples = sampler.get_chain(flat=True, discard=MCMCDISCARD)

        # plot the results
        fig = corner(mcmc_samples, labels=None, truths=flat_parameters, color="blue")
        fig = corner(generated_samples, color="red", fig=fig)
        plt.savefig(f"figures/{dataset.__class__.__name__}_run_{run}.pdf")
