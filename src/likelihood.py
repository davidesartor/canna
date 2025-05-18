import test
from jaxtyping import Float, Array, Key
import jax.random as jr
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

N=2
T=128
rng = jr.key(10)
times=np.linspace(0, 1, T)
parsfid,sig,_=test.sample_params_and_signal(rng, N, T)
parsfid=np.array(parsfid).flatten()

def h_model(params, t):
    A = params[0::3]
    freqs = params[1::3]
    phi = params[2::3]

    phase = 2*np.pi*freqs[:,None]*t[None] + phi[:, None]  # shape: (N, T)

    sin_terms = (A[:, None] * np.sin(phase))  # shape: (N, T)
    cos_terms = (A[:, None] * np.cos(phase))  # shape: (N, T)
    
    return np.array([sin_terms.sum(axis=0), cos_terms.sum(axis=0)]).T # shape: (T, 2)

sigfid=h_model(parsfid, times)
'''
plt.plot(times, sigfid[:, 0], label="h_plus")
plt.plot(times, sigfid[:, 1], label="h_cross")
plt.plot(times, sig[:, 0], label="h_plus_data")
plt.plot(times, sig[:, 1], label="h_cross_data")
plt.show()
'''
def log_likelihood(params, t, data,sigma=1.0):
    model = h_model(params, t) # shape: (T, 2)
    residual = data - model
    return -0.5 * np.sum(residual ** 2)/ sigma ** 2

''' 
parstest=np.stack([parsfid]*100, axis=0)
parstest[:,0]=(0.1*np.linspace(-1, 1, 100)+1)*parsfid[0]

likels=np.array([log_likelihood(parstest[i], times, sig) for i in range(100)])
print(parsfid[0])
plt.plot(parstest[:,0], likels)
plt.show()
'''




import emcee
# Set up MCMC sampling with emcee

def log_prob(params, t, data, sigma=1.0):
    return log_likelihood(params, t, data, sigma)

ndim = len(parsfid)
nwalkers = 32
p0 = parsfid + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_prob, args=(times, sigfid)
)

print("Running MCMC...")
sampler.run_mcmc(p0, 1000, progress=True)
print("Done.")

# Example: get the samples
samples = sampler.get_chain(flat=True)
samples_reshaped=samples.reshape(samples.shape[0], -1, 3)
'''
# Plot the results
import corner
fig = corner.corner(samples, labels=[f"param {i}" for i in range(ndim)], truths=parsfid)
plt.show()

'''

np.savez('cose', generated_samples=samples_reshaped,
         datastream=sig, times=times, true_params=parsfid,noise_std=1.0)