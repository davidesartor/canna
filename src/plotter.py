from matplotlib import pyplot as plt
import numpy as np

def h_model(params, t):
    A = params[0::3]
    freqs = params[1::3]
    phi = params[2::3]

    phase = 2*np.pi*freqs[:,None]*t[None] + phi[:, None]  # shape: (N, T)

    sin_terms = (A[:, None] * np.sin(phase))  # shape: (N, T)
    cos_terms = (A[:, None] * np.cos(phase))  # shape: (N, T)
    
    return np.array([sin_terms.sum(axis=0), cos_terms.sum(axis=0)]).T # shape: (T, 2)

def log_likelihood(params, t, data,sigma=1.0):
    model = h_model(params, t) # shape: (T, 2)
    residual = data - model
    return -0.5 * np.sum(residual ** 2)/ sigma ** 2

def log_prob(params, t, data, sigma):
    return log_likelihood(params, t, data, sigma=sigma)

def plotter_from_npz(name):
    cose_load=np.load(name+'.npz')
    pars_plot=cose_load['generated_samples']
    datastream_plot=cose_load['datastream']
    times_plot=cose_load['times']
    parsfid_plot=cose_load['true_params']
    sigma_plot=cose_load['noise_std']

    import emcee
    ndim = len(parsfid_plot)
    nwalkers = 32
    p0 = parsfid_plot + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=(times_plot, datastream_plot,sigma_plot)
    )

    print("Running MCMC...")
    sampler.run_mcmc(p0, 1000, progress=True)
    print("Done.")

    # Example: get the samples
    samples = sampler.get_chain(flat=True)

    # Plot the results
    import corner
    fig = corner.corner(samples, labels=[f"param {i}" for i in range(ndim)],
                        truths=parsfid_plot,color='red')

    pars_plot = pars_plot.reshape(pars_plot.shape[0], -1)
    corner.corner(pars_plot, labels=[f"param {i}" for i in range(ndim)],
                  truths=parsfid_plot,fig=fig,color='blue')
    plt.show()

plotter_from_npz('cose')