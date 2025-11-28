import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

traces = {}
posterior_predictive = {}

print("Computing posterior distributions for n...\n")

for Y in Y_values:
    for theta in theta_values:
        key = f"Y={Y}, theta={theta}"
        print(f"Running: {key}")
        
        with pm.Model() as model:
            n = pm.Poisson("n", mu=10)
            pm.Binomial("Y", n=n, p=theta, observed=Y)
            
            trace = pm.sample(
                draws=2000,
                tune=2000,
                chains=2,
                cores=1,
                random_seed=2025,
                progressbar=False
            )
            
            ppc = pm.sample_posterior_predictive(trace, random_seed=2025, progressbar=False)
            
        traces[key] = trace
        posterior_predictive[key] = ppc

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (key, trace) in enumerate(traces.items()):
    az.plot_posterior(trace, var_names=["n"], ax=axes[idx], hdi_prob=0.94)
    axes[idx].set_title(key)

plt.tight_layout()
plt.savefig("posterior_n.png")
print("\nPosterior plots saved to posterior_n.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (key, ppc) in enumerate(posterior_predictive.items()):
    Y_star = np.ravel(ppc.posterior_predictive["Y"].values)
    az.plot_dist(Y_star, ax=axes[idx])
    axes[idx].set_title(f"Posterior Predictive Y* | {key}")
    axes[idx].set_xlabel("Y*")

plt.tight_layout()
plt.savefig("posterior_predictive.png")
print("Posterior predictive plots saved to posterior_predictive.png")

print("\n" + "="*60)
print("POSTERIOR SUMMARIES FOR n")
print("="*60)
for key, trace in traces.items():
    print(f"\n{key}:")
    print(az.summary(trace, var_names=["n"], hdi_prob=0.94))
