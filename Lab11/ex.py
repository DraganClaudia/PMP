import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
df = pd.read_csv('Prices.csv')

y = df['Price'].values
x1 = df['Speed'].values
x2 = np.log(df['HardDrive'].values)
a)
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=1000)
    beta1 = pm.Normal('beta1', mu=0, sigma=1000)
    beta2 = pm.Normal('beta2', mu=0, sigma=1000)
    sigma = pm.HalfNormal('sigma', sigma=1000)

    mu = alpha + beta1 * x1 + beta2 * x2

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True, 
                      random_seed=42, 
                      compile_kwargs={'mode': 'FAST_COMPILE'})
b)
hdi_beta1 = az.hdi(trace, hdi_prob=0.95)['beta1'].values
hdi_beta2 = az.hdi(trace, hdi_prob=0.95)['beta2'].values

print("b) HDI 95%:")
print(f"beta1: [{hdi_beta1[0]:.4f}, {hdi_beta1[1]:.4f}]")
print(f"beta2: [{hdi_beta2[0]:.4f}, {hdi_beta2[1]:.4f}]")
c)
print("\nc) Utilitate predictori:")
print(f"beta1 (Speed): Da, HDI nu contine 0" if hdi_beta1[0] > 0 or hdi_beta1[1] < 0 else "beta1: Nu, HDI contine 0")
print(f"beta2 (log HardDrive): Da, HDI nu contine 0" if hdi_beta2[0] > 0 or hdi_beta2[1] < 0 else "beta2: Nu, HDI contine 0")
d)
speed_new = 33
hd_new = np.log(540)

posterior = trace.posterior
alpha_samples = posterior['alpha'].values.flatten()
beta1_samples = posterior['beta1'].values.flatten()
beta2_samples = posterior['beta2'].values.flatten()

mu_new = alpha_samples + beta1_samples * speed_new + beta2_samples * hd_new
hdi_mu = az.hdi(mu_new, hdi_prob=0.90)

print(f"\nd) HDI 90% pentru E(Price) la Speed=33, HardDrive=540:")
print(f"[{hdi_mu[0]:.2f}, {hdi_mu[1]:.2f}]")
e)
sigma_samples = posterior['sigma'].values.flatten()
y_pred = np.random.normal(mu_new, sigma_samples)
hdi_pred = az.hdi(y_pred, hdi_prob=0.90)

print(f"\ne) HDI 90% predictie pentru Price la Speed=33, HardDrive=540:")
print(f"[{hdi_pred[0]:.2f}, {hdi_pred[1]:.2f}]")
Bonus
premium = (df['Premium'] == 'Yes').astype(int).values

with pm.Model() as model_bonus:
    alpha = pm.Normal('alpha', mu=0, sigma=1000)
    beta1 = pm.Normal('beta1', mu=0, sigma=1000)
    beta2 = pm.Normal('beta2', mu=0, sigma=1000)
    beta3 = pm.Normal('beta3', mu=0, sigma=1000)
    sigma = pm.HalfNormal('sigma', sigma=1000)
    
    mu = alpha + beta1 * x1 + beta2 * x2 + beta3 * premium
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    
    trace_bonus = pm.sample(2000, tune=1000, return_inferencedata=True, 
                        random_seed=42, 
                        compile_kwargs={'mode': 'FAST_COMPILE'})
hdi_beta3 = az.hdi(trace_bonus, hdi_prob=0.95)['beta3'].values
print(f"\nBonus: HDI 95% pentru beta3 (Premium): [{hdi_beta3[0]:.2f}, {hdi_beta3[1]:.2f}]")
print(f"Premium afecteaza pretul: {'Da' if hdi_beta3[0] > 0 or hdi_beta3[1] < 0 else 'Nu'}")
