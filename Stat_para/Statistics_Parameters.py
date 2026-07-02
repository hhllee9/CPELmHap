import Inference as inf
import numpy as np
import os

# --- 1. Parameter setup and path preparation ---
output_dir = 'Stat_para'
os.makedirs(output_dir, exist_ok=True)
data_path = os.path.join(output_dir, 'computed_metrics.npz')

n = np.array([4])
# Note: 2000x2000 grid means 4,000,000 iterations. This will take a significant amount of time.
alpha_range = np.linspace(-8, 8, 2000)
beta_range = np.linspace(-8, 8, 2000)
alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)

# --- 2. Data computation and saving ---
print("Starting computation (this process might take a while)...")

# Initialize data matrices
mml_values = np.zeros_like(alpha_grid)
nme_values = np.zeros_like(alpha_grid)
pdr_values = np.zeros_like(alpha_grid)
chalm_values = np.zeros_like(alpha_grid)
mcr_values = np.zeros_like(alpha_grid)

for i in range(len(alpha_range)):
    # Print progress every 100 iterations of alpha
    if i % 100 == 0:
        print(f"Progress: {i}/{len(alpha_range)} ({(i/len(alpha_range))*100:.1f}%)")
        
    for j in range(len(beta_range)):
        theta = np.array([alpha_range[i], beta_range[j]])
        grad_logZ = inf.get_grad_logZ(n, theta)
        
        mml_values[j, i] = inf.comp_mml(n, grad_logZ) 
        nme_values[j, i] = inf.comp_nme(n, theta, grad_logZ)
        pdr_values[j, i] = inf.comp_pdr(n, theta)
        chalm_values[j, i] = inf.comp_chalm(n, theta)
        mcr_values[j, i] = inf.comp_mcr(n, theta, grad_logZ)

# Save the computed results to disk
np.savez(data_path, 
         mml=mml_values, 
         nme=nme_values, 
         pdr=pdr_values, 
         chalm=chalm_values, 
         mcr=mcr_values,
         alpha=alpha_grid,
         beta=beta_grid)
         
print(f"Data computation completed and successfully saved to: {data_path}")