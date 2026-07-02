import matplotlib.pyplot as plt
import numpy as np
import os

def run_plotting():
    output_dir = 'Stat_para'
    data_path = os.path.join(output_dir, 'computed_metrics.npz')

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Please run Stat_para/Statistics_Parameters.py first.")
        return

    print(f"Loading cached data: {data_path}...")
    cached_data = np.load(data_path)
    
    # Extract data
    mml_values = cached_data['mml']
    nme_values = cached_data['nme']
    pdr_values = cached_data['pdr']
    chalm_values = cached_data['chalm']
    mcr_values = cached_data['mcr']
    alpha_grid = cached_data['alpha']
    beta_grid = cached_data['beta']

    MY_CMAP = 'viridis' 

    def save_individual_plot(data, title, filename):
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(alpha_grid, beta_grid, data, shading='auto', cmap=MY_CMAP)
        plt.colorbar(label=f'{title} Value')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        plt.title(f' {title} Values for $\\theta=[\\alpha, \\beta]$')
        
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'{filename}.pdf'))
        plt.close()

    print("Generating individual plots...")
    save_individual_plot(mml_values, 'MML', 'MML')
    save_individual_plot(nme_values, 'NME', 'NME')
    save_individual_plot(pdr_values, 'PDR', 'PDR')
    save_individual_plot(chalm_values, 'CHALM', 'CHALM')
    save_individual_plot(mcr_values, 'MCR', 'MCR')

    print("Generating combined plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    plot_list = [
        (mml_values, 'MML'),
        (pdr_values, 'PDR'),
        (mcr_values, 'MCR'),
        (chalm_values, 'CHALM'),
        (nme_values, 'NME')
    ]

    for i, (data, title) in enumerate(plot_list):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        im = ax.pcolormesh(alpha_grid, beta_grid, data, shading='auto', cmap=MY_CMAP)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')

    axes[1, 2].axis('off')

    plt.suptitle('Combined Metrics Analysis', fontsize=20, y=1.02)
    plt.savefig(os.path.join(output_dir, 'Combined_Metrics_Analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Combined_Metrics_Analysis.pdf'), bbox_inches='tight')

    print(f"All plots have been updated and saved to the {output_dir} folder.")
    plt.show()

if __name__ == "__main__":
    run_plotting()