from matplotlib import pyplot as plt
import numpy as onp


def plot_residuals(predicted, target, mol_name, target_name):
    scatter_kwargs = dict(s=1, alpha=0.5)
    refline_kwargs = dict(linestyle='--', color='grey')

    target_extent = [onp.min(target), onp.max(target)]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(target, predicted, **scatter_kwargs)
    plt.plot(target_extent, target_extent, **refline_kwargs)
    unit = '(kJ/mol / nm)'
    plt.xlabel(f'target {unit}')
    plt.ylabel(f'predicted {unit}')
    # TODO: and a horizontal line
    plt.title(f'{mol_name}: {target_name} force')

    plt.subplot(1, 2, 2)
    residuals = predicted - target
    plt.scatter(target, residuals, **scatter_kwargs)
    # don't let the y axis be smaller than [-1,+1]
    if (onp.max(residuals) - onp.min(residuals)) < 2.0:
        plt.ylim(-1,1)
    plt.hlines(0, *target_extent, **refline_kwargs)
    plt.xlabel(f'target {unit}')
    plt.ylabel(f'predicted - target {unit}')
    plt.title(f'{mol_name}: {target_name} force residuals')
    plt.tight_layout()

    plt.savefig(f'plots/{mol_name}_{target_name}_residuals.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_loss_traj(loss_traj, method, mol_name, target_name):
    running_min_loss_traj = onp.minimum.accumulate(loss_traj)
    plt.plot(running_min_loss_traj)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f'{method} iteration (within basin-hopping)')
    plt.ylabel(f'running minimum of RMSE loss\n(predicted MM {target_name} force vs. OFF1.0 {target_name} force, in kJ/mol / nm)')
    plt.title(f'{mol_name}: {target_name} force regression')
    plt.savefig(f'plots/{mol_name}_{target_name}_loss_traj.png', bbox_inches='tight', dpi=300)
    plt.close()