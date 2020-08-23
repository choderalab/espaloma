from matplotlib import pyplot as plt


def plot_residuals(predicted, target, mol_name, target_name):
    scatter_kwargs = dict(s=1, alpha=0.5)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(predicted, target, **scatter_kwargs)
    unit = '(kJ/mol / nm)'
    plt.xlabel(f'predicted {unit}')
    plt.ylabel(f'target {unit}')
    plt.title(f'{mol_name}: {target_name} force')

    plt.subplot(1, 2, 2)
    plt.scatter(predicted, predicted - target, **scatter_kwargs)
    plt.xlabel(f'predicted {unit}')
    plt.ylabel(f'predicted - target {unit}')
    plt.title(f'{mol_name}: {target_name} force residuals')
    plt.tight_layout()

    plt.savefig(f'plots/{mol_name}_{target_name}_residuals.png', bbox_inches='tight', dpi=300)
    plt.close()