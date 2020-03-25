# third party imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_performance_curves(model_performance: pd.DataFrame,
                            dim: tuple=(12, 8)):

    highest_auc = np.round(max(max(model_performance['train_performance']),
                               max(model_performance['selection_performance']),
                               max(model_performance['validation_performance'])
                               ), 1)

    fig, ax = plt.subplots(figsize=dim)

    plt.plot(model_performance['train_performance'], marker=".", markersize=20,
             linewidth=3, label='AUC train')
    plt.plot(model_performance['selection_performance'], marker=".",
             markersize=20, linewidth=3, label='AUC selection')
    plt.plot(model_performance['validation_performance'], marker=".",
             markersize=20, linewidth=3, label='AUC validation')
    # Set x/yticks
    ax.set_xticks(np.arange(len(model_performance['last_added_predictor'])
                            + 1))
    ax.set_xticklabels(model_performance['last_added_predictor'].tolist(),
                       rotation=40, ha='right')
    ax.set_yticks(np.arange(0.5, highest_auc + 0.02, 0.05))
    #Make Pretty
    ax.legend(loc='lower right')
    fig.suptitle('Performance curves - forward feature selection',
                 fontsize=20)
    plt.ylabel('Model performance')
    plt.show()
