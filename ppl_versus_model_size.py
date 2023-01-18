import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
import pickle

sns.set_theme(style='whitegrid')

plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.25

from matplotlib import font_manager
font_path='/Users/ericwallace/Library/Fonts/cmunss.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path, weight='bold')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

plt.rcParams['legend.title_fontsize'] = 10


def plot_line(unprompted_models: Dict,
                          res: pd.DataFrame,
                          model: str,
                          ax: plt.axis):
        means = res[model].apply(lambda x: x['mean'])
        # stds = res[model].apply(lambda x: x['std'])
        x_axis = list(res[list(res.keys())[0]].keys())
        color = unprompted_models[model]['color']
        sns.lineplot(x=x_axis,
                                  y=means,
                                  label=model,
                                  color=color,
                                  ax=ax,
                                  linewidth=3,
                                  markers=True,
                                  marker='o',
                                  markersize=8)
        return ax

def add_line(x: int, y: int):
        plt.plot([x, x], [0, y], linestyle='dashed', color='black')
        plt.plot([0, x], [y, y], linestyle='dashed', color='black')
        plt.scatter(x=x, y=y, color='black', zorder=3)

def reorder_legend(unprompted_models: Dict, ax: plt.axis):
        label_order = [(x, unprompted_models[x]['order']) for x in unprompted_models.keys()]
        label_order.sort(key = lambda x: x[1])
        label_order = [x[0] for x in label_order]
        handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels

        dict_ = dict(zip(labels, handles))
        ordered = [(model, dict_[model]) for model in label_order]
        labels = [x[0] for x in ordered]
        handles = [x[1] for x in ordered]
        leg = ax.legend(handles, labels, loc='upper right', prop=prop)

def plot(unprompted_models: Dict, results: pd.DataFrame, output_file: str, ylabel, ylim=None, yaxis_ticks=None, title=None, xlabel='LM Parameter Count', hline_value=None, xaxis_ticks=None, xaxis_labels=None):
        fig, ax = plt.subplots(1,1, figsize=(4,4))
        if hline_value is not None:
            ax.axhline(y=hline_value, linestyle='--', color='black')

        results = pd.DataFrame(results)
        for model in unprompted_models.keys():
            ax = plot_line(unprompted_models, results, model, ax)

        ax.set_xscale('log')

        if ylim is not None:
            plt.ylim(ylim)
        if yaxis_ticks is not None:
            ax.set_yticks(yaxis_ticks)
            ax.set_yticklabels([str(y) for y in yaxis_ticks], fontsize=11)

        if xaxis_labels:
            ax.set_xticks(xaxis_ticks)
            ax.set_xticklabels([str(y) for y in xaxis_labels], fontsize=10, rotation=30)

        ax.set_xlabel(xlabel, fontsize=11)
        plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=11)

        leg = plt.legend(loc='upper right', title='Method', prop=prop)
        for text in leg.get_texts():
                plt.setp(text, fontsize=8)
        plt.title(title)
        plt.tight_layout()

        ## SAVE
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
          
methods = {
    "GPTQ": {
            "color": '#63a2d6',
            "order": 1
    },
    "Ours": {
            "color": '#ed5e61',
            "order": 1
    },
   "FP16": {
            "color": '#84ca81',
            "order": 1
    },
}

# Wikitext results
results = {
        'Ours': {
                0.125: {'mean': 34.5},
                0.350: {'mean': 20.7},
                2.7: {'mean': 14.31509113},
                6.7: {'mean': 10.47884274},
                13: {'mean': 9.076656342},
                30: {'mean': 8.379032135},
                175: {'mean': 8.56},
        },
        'GPTQ': {             
                0.125: {'mean': 54.2050209},
                0.350: {'mean': 34.3410759},
                2.7: {'mean': 16.95541763},
                6.7: {'mean': 15.00282001},
                13: {'mean': 11.57760239},
                30: {'mean': 10.33251476},
                175: {'mean': 8.7202939},
        },        
        'FP16': {             
                0.125: {'mean': 27.65548897},
                0.350: {'mean': 22.00401306},
                2.7: {'mean': 12.47104645},
                6.7: {'mean': 10.85984898},
                13: {'mean': 10.12800121},
                30: {'mean': 9.558148384},
                175: {'mean': 8.34},
        }
}

ylim = [7.0, 56.0]
yaxis_ticks = [10, 20, 30, 40, 50]
xaxis_labels = ['125M','350M','2.7B','6.7B','13B','30B','175B']
xaxis_ticks = [0.125, 0.35, 2.7, 6.7, 13, 30, 175]
ylabel = "WikiText Perplexity"
title = None
plot(methods, results, 'plots/model_size_wikitext.pdf', ylabel, ylim=ylim, yaxis_ticks=yaxis_ticks, title=title, xaxis_ticks=xaxis_ticks, xaxis_labels=xaxis_labels)

# C4 results
results = {
        'Ours': {             
                0.125: {'mean': 38.2895813},
                0.350: {'mean': 25.6443634},
                2.7: {'mean': 19.98594856},
                6.7: {'mean': 15.55011654},
                13: {'mean': 12.55710697},
                30: {'mean': 11.58844376},
                175: {'mean': 9.406469345},
        },
        'GPTQ': {             
                0.125: {'mean': 73.5815506},
                0.350: {'mean': 39.70460129},
                2.7: {'mean': 20.13529205},
                6.7: {'mean': 18.26984215},
                13: {'mean': 13.8610096},
                30: {'mean': 12.76215935},
                175: {'mean': 9.98164463},
        },        
        'FP16': {
                0.125: {'mean': 24.60574722},
                0.350: {'mean': 20.71434784},
                2.7: {'mean': 13.16470623},
                6.7: {'mean': 11.74290466},
                13: {'mean': 11.199646},
                30: {'mean': 10.69407654},
                175: {'mean': 9.56},
        },
}

ylim = [7.0, 76.0]
yaxis_ticks = [10, 20, 30, 40, 50, 60, 70]
xaxis_labels = ['125M','350M','2.7B','6.7B','13B','30B','175B']
xaxis_ticks = [0.125, 0.35, 2.7, 6.7, 13, 30, 175]
ylabel = "C4 Perplexity"
title = None
plot(methods, results, 'plots/model_size_c4.pdf', ylabel, ylim=ylim, yaxis_ticks=yaxis_ticks, title=title, xaxis_ticks=xaxis_ticks, xaxis_labels=xaxis_labels)


# M2D2 results
# TODO


# BLOOM model for C4
# TODO