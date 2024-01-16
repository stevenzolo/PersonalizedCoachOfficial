import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')
sns.set(style="whitegrid", font_scale=2)   # 1.5, darkgrid


def lineplot_smoothly(data, xaxis, value, smooth=1, **kwargs):
    # refer to: https://github.com/openai/spinningup/blob/master/spinup/utils/plot.py
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.lineplot(data=data, x=xaxis, y=value, lw=3, **kwargs)  # hue=condition, errorbar='sd', time-consuming
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs) ->
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 2e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # plt.tight_layout(pad=0.5)


def worker_wrapper(args):
    func, kwargs = args
    return func(**kwargs)


def eval_one_config(log_path, plot_kwargs, smooth):
    # plot one configration in coach evaluation
    with open(log_path, 'rb') as lp:
        student_learn_log = pickle.load(lp)
    return_over_trials = []
    for trial in student_learn_log['trial_results']:
        return_over_trials.append(
            pd.DataFrame({
                'Evaluation Episodes': np.linspace(
                    int(len(trial['return_in_epis'])/len(trial['eval_score_lst'])),   # ['epis_in_session']
                    len(trial['return_in_epis']),   # training episodes
                    len(trial['eval_score_lst'])
                ),
                '-Steps per Episode': -np.array(trial['eval_score_lst'])
            })
        )
    lineplot_smoothly(
        data=return_over_trials, xaxis='Evaluation Episodes', value='-Steps per Episode',
        smooth=smooth, **plot_kwargs
    )
    return




