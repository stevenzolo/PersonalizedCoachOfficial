"""
Plot figures for the paper.
"""
import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import gymnasium as gym
from matplotlib import pyplot as plt

import gym_games
from utils import lineplot_smoothly
from hyperparams.grid_world_arguments import coach_args, student_learn_args, student_play_args
from gym_games.wrappers import WindyWrapper
from coach_agent import InstantCoach, DelayedCoach, InstantBufferedCoach
from train import coach_train


def student_saturation_in_coach_training(coach_agent, plot=True, select_first_epis=np.inf):
    """
    When the student converges and always provide the same intention, the coach training becomes weak.
    To evaluate the student reset timing, we align the policy state variability to episode number.
    :return:
    """
    if coach_agent is InstantCoach:
        log_folder = os.path.join("logs", "windy_grid_world", "student_saturation_trend_instant_coaching")
    elif coach_agent is DelayedCoach:
        log_folder = os.path.join("logs", "windy_grid_world", "student_saturation_trend_delayed_coaching")
    else:
        raise NotImplementedError

    if len(os.listdir(log_folder)) > 0:
        print("Plot figure with existed log files")
    else:
        train_kwargs = deepcopy(coach_kwargs)
        train_kwargs['coach_agent'] = coach_agent
        train_kwargs['student_reset_after_epis'] = np.inf
        train_kwargs['log_path'] = os.path.join(log_folder, "reset_episodes_inf")
        coach_train(**train_kwargs)

    if not plot:  # generate data only
        return

    log_path = os.path.join(log_folder, "reset_episodes_inf")
    with open(log_path, 'rb') as lp:
        instr_coach_train_log = pickle.load(lp)
    coach_return_lst = []
    coach_state_nums_lst = []
    for repeat_trial_val in instr_coach_train_log["trial_results"]:
        epi_return_in_trial = np.array(repeat_trial_val['coach_return_in_epis'][:select_first_epis])
        coach_return_lst.append(
            pd.DataFrame({
                'Training Episodes': np.arange(len(epi_return_in_trial)) + 1,
                '-Steps per Episode': epi_return_in_trial
            })
        )
        epi_state_num_in_trial = np.array(repeat_trial_val['coach_state_nums'][:select_first_epis])
        coach_state_nums_lst.append(
            pd.DataFrame({
                'Training Episodes': np.arange(len(epi_state_num_in_trial)) + 1,
                'State-action Pairs': epi_state_num_in_trial
            })
        )

    fig, ax1 = plt.subplots(figsize=(10, 8))
    lineplot_smoothly(
        data=coach_return_lst, xaxis='Training Episodes', value='-Steps per Episode',
        smooth=student_learn_args.learn_epis, **{"label": '-Steps per Episode', "ax": ax1, "color": 'g'}
    )
    ax2 = ax1.twinx()
    lineplot_smoothly(
        data=coach_state_nums_lst, xaxis='Training Episodes', value='State-action Pairs',
        smooth=student_learn_args.learn_epis, **{"label": 'State-action Pairs', "ax": ax2, "color": 'b'}
    )
    plt.legend(loc='right').set_draggable(True)
    plt.show()
    return


def finetune_student_reset_in_coach_training(
        coach_agent, student_reset_lst, select_first_epis, train_smooth, eval_smooth, plot=True
):
    """
    See the training/evaluation performance under different 'student_reset' parameter
    :param select_first_epis:
    :param coach_agent:
    :param eval_smooth:
    :param train_smooth:
    :param student_reset_lst:
    :param plot:
    :return:
    """
    if coach_agent is InstantCoach:
        log_folder = os.path.join("logs", "windy_grid_world", "finetune_student_reset_instant_instr_coach")
    elif coach_agent is DelayedCoach:
        log_folder = os.path.join("logs", "windy_grid_world", "finetune_student_reset_delayed_instr_coach")
    elif coach_agent is InstantBufferedCoach:
        log_folder = os.path.join("logs", "windy_grid_world", "finetune_student_reset_instant_buffer_coach")
    else:
        raise NameError('Undefined coach agent')

    if len(os.listdir(log_folder)) > 0:
        print("Plot figure with existed log files")
    else:
        print("Finetune best student_reset parameter over: ", student_reset_lst)
        for student_reset in student_reset_lst:
            train_kwargs = deepcopy(coach_kwargs)
            train_kwargs['coach_agent'] = coach_agent
            train_kwargs["student_reset_after_epis"] = student_reset
            train_kwargs['log_path'] = os.path.join(log_folder, "reset_episodes_{}".format(student_reset))
            coach_train(**train_kwargs)

    if not plot:  # generate data only
        return

    training_return_in_trials = []
    eval_return_in_trials = []
    for reset_epis in student_reset_lst:
        log_path = os.path.join(log_folder, "reset_episodes_{}".format(reset_epis))
        with open(log_path, 'rb') as lp:
            instr_coach_train_log = pickle.load(lp)
        training_trial_in_reset, eval_trial_in_reset = [], []
        for repeat_trial_val in instr_coach_train_log["trial_results"]:
            training_trial_in_reset.append(np.average(
                repeat_trial_val['coach_return_in_epis'][:select_first_epis][-train_smooth:]))
            eval_trial_in_reset.append(-np.average(
                repeat_trial_val['eval_return_in_epis'][:int(select_first_epis/coach_args.eval_every_epis)]
                [-eval_smooth:]))
        training_return_in_trials.append(np.array(training_trial_in_reset))
        eval_return_in_trials.append(np.array(eval_trial_in_reset))

    # line smooth plot
    training_return_collection = [
        pd.DataFrame({
            'Student Reset Frequency': np.array(student_reset_lst),
            '-Steps per Episode': train_trial
        }) for train_trial in np.array(training_return_in_trials).T
    ]
    eval_return_collection = [
        pd.DataFrame({
            'Student Reset Frequency': np.array(student_reset_lst),
            '-Steps per Episode': eval_trial
        }) for eval_trial in np.array(eval_return_in_trials).T
    ]
    fig, ax1 = plt.subplots(figsize=(10, 8))
    lineplot_smoothly(
        data=training_return_collection, xaxis='Student Reset Frequency', value='-Steps per Episode',
        smooth=3, **{"label": 'Train', "color": 'g'}
    )
    lineplot_smoothly(
        data=eval_return_collection, xaxis='Student Reset Frequency', value='-Steps per Episode',
        smooth=3, **{"label": 'Evaluation', "color": 'b'}
    )
    plt.legend().set_draggable(True)
    plt.show()

    # # fill between plot
    # training_avg = np.average(training_return_in_trials, axis=1)
    # training_std = np.std(training_return_in_trials, axis=1) + np.fabs(np.random.normal(0, 1, len(training_avg)))
    # eval_avg = np.average(eval_return_in_trials, axis=1)
    # eval_std = np.std(eval_return_in_trials, axis=1) + np.fabs(np.random.normal(0, 1, len(eval_avg)))
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.plot(student_reset_lst, training_avg, label='training return')  # c='b',
    # plt.fill_between(student_reset_lst, training_avg-training_std, training_avg+training_std, alpha=0.5)
    # ax.plot(student_reset_lst, eval_avg, label='eval return')    # c='g',
    # plt.fill_between(student_reset_lst, eval_avg-eval_std, eval_avg+eval_std, alpha=0.5)
    # plt.xlabel('Student Reset Episodes')
    # plt.ylabel('Converged Return')
    # plt.legend()
    # plt.show()
    return


def dive_into_train_eval_detail(
        coach_agent, self_play_log_path, selected_student_reset_lst, train_smooth, eval_smooth
):
    if coach_agent is InstantCoach:
        log_folder = os.path.join("logs", "windy_grid_world", "finetune_student_reset_instant_instr_coach")
    elif coach_agent is DelayedCoach:
        log_folder = os.path.join("logs", "windy_grid_world", "finetune_student_reset_delayed_instr_coach")
    else:
        raise NotImplementedError

    if os.path.exists(self_play_log_path):
        with open(self_play_log_path, 'rb') as lp:
            self_play_log = pickle.load(lp)
        eval_over_trials = []
        sampled_eval_len = int(coach_args.eval_epis / student_play_args.eval_every_epis)
        for trial in self_play_log['trial_results']:
            eval_over_trials.append(np.average(trial['eval_score_lst'][:sampled_eval_len]))
        student_self_play_return = -np.average(eval_over_trials)
    else:
        print("Generate student self play log first")
        return

    color_lst = ['b', 'm', 'g', 'c']
    fig, ax1 = plt.subplots(figsize=(10, 8))
    for ix, reset_epis in enumerate(selected_student_reset_lst):
        log_path = os.path.join(log_folder, "reset_episodes_{}".format(reset_epis))
        with open(log_path, 'rb') as lp:  # try error
            instant_instr_coach_train_log = pickle.load(lp)
        coach_return_lst, eval_return_lst = [], []
        for repeat_trial_val in instant_instr_coach_train_log["trial_results"]:
            train_return_trial = np.array(repeat_trial_val['coach_return_in_epis'][:])  # select_first_epis=np.inf,
            coach_return_lst.append(
                pd.DataFrame({
                    'Training Episodes': np.arange(len(train_return_trial)) + 1,
                    '-Steps per Episode': train_return_trial
                })
            )
            eval_return_trial = -np.array(repeat_trial_val['eval_return_in_epis'])
            eval_return_lst.append(
                pd.DataFrame({
                    'Training Episodes': np.arange(
                        coach_args.eval_every_epis, coach_args.train_epis + coach_args.eval_every_epis,
                        step=coach_args.eval_every_epis),
                    '-Steps per Episode': eval_return_trial
                })

            )
        lineplot_smoothly(
            data=coach_return_lst, xaxis='Training Episodes', value='-Steps per Episode', smooth=train_smooth,
            **{"label": r'$\rho={}$, Train'.format(reset_epis), "color": color_lst[ix], "ax": ax1}  #
        )
        lineplot_smoothly(
            data=eval_return_lst, xaxis='Training Episodes', value='-Steps per Episode', smooth=eval_smooth,
            **{"label": r'$\rho={}$, Eval'.format(reset_epis), "color": color_lst[ix], "ls": '--', "ax": ax1}  #
        )
    plt.plot(
        [1, coach_args.train_epis], [student_self_play_return, student_self_play_return],
        '--r', lw=3, label='Self-study'
    )
    plt.legend().set_draggable(True)
    plt.show()
    return


def compare_train_speed(student_reset_lst, select_first_epis):
    instant_log_folder = "logs/windy_grid_world/finetune_student_reset_instant_instr_coach"
    delayed_log_folder = "logs/windy_grid_world/finetune_student_reset_delayed_instr_coach"

    def _load_avg_return(_reset, _folder):
        log_file = os.path.join(_folder, 'reset_episodes_{}'.format(_reset))
        with open(log_file, 'rb') as lf:
            train_log_dict = pickle.load(lf)
        avg_return_trials = [np.average(trial['coach_return_in_epis'][:select_first_epis])
                             for trial in train_log_dict["trial_results"]]
        return np.average(avg_return_trials)

    instant_avg_return_lst = [_load_avg_return(reset_epi, instant_log_folder)for reset_epi in student_reset_lst]
    delayed_avg_return_lst = [_load_avg_return(reset_epi, delayed_log_folder) for reset_epi in student_reset_lst]
    fig, ax1 = plt.subplots(figsize=(10, 8))
    plt.plot(np.array(student_reset_lst), instant_avg_return_lst, lw=4, label='Instant Teacher')
    plt.plot(np.array(student_reset_lst), delayed_avg_return_lst, lw=4, label='Delayed Teacher')
    plt.xlabel("Student Reset Frequency")
    plt.ylabel("-Steps per Episode")
    plt.legend().set_draggable(True)
    plt.show()
    return


def dive_coach_train_speed(instant_coach_log, instant_buffer_coach_log, delayed_coach_log, smooth):
    def _plot_coach_train(_coach_train_log, _label, _return_ax, _avg_q_ax, _color):
        with open(_coach_train_log, 'rb') as tl:
            train_log_dict = pickle.load(tl)
        return_lst, avg_q_lst = [], []
        for repeat_trial_val in train_log_dict["trial_results"]:
            xaxis = np.arange(len(repeat_trial_val['coach_return_in_epis'])) + 1
            return_lst.append(
                pd.DataFrame({
                    'Training Episodes': xaxis,
                    '-Steps per Episode': np.array(repeat_trial_val['coach_return_in_epis'])
                })
            )
            avg_q_lst.append(
                pd.DataFrame({
                    'Training Episodes': xaxis,
                    'Average Q Value': np.array(repeat_trial_val['average_q_in_epis'])
                })
            )
        lineplot_smoothly(
            data=return_lst, xaxis='Training Episodes', value='-Steps per Episode', smooth=smooth,
            **{"label": _label, 'ax': _return_ax, 'color': _color}
        )
        lineplot_smoothly(
            data=avg_q_lst, xaxis='Training Episodes', value='Average Q Value', smooth=smooth,
            **{'ax': _avg_q_ax, 'color': _color, 'ls': '--'}    # "label": _avg_q_label,
        )
        return

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()
    coach_log_lst = [
        instant_coach_log,
        # instant_buffer_coach_log,
        delayed_coach_log
    ]
    label_lst = [
        'instant coach',
        # 'instant buffer',
        'delayed coach (buffered)'
    ]
    color_lst = ['b', 'c']  # , 'g'
    for coach_train_log, label, color in zip(coach_log_lst, label_lst, color_lst):
        _plot_coach_train(coach_train_log, label, ax1, ax2, color)
    plt.legend().set_draggable(True)
    plt.show()
    return


if __name__ == '__main__':
    coach_kwargs = {
        'coach_game': WindyWrapper(gym.make("gym_games/WindyGridWorld-v0")),
        'repeat_train_trials': 3,   # 3 for debug, 10 for strong machine running
    }

    # student_saturation_in_coach_training(
    #     # InstantCoach: take <2k episodes for coach return -15, state numbers 30+ ~ 15.
    #     # DelayedCoach: take <3k episodes for coach return -15, state numbers 40+ ~ 15.
    #     coach_agent=DelayedCoach,
    #     plot=True,
    #     select_first_epis=int(5e3)     # Select same length of samples for different coach agents
    # )

    # finetune_student_reset_in_coach_training(
    #     # coach_agent=InstantCoach, select_first_epis=int(5e3),    # best 2, 1/4/8 is also ok
    #     coach_agent=DelayedCoach, select_first_epis=int(8e3),  # best 8
    #     # coach_agent=InstantBufferedCoach, select_first_epis=int(10e3),  # best 1
    #     student_reset_lst=np.concatenate((
    #         np.power(2, np.arange(9)),  # log space for fine-grained tuning
    #         np.linspace(400, 2000, num=9, dtype=int)    # linear space for sampling plot, until student saturation
    #     )),
    #     plot=True, train_smooth=student_learn_args.learn_epis, eval_smooth=10
    # )

    # dive_into_train_eval_detail(
    #     # coach_agent=InstantCoach, selected_student_reset_lst=[2, 2000],
    #     coach_agent=DelayedCoach, selected_student_reset_lst=[1, 8, 2000],
    #     self_play_log_path=student_play_args.log_path,
    #     train_smooth=student_learn_args.learn_epis,    # So, each point in train/eval is comparable
    #     eval_smooth=5   # experimental
    # )

    compare_train_speed(
        student_reset_lst=np.concatenate((
            np.power(2, np.arange(9)),  # log space for fine-grained tuning
            np.linspace(400, 2000, num=9, dtype=int)  # linear space for sampling plot, until student saturation
        )),
        select_first_epis=int(1e3)
    )   # for small reset, delayed training is faster; larger reset, instant performs better

    # dive_coach_train_speed(
    #     instant_coach_log="logs/finetune_student_reset_instant_instr_coach/reset_episodes_2",
    #     instant_buffer_coach_log="logs/finetune_student_reset_instant_buffer_coach/reset_episodes_8",
    #     delayed_coach_log="logs/finetune_student_reset_delayed_instr_coach/reset_episodes_8",
    #     smooth=student_learn_args.learn_epis
    # )

