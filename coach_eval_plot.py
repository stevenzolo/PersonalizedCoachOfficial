"""
Plot figures related to coach evaluation in the paper.
"""
import os
import gymnasium as gym
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

import gym_games
from utils import eval_one_config
from hyperparams.grid_world_arguments import student_play_args
from gym_games.wrappers import WindyWrapper
from student_agent import InstantStudent, DelayedStudent
from train import student_learn, student_self_play


def finetune_budget_session_in_coach_eval(
        budget_session_lst, instant_coach_model_path, delayed_coach_model_path, smooth, plot=True
):
    log_folder = os.path.join("logs", "grid_world_gym", "finetune_budget_session_in_coach_eval")
    student_lst = [InstantStudent, DelayedStudent]
    coach_model_lst = [instant_coach_model_path, delayed_coach_model_path]
    mark_lst = ['instant', 'delayed']

    if len(os.listdir(log_folder)) > 0:
        print("Plot figure with existed log files")
    else:
        print("Finetune budget and session parameter over: ", budget_session_lst)
        learn_kwargs = deepcopy(student_kwargs)
        for student_agent, coach_model_path, mark in zip(student_lst, coach_model_lst, mark_lst):
            learn_kwargs['student_agent'] = student_agent
            learn_kwargs['coach_model_path'] = coach_model_path
            for (advice_budget, epis_in_session, advice_sessions) in budget_session_lst:
                learn_kwargs['advice_budget'] = advice_budget
                learn_kwargs['epis_in_session'] = epis_in_session
                learn_kwargs['advice_sessions'] = advice_sessions
                learn_kwargs['log_path'] = os.path.join(log_folder, '{}_budget{}_session{}_coach{}'.format(
                    mark, advice_budget, epis_in_session, advice_sessions))
                student_learn(**learn_kwargs)
            # break   # only run 'instant' in debug

    if not os.path.exists(student_play_args.log_path):
        print('Please run self play trials first!')
        return

    if not plot:
        return

    compared_configs = {
        'instant': budget_session_lst,    # selected groups
        'delayed': budget_session_lst
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for student_mark in compared_configs:
        for (advice_budget, epis_in_session, advice_sessions) in compared_configs[student_mark]:
            log_path = os.path.join(log_folder, '{}_budget{}_session{}_coach{}'.format(
                student_mark, advice_budget, epis_in_session, advice_sessions))
            eval_one_config(log_path, {"label": '{}_budget{}_session{}_coach{}'.format(
                student_mark, advice_budget, epis_in_session, advice_sessions)}, smooth)
    eval_one_config(student_play_args.log_path, {"label": 'self-play', "color": 'r', "ls": '--'}, smooth)
    plt.legend().set_draggable(True)
    plt.show()
    return


def standard_vs_personalized_instr(
        student_agent, personal_coach_path, elite_student_path, smooth=10, plot=True
):
    log_parent_folder = os.path.join("logs", "windy_grid_world", "standard_vs_personalized_instr")
    if student_agent is InstantStudent:
        log_folder = os.path.join(log_parent_folder, "coach_instantly")
    elif student_agent is DelayedStudent:
        log_folder = os.path.join(log_parent_folder, "coach_in_delay")
    else:
        raise NotImplementedError

    if not os.path.exists(student_play_args.log_path):
        print("Generate student self-play log first")
        return

    if len(os.listdir(log_folder)) > 0:
        print("Plot figure with existed log files")
    else:
        learn_kwargs = deepcopy(student_kwargs)
        learn_kwargs['student_agent'] = student_agent
        for (coach_path, log_name) in [
            (elite_student_path, 'learn_from_elite'),
            (personal_coach_path, 'learn_from_coach')
        ]:
            learn_kwargs['coach_model_path'] = coach_path
            learn_kwargs['log_path'] = os.path.join(log_folder, log_name)
            student_learn(**learn_kwargs)

    if not plot:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for log_name, label in zip(['learn_from_coach', 'learn_from_elite'], ['Personalized', 'Standard']):
        log_path = os.path.join(log_folder, log_name)
        eval_one_config(log_path, {"label": label}, smooth)
    eval_one_config(student_play_args.log_path, {"label": 'Self-study', "color": 'r', "ls": '--'}, smooth)
    plt.legend().set_draggable(True)
    plt.show()
    return


def coach_varied_level_student(student_agent, coach_path, q_delta, smooth=10, plot=True):
    # varied student self-play, and they learn under instant-coach, delayed-coach
    parent_folder = os.path.join("logs", "windy_grid_world", "finetune_q_delta_in_student_learn")
    if student_agent is InstantStudent:
        log_folder = os.path.join(parent_folder, 'coach_instantly')
    elif student_agent is DelayedStudent:
        log_folder = os.path.join(parent_folder, 'coach_in_delay')
    else:
        raise NotImplementedError

    if not os.path.exists(student_play_args.log_path):
        print("Generate student self-play log first")
        return

    if len(os.listdir(log_folder)) > 0:
        print("Plot figure with existed log files")
    else:
        learn_kwargs = deepcopy(student_kwargs)
        learn_kwargs['student_agent'] = student_agent
        learn_kwargs['coach_model_path'] = coach_path
        learn_kwargs['log_path'] = os.path.join(log_folder, 'uniformed_learn')
        student_learn(**learn_kwargs)

        learn_kwargs['q_delta'] = q_delta
        learn_kwargs['log_path'] = os.path.join(log_folder, 'varied_learn')
        student_learn(**learn_kwargs)

        learn_kwargs['log_path'] = os.path.join(log_folder, 'varied_self_play')
        student_self_play(**learn_kwargs)

    if not plot:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    log_lst = ['uniformed_learn', 'varied_learn', 'varied_self_play']
    label_lst = ['Identical Learning', 'Varied Learning', 'Varied Self-study']
    for log_name, label in zip(log_lst, label_lst):
        log_path = os.path.join(log_folder, log_name)
        eval_one_config(log_path, {"label": label}, smooth)
    eval_one_config(student_play_args.log_path, {"label": 'Identical Self-study', "color": 'r', "ls": '--'}, smooth)
    plt.legend().set_draggable(True)
    plt.show()
    return


def instruction_reward_ablation(student_agent, coach_path, smooth=10, plot=True):
    parent_folder = os.path.join("logs", "grid_world_gym", "instruction_reward_ablation")
    if student_agent is InstantStudent:
        log_folder = os.path.join(parent_folder, 'coach_instantly')
    elif student_agent is DelayedStudent:
        log_folder = os.path.join(parent_folder, 'coach_in_delay')
    else:
        raise NotImplementedError

    if not os.path.exists(student_play_args.log_path):
        print("Generate student self-play log first")
        return

    if len(os.listdir(log_folder)) > 0:
        print("Plot figure with existed log files")
    else:
        learn_kwargs = deepcopy(student_kwargs)
        learn_kwargs['student_agent'] = student_agent
        learn_kwargs['coach_model_path'] = coach_path
        learn_kwargs['log_path'] = os.path.join(log_folder, 'only_task_reward')
        student_learn(**learn_kwargs)

        learn_kwargs['evaluative_reward'] = True
        learn_kwargs['log_path'] = os.path.join(log_folder, 'with_evaluative_reward')
        student_learn(**learn_kwargs)

    if not plot:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    log_lst = ['only_task_reward', 'with_evaluative_reward']
    label_lst = ['Only Task Reward', 'Task+Evaluative Reward']
    for log_name, label in zip(log_lst, label_lst):
        log_path = os.path.join(log_folder, log_name)
        eval_one_config(log_path, {"label": label}, smooth)
    eval_one_config(student_play_args.log_path, {"label": 'Self-study', "color": 'r', "ls": '--'}, smooth)
    plt.legend().set_draggable(True)
    plt.show()
    return


if __name__ == '__main__':
    student_kwargs = {
        "learn_game": WindyWrapper(gym.make("gym_games/WindyGridWorld-v0")),
        "repeat_learn_trials": 10,   # 3 for debug, 10 for strong machine running
    }

    # finetune_budget_session_in_coach_eval(
    #     budget_session_lst=[(10, 5, np.inf)],    # (advice_budget, epis_in_session, advice_sessions)
    #     instant_coach_model_path="models/windy_grid_world/instant_coach_reset_2.pt",
    #     delayed_coach_model_path="models/windy_grid_world/delayed_coach_reset_8.pt",
    #     smooth=5, plot=True
    # )

    # standard_vs_personalized_instr(
    #     # student_agent=InstantStudent, personal_coach_path="models/windy_grid_world/instant_coach_reset_2.pt",
    #     student_agent=DelayedStudent, personal_coach_path="models/windy_grid_world/delayed_coach_reset_8.pt",
    #     elite_student_path=student_play_args.model_path,
    #     plot=True   # plot with previous log
    # )

    # coach_varied_level_student(
    #     # student_agent=InstantStudent, coach_path="models/windy_grid_world/instant_coach_reset_2.pt",
    #     student_agent=DelayedStudent, coach_path="models/windy_grid_world/delayed_coach_reset_8.pt",
    #     q_delta=1.0, plot=True
    # )

    # instruction_reward_ablation(
    #     # student_agent=InstantStudent, coach_path="models/windy_grid_world/instant_coach_reset_2.pt",
    #     student_agent=DelayedStudent, coach_path="models/windy_grid_world/delayed_coach_reset_8.pt",
    #     plot=True
    # )
