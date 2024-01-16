"""
Parallel implementation of coach training, student learning, and student self-study.
"""


import pickle
import numpy as np
from copy import deepcopy
import gymnasium as gym
from multiprocessing import Pool

from utils import worker_wrapper
from gym_games.wrappers import WindyWrapper, CoachWrapper
from hyperparams.grid_world_arguments import coach_args, student_learn_args, student_play_args
from coach_agent import DelayedCoach, InstantCoach
from student_agent import Student, DelayedStudent, InstantStudent


def coach_training_trial(
        coach_agent, coach_game, model_path, student_reset_after_epis,
        coach_seed, student_seed
):
    """
    One coach training trial for a single cpu process
    :param coach_game:
    :param coach_agent: InstantCoach or DelayedCoach
    student following the coach
    :param model_path: save trained coach model
    :param student_reset_after_epis: after N episodes, the student is reset and the intention is random.
    :param coach_seed: provide different selection in np.random() in each cpu process
    :param student_seed: similar to coach, random choice in student should also be different.
    :return:
    """
    if coach_agent is InstantCoach:
        coached_student = InstantStudent(windy_env=coach_game, seed=student_seed)
    elif coach_agent is DelayedCoach:
        coached_student = DelayedStudent(windy_env=coach_game, seed=student_seed)
    else:
        raise NotImplementedError
    trained_coach = coach_agent(
        coach_env=CoachWrapper(game_env=coach_game, student=coached_student),
        model_path=model_path,
        student_reset_after_epis=student_reset_after_epis,
        seed=coach_seed
    )
    trained_coach.learn(
        coach_train_epis=coach_args.train_epis, coach_eval_epis=coach_args.eval_epis
    )

    trial_res = {
        "average_q_in_epis": trained_coach.average_q_in_epis, "q_value": trained_coach.q_value,
        "coach_return_in_epis": trained_coach.epi_return_lst,
        "coach_state_nums": trained_coach.epi_state_nums_lst,
        "eval_return_in_epis": trained_coach.eval_epi_return_lst
    }
    return trial_res


def coach_train(
        coach_agent, coach_game, repeat_train_trials, student_reset_after_epis,
        **kwargs
):  # -> instant-work instruction
    print("student_reset_after_epis", student_reset_after_epis)
    log_path = kwargs['log_path'] if 'log_path' in kwargs else None
    model_path = kwargs['model_path'] if 'model_path' in kwargs else None
    trial_kwargs = {
        'coach_agent': coach_agent,
        'coach_game': coach_game,
        'model_path': model_path,
        'student_reset_after_epis': student_reset_after_epis
    }

    train_coach_log_dict = {
        "student_reset_after_epis": student_reset_after_epis,
        "model_path": model_path, "trial_results": []
    }
    func_lst = [coach_training_trial for _ in range(repeat_train_trials)]
    trial_kwargs_lst = []
    for _ in range(repeat_train_trials):
        trial_kwargs_process = deepcopy(trial_kwargs)
        trial_kwargs_process["coach_seed"] = np.random.randint(1, 100)
        trial_kwargs_process["student_seed"] = np.random.randint(1, 100)
        trial_kwargs_lst.append(trial_kwargs_process)

    with Pool(repeat_train_trials) as mp_pool:
        # 'worker_wrapper' must in main, not sub-def
        # to obtain varied simulation, the random seed must be passed out of parallel processing, as parameter
        trial_res_lst = mp_pool.map(worker_wrapper, zip(func_lst, trial_kwargs_lst))
        train_coach_log_dict["trial_results"] = trial_res_lst

    if log_path is not None:
        with open(log_path, "wb") as tcl:
            pickle.dump(train_coach_log_dict, tcl)
    return


def student_learn_trial(
        student_agent, learn_game, model_path, q_delta,
        advice_budget, epis_in_session, advice_sessions,
        coach_model_path, evaluative_reward, student_seed
):
    if student_agent is InstantStudent:
        coached_student = InstantStudent(
            windy_env=learn_game, seed=student_seed,
            q_delta=q_delta, model_path=model_path, advice_budget=advice_budget
        )
    elif student_agent is DelayedStudent:
        coached_student = DelayedStudent(
            windy_env=learn_game, seed=student_seed,
            q_delta=q_delta, model_path=model_path, advice_budget=advice_budget
        )
    else:
        raise NotImplementedError

    coach_q_value = coached_student.load_coach_policy(coach_model_path)
    coached_student.learn(
        coach_q_value=coach_q_value, student_learn_epis=student_learn_args.learn_epis,
        evaluative_reward=evaluative_reward,
        epis_in_session=epis_in_session, advice_sessions=advice_sessions
    )
    trial_res = {
        "average_q_in_epis": coached_student.average_q_lst, "q_value": coached_student.q_value,
        "return_in_epis": coached_student.epi_return_lst, "eval_score_lst": coached_student.eval_score_lst
    }
    return trial_res


def student_learn(
        student_agent, learn_game, repeat_learn_trials, coach_model_path, **kwargs
):   # act as coach evaluation
    log_path = kwargs['log_path'] if 'log_path' in kwargs else None
    q_delta = kwargs['q_delta'] if 'q_delta' in kwargs else student_play_args.q_delta
    evaluative_reward = kwargs['evaluative_reward'] if 'evaluative_reward' in kwargs else student_learn_args.evaluative_reward
    advice_budget = kwargs['advice_budget'] if 'advice_budget' in kwargs else student_learn_args.advice_budget
    epis_in_session = kwargs['epis_in_session'] if 'epis_in_session' in kwargs else student_learn_args.epis_in_session
    advice_sessions = kwargs['advice_sessions'] if 'advice_sessions' in kwargs else student_learn_args.advice_sessions
    trial_kwargs = {
        'student_agent': student_agent,
        'learn_game': learn_game,
        'model_path': kwargs['model_path'] if 'model_path' in kwargs else None,
        'q_delta': q_delta,
        'advice_budget': advice_budget,
        'epis_in_session': epis_in_session,
        'advice_sessions': advice_sessions,
        'coach_model_path': coach_model_path,
        'evaluative_reward': evaluative_reward
    }

    student_learn_log_dict = {
        'advice_budget': advice_budget,
        'epis_in_session': epis_in_session,
        'advice_sessions': advice_sessions,
        'coach_model_path': coach_model_path,
        "evaluative_reward": evaluative_reward,
        "trial_results": []
    }
    func_lst = [student_learn_trial for _ in range(repeat_learn_trials)]
    trial_kwargs_lst = []
    for _ in range(repeat_learn_trials):
        trial_kwargs_process = deepcopy(trial_kwargs)
        trial_kwargs_process['student_seed'] = np.random.randint(1, 100)
        trial_kwargs_lst.append(trial_kwargs_process)

    with Pool(min(repeat_learn_trials, 5)) as mp_pool:
        # 'worker_wrapper' must in main, not sub-def
        # to obtain varied simulation, the random seed must be passed out of parallel processing, as parameter
        trial_res_lst = mp_pool.map(worker_wrapper, zip(func_lst, trial_kwargs_lst))
        student_learn_log_dict["trial_results"] = trial_res_lst

    if log_path is not None:
        with open(log_path, "wb") as icl:
            pickle.dump(student_learn_log_dict, icl)
    return


def self_play_trial(learn_game, student_seed, q_delta):
    novice_student = Student(windy_env=learn_game, seed=student_seed, q_delta=q_delta)
    novice_student.self_play(student_learn_args.learn_epis)
    trial_res = {
        "average_q_in_epis": novice_student.average_q_lst, "q_value": novice_student.q_value,
        "return_in_epis": novice_student.epi_return_lst, "eval_score_lst": novice_student.eval_score_lst
    }
    return trial_res


def student_self_play(learn_game, repeat_learn_trials, **kwargs):
    log_path = kwargs['log_path'] if 'log_path' in kwargs else None
    q_delta = kwargs['q_delta'] if 'q_delta' in kwargs else student_play_args.q_delta
    trial_kwargs = {
        'learn_game': learn_game,
        'q_delta': q_delta,
    }
    student_learn_log_dict = {
        "log_path": log_path,
        'q_delta': q_delta,
        "trial_results": []
    }
    func_lst = [self_play_trial for _ in range(repeat_learn_trials)]
    trial_kwargs_lst = []
    for _ in range(repeat_learn_trials):
        trial_kwargs_process = deepcopy(trial_kwargs)
        trial_kwargs_process["student_seed"] = np.random.randint(1, 100)
        trial_kwargs_lst.append(trial_kwargs_process)
    with Pool(repeat_learn_trials) as mp_pool:
        # 'worker_wrapper' must in main, not sub-def
        # to obtain varied simulation, the random seed must be passed out of parallel processing, as parameter
        trial_res_lst = mp_pool.map(worker_wrapper, zip(func_lst, trial_kwargs_lst))
        student_learn_log_dict["trial_results"] = trial_res_lst

    if log_path is not None:
        with open(log_path, "wb") as icl:
            pickle.dump(student_learn_log_dict, icl)
    return


if __name__ == '__main__':
    self_play_kwargs = {
        'learn_game': WindyWrapper(gym.make("gym_games/WindyGridWorld-v0")),
        'repeat_learn_trials': 3,
        'log_path': student_play_args.log_path
    }
    # student_self_play(**self_play_kwargs)

    coach_kwargs = {
        'coach_game': WindyWrapper(gym.make("gym_games/WindyGridWorld-v0")),
        'repeat_train_trials': 1,   # 3 for debug, 10 for strong machine running
        # 'model_path': "models/windy_grid_world/instant_coach_reset_2.pt",
        # 'log_path': "logs/windy_grid_world/instant_coach_reset_2",
        # 'model_path': "models/windy_grid_world/delayed_coach_reset_8.pt"
    }
    # coach_train(
    #     coach_agent=InstantCoach, student_reset_after_epis=2,
    #     # coach_agent=DelayedCoach, student_reset_after_epis=8,
    #     **coach_kwargs
    # )

    student_kwargs = {
        "learn_game": WindyWrapper(gym.make("gym_games/WindyGridWorld-v0")),
        "repeat_learn_trials": 1,   # 3 for debug, 10 for strong machine running
        # "evaluative_reward": True
    }
    # student_learn(
    #     student_agent=InstantStudent, coach_model_path="models/windy_grid_world/instant_coach_reset_2.pt",
    #     # student_agent=DelayedStudent, coach_model_path="models/windy_grid_world/delayed_coach_reset_8.pt",
    #     **student_kwargs
    # )




