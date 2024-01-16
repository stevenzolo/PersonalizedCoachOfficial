"""
Hyperparameters used in this work
"""
import argparse
import numpy as np

__all__ = ['coach_args', 'student_learn_args', 'student_play_args']


coach_parser = argparse.ArgumentParser(description='Arguments used in coach training.')
coach_parser.add_argument('--train_epis', type=int, default=int(1e4)
                          )  # int(1e3) for debug, 5e3 converges most cases. Extra line can be cropped
coach_parser.add_argument('--eval_every_epis', type=int, default=100,
                          help='after training N episodes, the coach is evaluated once')
coach_parser.add_argument('--eval_epis', type=int, default=100,
                          help='To be evaluated training coach is asked to teach student N episodes.'
                          )  # follow-coach is faster than self-play
coach_parser.add_argument('--eval_trials', type=int, default=10)    # debug 1, too small cause big variance
coach_parser.add_argument('--lr', type=float, default=0.1)   # learning rate
coach_parser.add_argument('--discount', type=float, default=1.0)     # discount factor
coach_parser.add_argument('--explore_start', type=float, default=0.1)
coach_parser.add_argument('--explore_end', type=float, default=0.001)
coach_parser.add_argument('--explore_decay_epis', type=int, default=200)
coach_parser.add_argument('--model_path', type=str, help='path to saved the trained coach', default=None)
coach_args = coach_parser.parse_args()


student_learn_parser = argparse.ArgumentParser(
    description='Arguments used in trained coach evaluation / student learns from coach.')
student_learn_parser.add_argument(
    '--learn_epis', type=int, default=200
)   # allow self-play convergence, larger than 'coach.eval_epis'
student_learn_parser.add_argument(
    '--eval_epis', type=int, default=1,
    help='To be evaluated learning student is asked to play N episodes itself.'
)
student_learn_parser.add_argument(
    '--instruction_lib', type=list, default=[-1, 0, 1]
)   # [-1, 0, 1, 2] in standard coach
student_learn_parser.add_argument(
    '--advice_budget', type=int, default=10,
    help='The advice budget provides in each session for student learning.'
)
student_learn_parser.add_argument(
    '--epis_in_session', type=int, default=5,
    help='after N episodes learning, the advice budget is reset'
)   # the advice should be given before the optimal path is eroded.
student_learn_parser.add_argument(
    '--advice_sessions', type=int, default=np.inf,
    help='after N sessions instruction, the student needs play itself'
)   # in T-S framework, the advice is provided until the end
student_learn_parser.add_argument(
    '--evaluative_reward', type=bool, default=False,
    help='the intention(instant) or action(delayed) Q value will be updated with instruction evaluative reward or not'
)   # if not, the student may propose same wrong intention each time, without penalty.
student_learn_parser.add_argument('--evaluative_reward_value', type=float, default=0.1)
student_learn_parser.add_argument('--lr', type=float, default=0.5)
student_learn_parser.add_argument('--discount', type=float, default=0.1)     # under Q=-5, r=-1, gamma=0.8 boundary; 0.1 better
student_learn_args = student_learn_parser.parse_args()


student_play_parser = argparse.ArgumentParser(
    description='Arguments used in student self play')
student_play_parser.add_argument('--lr', type=float, default=0.5)   # learning rate
student_play_parser.add_argument('--discount', type=float, default=0.9)     # discount factor
student_play_parser.add_argument('--explore_start', type=float, default=0.05)
student_play_parser.add_argument('--explore_end', type=float, default=0.0)
student_play_parser.add_argument('--explore_decay_epis', type=int, default=10)
student_play_parser.add_argument(
    '--q_mu', type=float, default=-5.0,
    help='Average of Q table initialization of student'
)   # penessic for instructed action exploitation
student_play_parser.add_argument(
    '--q_delta', type=float, default=0.0,
    help='Variance of Q table initialization of student'
)   # ~0.2 in varied level students
student_play_parser.add_argument(
    '--eval_every_epis', type=int, default=5,
    help='after learning N episodes, the student is evaluated once'
)    # can be divided by coach.eval_epis and student.learn_epis is better

student_play_parser.add_argument(
    '--log_path', type=str, default='logs/windy_grid_world/student_self_play'
)
student_play_parser.add_argument(
    '--model_path', type=str, default='models/windy_grid_world/student_self_play'
)
student_play_args = student_play_parser.parse_args()
