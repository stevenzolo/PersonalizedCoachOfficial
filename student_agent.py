"""
student agent
"""

import pickle
import numpy as np
from copy import deepcopy
import gymnasium as gym
import matplotlib.pyplot as plt

import gym_games
from gym_games.wrappers import WindyWrapper
from hyperparams.grid_world_arguments import student_learn_args, student_play_args
import matplotlib
import warnings
matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")


class Student(object):
    def __init__(
            self, windy_env, q_mu=student_play_args.q_mu, q_delta=student_play_args.q_delta,
            eval_every_epis=student_play_args.eval_every_epis, advice_budget=student_learn_args.advice_budget,
            model_path=None, log_path=None, seed=None
    ):
        # !!! keep still in the whole process
        self.game_env = windy_env
        self.advice_budget = advice_budget
        self.eval_every_epis = eval_every_epis
        self.q_mu = q_mu    # pessimistic initialization
        self.q_delta = q_delta
        self.model_path = model_path
        self.log_path = log_path
        self.instructions_lib = student_learn_args.instruction_lib
        self.self_play_alpha = student_play_args.lr  # 0.5
        self.self_play_gamma = student_play_args.discount  # 0.9

        self.follow_coach_alpha = student_learn_args.lr   # 0.5
        self.follow_coach_gamma = student_learn_args.discount   # 0.5
        self.epsilon_start = student_play_args.explore_start  # 0.05,
        self.epsilon_end = student_play_args.explore_end      # 0.001
        np.random.seed(seed)

        # !!! update in process and need reset
        self.epi_return = 0
        self.consumed_budget_session = 0
        self.remained_advice = advice_budget
        self.epi_return_lst = []
        self.eval_score_lst = []
        self.average_q_lst = []
        self.q_value = np.random.normal(
            self.q_mu, self.q_delta,
            self.game_env.observation_space.n * self.game_env.action_space.n
        ).reshape((self.game_env.observation_space.n, self.game_env.action_space.n))
        self.last_action_index_idea = None
        self.action_index_idea = None
        self.actual_action_index = None   # action idea + instruction
        self.action_instructed = False  # distinguish the actual action benefited from instruction or not

    @property
    def _average_q(self):
        return np.round(np.average(self.q_value), 5)

    @property
    def decayed_epsilon(self):
        epsilon_percent = min(1.0, len(self.epi_return_lst) / student_play_args.explore_decay_epis)
        epsilon = np.round(self.epsilon_start * (1 - epsilon_percent) + self.epsilon_end * epsilon_percent, 5)
        return epsilon

    def _round_action(self, action):
        return action % self.game_env.action_space.n  # actions that student can choose from

    @staticmethod
    def coach_eval_predict(policy_obs, coach_q_value):
        """
        distinguished from coach.predict() to prevent pollution;
        greedily employ coach's instruction
        :param policy_obs:
        :param coach_q_value:
        :return:
        """
        game_obs, action_idea = policy_obs
        vals_ = coach_q_value[game_obs, action_idea, :]
        max_values_index = [index_ for index_, value_ in enumerate(vals_) if value_ == np.max(vals_)]
        instr_index = np.random.choice(max_values_index)
        return instr_index

    @staticmethod
    def load_coach_policy(coach_model_path):
        with open(coach_model_path, 'rb') as f:
            coach_q_value = np.round(pickle.load(f), 5)   # Important for compare
        return coach_q_value

    def predict_action_idea(self, s, greedy_student=False):
        self.last_action_index_idea = deepcopy(self.action_index_idea)
        if (not greedy_student) and np.random.binomial(1, self.decayed_epsilon) == 1:
            self.action_index_idea = np.random.choice(np.arange(self.game_env.action_space.n))
        else:
            self.action_index_idea = np.random.choice(
                [action_ for action_, value_ in enumerate(self.q_value[s]) if value_ == np.max(self.q_value[s])])
        return self.action_index_idea

    def predict(self, game_obs, coach_q_value):
        # predict actual idea under coach's instruction
        raise NotImplementedError

    def predict_under_elite(self, game_obs, elite_q_value):
        # predict actual idea under elite's instruction
        raise NotImplementedError

    def q_learning_update(self, last_a, reward, done, evaluative_reward):
        last_s = self.game_env.last_state
        curr_s = self.game_env.state
        discount = self.follow_coach_gamma if self.action_instructed else self.self_play_gamma
        lr = self.follow_coach_alpha if self.action_instructed else self.self_play_alpha
        if self.action_instructed and evaluative_reward and last_a == self.action_index_idea:
            reward += student_learn_args.evaluative_reward_value
        td_target = reward + discount * max(self.q_value[curr_s]) * (1 - done)

        # new action generated after student policy update, self-play != learn
        td_err = td_target - self.q_value[last_s, last_a]
        self.q_value[last_s, last_a] += lr * td_err
        self.epi_return += reward
        return

    def learn(
            self, coach_q_value, student_learn_epis,
            evaluative_reward=student_learn_args.evaluative_reward,
            epis_in_session=student_learn_args.epis_in_session,
            advice_sessions=student_learn_args.advice_sessions
    ):
        """
        student learn from novice to high-level by high-level coach.
        :param advice_sessions:
        :param epis_in_session:
        :param student_learn_epis:
        :param coach_q_value:
        :param evaluative_reward:
        :return:
        """
        game_obs, info = self.game_env.reset()
        state_traj = [game_obs]
        while 1:
            if coach_q_value.shape == self.q_value.shape:
                action = self.predict_under_elite(game_obs=game_obs, elite_q_value=coach_q_value)
            else:
                action = self.predict(game_obs=game_obs, coach_q_value=coach_q_value)
            game_obs, reward, terminated, truncated, info = self.step_game(action)  # after step instruction
            state_traj.append(game_obs)
            self.q_learning_update(
                last_a=self.actual_action_index, reward=reward, done=terminated,    #  or truncated
                evaluative_reward=evaluative_reward,
            )
            if terminated or truncated:
                self.stats_episode()
                if len(self.epi_return_lst) % self.eval_every_epis == 0:
                    eval_score = self.evaluate(deepcopy(self), 1)
                    self.eval_score_lst.append(eval_score)
                if len(self.epi_return_lst) % epis_in_session == 0 and self.consumed_budget_session < advice_sessions:
                    self.remained_advice = self.advice_budget
                    self.consumed_budget_session += 1
                if len(self.epi_return_lst) >= student_learn_epis:
                    break
                game_obs, info = self.game_env.reset()
                state_traj = [game_obs]
        return self

    def self_play(self, student_learn_epis):
        """
        Different from play under coach, now action idea is equal to actual action.
        """
        print("=========student is self playing=========")
        obs, info = self.game_env.reset()
        while 1:
            action = self.predict_action_idea(obs, greedy_student=False)  # updated epsilon
            obs, reward, terminated, truncated, info = self.step_game(action)
            self.q_learning_update(
                last_a=self.action_index_idea, reward=reward, done=terminated,  # or truncated
                evaluative_reward=False,  # different from self.learn()
            )
            if truncated or terminated:
                self.stats_episode()
                if len(self.epi_return_lst) % self.eval_every_epis == 0:
                    eval_score = self.evaluate(deepcopy(self), 1)
                    self.eval_score_lst.append(eval_score)
                if len(self.epi_return_lst) >= student_learn_epis:
                    break
                obs, info = self.game_env.reset()
        print("=========self playing done=========")
        if self.model_path is not None:
            self.save_student_policy()
        return self

    @staticmethod
    def evaluate(student, eval_trials):
        # No policy update and exploration
        eval_scores = []
        for trial in range(eval_trials):
            obs, info = student.game_env.reset()
            while 1:
                action = student.predict_action_idea(obs, greedy_student=True)
                obs, reward, terminated, truncated, info = student.step_game(action)
                if terminated or truncated:
                    break
            eval_scores.append(student.game_env.step_count)
        return np.average(eval_scores)

    def step_instruction(self, instr_index, intention_diff):
        """
        Given instructions, turn it into delta action, and impose improved action to the game.
        :param instr_index: instruction index, different action share the same instruction set length.
        :param intention_diff: the intention difference between instruction proposed and instruction adopted.
        :return:
        """
        instructed_delta = self.instructions_lib[int(instr_index)]
        self.actual_action_index = self._round_action(self.action_index_idea + instructed_delta + intention_diff)
        return self.actual_action_index

    def step_game(self, actual_action):
        """
        wrapper of game.step(), worked in e-coach training
        :param actual_action:
        :return:
        """
        obs, reward, terminated, truncated, info = self.game_env.step(actual_action)
        return obs, reward, terminated, truncated, info

    def save_student_policy(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.q_value, f)

    def load_student_policy(self):
        with open(self.model_path, 'rb') as f:
            self.q_value = np.round(pickle.load(f), 5)
        return self

    def stats_episode(self):
        self.average_q_lst.append(self._average_q)
        self.epi_return_lst.append(self.epi_return)
        self.epi_return = 0
        return

    def policy_reset(self, credible_instr):
        self.last_action_index_idea = None
        self.action_index_idea = None
        self.actual_action_index = None  # action idea + instruction
        self.q_value = np.random.normal(
            self.q_mu, self.q_delta,
            self.game_env.observation_space.n * self.game_env.action_space.n
        ).reshape((self.game_env.observation_space.n, self.game_env.action_space.n))
        self.action_instructed = credible_instr  # distinguish the actual action benefited from instruction or not
        self.remained_advice = self.advice_budget
        self.consumed_budget_session = 0
        return self

    def stats_reset(self):
        self.epi_return = 0
        self.epi_return_lst = []
        self.average_q_lst = []
        self.eval_score_lst = []


class DelayedStudent(Student):
    def __init__(
            self, windy_env, seed,
            q_mu=student_play_args.q_mu, q_delta=student_play_args.q_delta,
            eval_every_epis=student_play_args.eval_every_epis,
            advice_budget=student_learn_args.advice_budget,
            model_path=None, log_path=None
    ):
        super().__init__(
            windy_env=windy_env, q_mu=q_mu, q_delta=q_delta, advice_budget=advice_budget,
            eval_every_epis=eval_every_epis,
            log_path=log_path, model_path=model_path, seed=seed
        )
        self.instr_memory = dict()

    def get_actual_action(self):
        # greedily employ coach's policy
        if self.game_env.state in self.instr_memory:
            prev_instred_action_ix, prev_instr_ix = self.instr_memory[self.game_env.state]['new_instr']
            intention_diff = prev_instred_action_ix - self.action_index_idea
            self.step_instruction(instr_index=prev_instr_ix, intention_diff=intention_diff)
            self.action_instructed = True
        else:
            self.actual_action_index = deepcopy(self.action_index_idea)
            self.action_instructed = False
        return

    def update_memory(self, state, actual_action_ix, instr_ix):
        if state not in self.instr_memory:
            self.instr_memory[state] = dict()
        else:
            self.instr_memory[state]['old_instr'] = deepcopy(self.instr_memory[state]['new_instr'])
        self.instr_memory[state]['new_instr'] = (actual_action_ix, instr_ix)
        return

    def predict(self, game_obs, coach_q_value):
        if self.remained_advice:
            policy_state = [game_obs]
            self.predict_action_idea(game_obs)
            self.get_actual_action()  # update self.actual_action_index
            policy_state.append(self.actual_action_index)
            instr_index = self.coach_eval_predict(policy_state, coach_q_value)
            self.update_memory(
                state=self.game_env.state, actual_action_ix=self.actual_action_index,
                instr_ix=instr_index
            )
            self.remained_advice -= 1
        else:
            self.actual_action_index = self.predict_action_idea(game_obs)
            self.action_instructed = False
        return self.actual_action_index

    def predict_under_elite(self, game_obs, elite_q_value):
        if self.remained_advice:
            self.predict_action_idea(game_obs)
            self.get_actual_action()  # update self.actual_action_index
            action_elite = np.argmax(elite_q_value[game_obs[0], game_obs[1], :])
            delta = action_elite - self.actual_action_index
            if delta == -3:
                delta = 1
            elif delta == 3:
                delta = -1
            elif delta == 2 or delta == -2:
                delta = np.random.choice(self.instructions_lib)
            else:
                pass
            instr_index = self.instructions_lib.index(delta)
            self.update_memory(
                state=tuple(self.game_env.state), actual_action_ix=self.actual_action_index,
                instr_ix=instr_index
            )
            self.remained_advice -= 1
        else:
            self.actual_action_index = self.predict_action_idea(game_obs)
            self.action_instructed = False
        return self.actual_action_index

    def policy_reset(self, credible_instr):
        self.instr_memory = dict()
        super().policy_reset(credible_instr=credible_instr)


class InstantStudent(Student):
    def __init__(
            self, windy_env, seed,
            q_mu=student_play_args.q_mu, q_delta=student_play_args.q_delta,
            eval_every_epis=student_play_args.eval_every_epis,
            advice_budget=student_learn_args.advice_budget,
            model_path=None, log_path=None
    ):
        super().__init__(
            windy_env=windy_env, q_mu=q_mu, q_delta=q_delta, advice_budget=advice_budget,
            eval_every_epis=eval_every_epis,
            log_path=log_path, model_path=model_path, seed=seed
        )

    def predict(self, game_obs, coach_q_value):
        if self.remained_advice > 0:
            policy_state = [game_obs]
            action_idea = self.predict_action_idea(game_obs)
            policy_state.append(action_idea)
            instr_index = self.coach_eval_predict(policy_state, coach_q_value)  # greedily
            actual_action = self.step_instruction(instr_index, intention_diff=0)
            self.action_instructed = True
            self.remained_advice -= 1
        else:
            actual_action = self.predict_action_idea(game_obs)
            self.action_instructed = False
        self.actual_action_index = actual_action
        return actual_action

    def predict_under_elite(self, game_obs, elite_q_value):
        if self.remained_advice > 0:
            action_idea = self.predict_action_idea(game_obs)
            action_elite = np.argmax(elite_q_value[game_obs[0], game_obs[1], :])
            delta = action_elite - action_idea
            if delta == -3:
                delta = 1
            elif delta == 3:
                delta = -1
            elif delta == 2 or delta == -2:
                delta = np.random.choice(self.instructions_lib)
            else:   # delta = 0, 1, -1
                pass
            instr_index = self.instructions_lib.index(delta)
            actual_action = self.step_instruction(instr_index, intention_diff=0)
            self.action_instructed = True
            self.remained_advice -= 1
        else:
            actual_action = self.predict_action_idea(game_obs)
            self.action_instructed = False
        self.actual_action_index = actual_action
        return actual_action


def eval_varied_game_reward_design():
    grid_game = WindyWrapper(gym.make("gym_games/WindyGridWorld-v0"))
    plt.figure()
    for game in [grid_game]:
        student = Student(windy_env=game)
        student.model_path = student_play_args.model_path   # to get elite policy
        epi_returns_collection = []
        student.self_play(student_learn_epis=student_learn_args.learn_epis)
        print('Converged path length: ', np.average(student.eval_score_lst[-3:]))
        epi_returns_collection.append(student.epi_return_lst)
        plt.plot(
            [np.average(student.epi_return_lst[ix-30: ix]) for ix in range(30, len(student.epi_return_lst))]
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    eval_varied_game_reward_design()


