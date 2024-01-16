"""
personalized coach agent
"""

import pickle
import numpy as np
from copy import deepcopy
from tqdm import trange

from hyperparams.grid_world_arguments import coach_args


class Coach:
    def __init__(self, coach_env, model_path, student_reset_after_epis, seed):
        # !!! keep still in the whole process
        self.env = coach_env
        self.game_env = self.env.game_env
        self.student = self.env.student
        self.coach_policy_path = model_path
        self.student_reset_after_epi = student_reset_after_epis  # find_local_optimum_then, regular_steps

        self.alpha = coach_args.lr
        self.gamma = coach_args.discount
        self.explore_start = coach_args.explore_start
        self.explore_end = coach_args.explore_end
        self.eval_every_epis = coach_args.eval_every_epis
        self.seed = seed
        np.random.seed(seed)   # important for varied simulation in multi-process

        # !!! update in process and need reset
        self.average_q_in_epis = []
        self.instr_index = None
        self.epi_return = 0
        self.epi_return_lst = []
        self.epi_state_traj = []
        self.epi_state_nums_lst = []
        self.eval_epi_return_lst = []
        self.q_value = np.zeros(
            (self.game_env.observation_space.n, self.game_env.action_space.n,
             len(self.student.instructions_lib))
        )

    @property
    def average_q(self):
        return np.average(self.q_value)

    @property
    def decayed_epsilon(self):
        epsilon_percent = min(1.0, len(self.epi_return_lst) / coach_args.explore_decay_epis)
        epsilon = round(self.explore_start * (1 - epsilon_percent) + self.explore_end * epsilon_percent, 5)
        return epsilon

    def learn_one_episode(self):
        raise NotImplementedError

    def learn(self, coach_train_epis, coach_eval_epis):
        """
        coach learn from novice to high-level by provide instructions for every policy of student.
        :param coach_eval_epis:
        :param coach_train_epis:
        :return:
        """
        for epi in trange(coach_train_epis):
            self.learn_one_episode()
            self.stats_episode()
            self.student.stats_episode()
            if len(self.epi_return_lst) % self.eval_every_epis == 0:
                self.eval_epi_return_lst.append(self.evaluate(
                    trial_students=coach_args.eval_trials,
                    coach_eval_epis=coach_eval_epis,
                ))
            if len(self.epi_return_lst) % self.student_reset_after_epi == 0:
                self.student.policy_reset(credible_instr=False)

        if self.coach_policy_path is not None:
            self.save_coach_policy()
        return self

    def predict(self, policy_obs, greedy_coach=False):
        self.epi_state_traj.append(tuple(policy_obs))
        if (not greedy_coach) and np.random.binomial(1, self.decayed_epsilon) == 1:
            self.instr_index = self.env.action_space.sample()
        else:
            game_obs, action_idea = policy_obs
            vals_ = self.q_value[game_obs, action_idea, :]
            max_values_index = [index_ for index_, value_ in enumerate(vals_) if value_ == np.max(vals_)]
            self.instr_index = np.random.choice(max_values_index)
        return self.instr_index

    def q_learning_update(self, old_s, old_intent, old_instr, rew, done, new_s, new_intent):
        """
        update q table of instructions according to SaiarSai method.
        :return:
        """
        td_target = rew + self.gamma * max(self.q_value[new_s, new_intent]) * (1 - done)  # q learning
        td_error = td_target - self.q_value[old_s, old_intent, old_instr]
        self.q_value[old_s, old_intent, old_instr] += self.alpha * td_error
        self.epi_return += rew
        return

    def evaluate(self, trial_students, coach_eval_epis):
        """
        evaluate the trained e-coach with various novice students, ask_prob=1.0
        compare the episode reward between student self play and that following e-coach
        """
        student = deepcopy(self.student)   # new novice student
        coach_q_value = deepcopy(self.q_value)  # deepcopy to prevent pollution
        students_performance_lst = []
        for _ in range(trial_students):
            student.policy_reset(credible_instr=True)
            student.stats_reset()
            student.learn(coach_q_value=coach_q_value, student_learn_epis=coach_eval_epis)   # greedy coach
            students_performance_lst.append(np.average(student.eval_score_lst[-3:]))  # converged evaluation
        return round(np.average(students_performance_lst), 3)

    def save_coach_policy(self):
        with open(self.coach_policy_path, 'wb') as f:
            pickle.dump(self.q_value, f)

    def stats_episode(self):
        self.average_q_in_epis.append(self.average_q)
        self.epi_return_lst.append(self.epi_return)
        self.epi_return = 0
        self.epi_state_nums_lst.append(len(set(self.epi_state_traj)))
        self.epi_state_traj = []
        return

    def reset(self):
        self.q_value = np.zeros(
            (self.game_env.observation_space.n, self.game_env.action_space.n,
             len(self.student.instructions_lib))
        )
        self.average_q_in_epis = []
        self.instr_index = None
        self.epi_return = 0
        self.epi_return_lst = []
        self.epi_state_traj = []
        self.epi_state_nums_lst = []
        return


class InstantCoach(Coach):
    def __init__(self, coach_env, model_path, student_reset_after_epis, seed):
        super(InstantCoach, self).__init__(
            coach_env=coach_env, model_path=model_path,
            student_reset_after_epis=student_reset_after_epis,
            seed=seed
        )

    def learn_one_episode(self):
        policy_obs, info = self.env.reset(seed=self.seed)  # contains {game state, student intention}
        while 1:
            instr_index = self.predict(policy_obs, greedy_coach=False)
            self.student.step_instruction(instr_index=instr_index, intention_diff=0)  # new student.actual_action_index
            policy_obs, reward, terminated, truncated, info = self.env.step()
            self.q_learning_update(
                old_s=self.game_env.last_state, old_intent=self.student.last_action_index_idea,
                old_instr=self.instr_index, rew=reward, done=terminated,
                new_s=self.game_env.state, new_intent=self.student.action_index_idea
            )
            if terminated or truncated:
                break
        return


class DelayedCoach(Coach):
    def __init__(self, coach_env, model_path, student_reset_after_epis, seed):
        super(DelayedCoach, self).__init__(
            coach_env=coach_env, model_path=model_path,
            student_reset_after_epis=student_reset_after_epis,
            seed=seed
        )

    def learn_one_episode(self):
        policy_obs, info = self.env.reset()
        self.student.get_actual_action()
        while 1:
            policy_obs[-1] = self.student.actual_action_index   # delayed case, the coach can only see actual action
            instr_index = self.predict(policy_obs)
            self.student.update_memory(
                state=self.game_env.state, actual_action_ix=self.student.actual_action_index,
                instr_ix=instr_index)
            policy_obs, reward, terminated, truncated, info = self.env.step()
            if self.student.action_instructed:  # self.game_env.last_state in self.student.instr_memory and
                prev_instred_action_ix, prev_instr_ix = self.student.instr_memory[self.game_env.last_state]['old_instr']
                self.student.get_actual_action()   # update self.student.actual_action_index
                self.q_learning_update(
                    old_s=self.game_env.last_state, old_intent=prev_instred_action_ix,
                    old_instr=prev_instr_ix, rew=reward, done=terminated, new_s=self.game_env.state,
                    new_intent=self.student.actual_action_index  # delayed case, the coach can only see actual action
                )
            else:
                self.student.get_actual_action()  # update self.student.actual_action_index

            if terminated or truncated:
                break
        return

