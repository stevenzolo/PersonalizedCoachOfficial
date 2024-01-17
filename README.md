# Personalized Coach 
 Official implementation of the paper: A Bi-Level Reinforcement Learning Framework for Personalized Coaching

## A Novel Interactive Coaching Paradigm.
![](/imgs/general_bi-level_RL.pdf)
In this paper, we adopt a bi-level reinforcement learning framework to model the interaction among the coaches, 
the students, and the task environment.
- Lower-level RL: the student agent interacts with the task environment, like a standard RL problem.
- Higher-level RL: the coach agent observes the lower-level interaction and offers instructions to improve 
the student’s policy.

Depending on whether the coach can provide timely suggestions to students during the interaction, two basic
problem formulations are considered:
Instant Coaching           |  Delayed Coaching
:-------------------------:|:-------------------------:
![](/imgs/instant_coach.pdf)  |  ![](/imgs/delayed_coach.pdf)

- Instant Coaching: In turn-based games (such as Go), students have the opportunity to report their intended
actions to the coach before taking action, thus the instruction can be adopted and evaluated _instantly_.
- Delayed Coaching: If the coach provides instruction after the student’s action is executed, for example
in tennis training, the effectiveness can only be evaluated by _delaying_ the student’s adoption of this instruction until
the next occurrence of the same task state.

## Experiments in Windy Gridworld
The Gridworld game features a boundary area divided into multiple unit squares. The agent’s objective is to navigate from
the start square to the goal square, with available actions of {Up, Down, Left, Right}. Attempts to move beyond the
boundary do not change the agent’s position. The game complexity is heightened by introducing unknown wind forces under 
each column, influencing the agent’s movement. An optimal path of the designed map, marked with a blue line, serves as 
a reference.
![](/imgs/optimal_windy_gridworld.pdf)

### Comparison between the personalized coach and standard ones.
Results demonstrate the coaches in both scenarios outperform standard coaches and can better facilitate student learning.
Instant Scenario           |  Delayed Scenario
:-------------------------:|:-------------------------:
![](/imgs/instant_coach_vs_elite.pdf)  |  ![](/imgs/delayed_coach_vs_elite.pdf)

### Coach students with varied initial skill level
Even for the students initialized with varied skill levels, the proposed personalized coach can students achieve better 
efficiency than their self-study.
Instant Scenario           |  Delayed Scenario
:-------------------------:|:-------------------------:
![](/imgs/instant_coach_varied_level.pdf)  |  ![](/imgs/delayed_coach_varied_level.pdf)
