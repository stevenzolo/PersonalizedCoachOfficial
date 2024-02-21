# Personalized Teacher 
 Official implementation of the paper: 
 
 Towards Personalized Learning: A Bi-Level Reinforcement Learning Framework for Machine Teaching Human Tasks

## A Novel Interactive Teaching Paradigm.
<p align="center">
  <img src="/imgs/general_bi-level_RL.png" width="400">
</p>
In this paper, we adopt a bi-level reinforcement learning framework to model the interaction among the teachers, 
the students, and the task environment.
- Lower-level RL: the student agent interacts with the task environment, like a standard RL problem.
- Higher-level RL: the teacher agent observes the lower-level interaction and offers instructions to improve 
the student’s policy.

Depending on whether the teacher can provide timely suggestions to students during the interaction, two basic
problem formulations are considered:
Instant Teaching           |  Delayed Teaching
:-------------------------:|:-------------------------:
![](/imgs/instant_coach.png)  |  ![](/imgs/delayed_coach.png)

- Instant Teaching: In turn-based games (such as Go), students have the opportunity to report their intended
actions to the teacher before taking action, thus the instruction can be adopted and evaluated _instantly_.
- Delayed Teaching: If the teacher provides instruction after the student’s action is executed, for example
in tennis training, the effectiveness can only be evaluated by _delaying_ the student’s adoption of this instruction until
the next occurrence of the same task state.

## Experiments in Windy Gridworld
The Gridworld game features a boundary area divided into multiple unit squares. The agent’s objective is to navigate from
the start square to the goal square, with available actions of {Up, Down, Left, Right}. Attempts to move beyond the
boundary do not change the agent’s position. The game complexity is heightened by introducing unknown wind forces under 
each column, influencing the agent’s movement. An optimal path of the designed map, marked with a blue line, serves as 
a reference.
<p align="center">
  <img src="/imgs/optimal_windy_gridworld.png" width="400">
</p>

### Comparison between the personalized teacher and elite-player teachers.
Results demonstrate that personalized teachers in both scenarios outperform elite-player and can better facilitate student learning.
Instant Scenario           |  Delayed Scenario
:-------------------------:|:-------------------------:
![](/imgs/instant_coach_vs_elite.png)  |  ![](/imgs/delayed_coach_vs_elite.png)

### Teach students with varied initial skill levels
Even for the students initialized with varying skill levels, the proposed personalized teacher help can students achieve better 
efficiency than their self-study.
Instant Scenario           |  Delayed Scenario
:-------------------------:|:-------------------------:
![](/imgs/instant_coach_varied_level.png)  |  ![](/imgs/delayed_coach_varied_level.png)
