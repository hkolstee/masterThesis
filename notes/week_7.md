# Notes Week 7:

---

### 2022 CityLearn 4th Place: Policy Ensemble Method:

Added observations: 

Building level 16 day average of load, and solar gen for:
- next hour
- next next hour
- 6th next hour
- 12th next hour

District level 16 day average of load, and solar gen for:
- next hour
- next next hour
- 6th next hour
- 12th next hour

District level energy storage state of charge

#### Ensemble actions:
For building 1-5, the agent gets 5 actions and stds.
The weight of the action is determined by 1/std.
The ensemble action is: (1/std) - weighted average of the 5 actions

#### Soft Actor-Critic:
Standard SAC augmented with autoencoder. 
The encoder encodes the observation into a latent representation. 
The decoder tries to reconstruct the original observation from the latent representation. 
They claim this improved the convergence speed by a lot.

---
#### PAPER: [Ensembling Diverse Policies Improves Generalizability of Reinforcement Learning Algorithms in Continuous Control Tasks](https://alaworkshop2023.github.io/papers/ALA2023_paper_31.pdf)

**ABSTRACT**: Deep Reinforcement Learning (DRL) algorithms have shown great success in solving continuous control tasks. However, they often struggle to generalize to changes in the environment. Although retraining may help policies adapt to changes, it may be quite costly in some environments. Ensemble methods, which are widely used in machine learning to boost generalization, have not been commonly adopted in DRL for continuous control applications. In this work, we introduce a simple ensembling technique for DRL policies with continuous action spaces. It aggregates actions by performing weighted averaging based on the uncertainty levels of the policies. We investigate its zero-shot generalization properties in a complex continuous control domain: the optimal control of home batteries in the CityLearn environment — the subject of a 2022 international AI competition. Our results indicate that the proposed ensemble has better generalization capacity than a single policy. Further, we show that promoting diversity among policies during training can reliably improve the zero-shot performance of the ensemble in the test phase. Finally, we examine the merits of the uncertainty-based weighted averaging in an ensemble by comparing it to two alternative approaches: unweighted averaging and selecting the action of the least uncertain policy. 

#### Summary:
*Introduction:*
DRL agents have shown to tend to be overly specialized to their environment and have limited generalizability. This is especially appearant in simulated training for real world deployment, where the agents fail to perform optimally when faced with the real world perturbations. 

Closing this generalization gap is the focus of a broad body of research. Research has shown that supervised learning methods like l2 regularization, dropout, data augmentation, and batch normalization, prove useful in DRL aswell. Another approach to gain generalization in ML is to build ensembles of diverse models. The DRL research using this method is scarse. 

The paper introduces a 'Diverse $\sigma$-weighted ensemble' for continuous action spaces in DRL. The main contribution is the training of diverse DRL policies and combining them according to their uncertainty. 

*Ensembles in Deep Reinforcement Learning*
Previous research: 

- An ensemlbe of Q-networks has been used in an offline RL setting. The Q-value is estimated by choosing the minimal value outputted by the set of Q networks, which leads to the penalization of out-of-distribution actions for which there is high uncertainty in Q-value estimates. 
- Ensembling both critics and actors proved useful in stabilizing learning and improving exploration during training, where the mean and std of Q-value estimates are used to reweight Bellman backups and to perform UCB exploration.

Unlike the above, the authors problem environment involves online training. 

More previous research:

- One paper uses three different algorithms PPO, A2C, and DDPG in an ensemble to trade stock shares. In each quarter, only one of the algorithms is used to trade, but all three can be evaluated in the background. The algorithm with the best evaluation score is selected to trade in the next quarter. According to the authors, the different models are sensitive to different trends, so ensembles should work better than any of their members alone. 
- Another paper shows generalization abilities of ensembles in RL in discrete action spaces.

*Policy Diversity:*
Ensembles are highly effective in ML becuase they leverage some form of diversity, which may come from an auxiliary penalty term imposed on outputs or from variations in training data, input representations, learning algorithms, etc. For this reason, one of the goals of this paper is to investigate the effect of policy diversity on the DRL ensemble's generaliztion capacity. 

In RL, diversity can stem from variations in environment or the agent policies. In the paper, we focus on policy diversity, which can be quantified by measuring the difference between trajectories (state-action or observation-action sequences) traversed by the policies, or be evaluating the disparity in policy actions when provided with the same state/observations. 

In the papers study, they employ Diversity via Determinants (DvD) method proposed in a different paper. It adds an auxiliary diversity term to the objective, which encourages policies to output diverse actions when provided with the same observations.

EXPLANATION OF DvD HERE.... , [IMPLEMENTATION](https://github.com/holounic/DvD-TD3/?tab=readme-ov-file)

*Diverse $\sigma$-Weighted Ensembling Technique*
They train multiple actors with one shared critic in separate (but identical) environments and aggregate them in an ensemble during the test phase. The ensemble's output is a weighted average of its individual members' actions, where each weight is inversely proportional to the degree of uncertainty of the policy (std).

They train the actors on the source training houses, then use them as ensemble for a single test house, per house. The actors are trained to encourage diversity (DvD), however, the houses differ from each other already.

*Evaluation*
Dataset contains 15 houses. Train on 5, evaluate on remaining 10. Testing done via 3-fold cross-validation with the groups of 5, with 5 independent trails each. Then, a wilcoxon rank-sum test is done for statistical evidence.

Furthermore, the authors restrict training to the first 5 months of data, and perform testing on the remaining 7 months. This mimics a deployment scenario. 

While training, each pass through an episode is counted, such that, when training 4 agents, each episode counts as 4. 