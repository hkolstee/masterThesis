# Notes Week 3

---

Worked on:
- Different reward function trained agents evaluation (also in week 2)
- Custom SAC agent for multi-agent setting

---

### Soft Actor Critic (SAC)

#### Entropy-Regularized RL:

Entropy is a quantity that (roughly speaking) says how random a variable is. A coin that is heavily weighted and almost always ends up heads, has low entropy, while a evenly weighted coin has high entropy. 

Entropy $H$ of $x$ with probability density function $P$:

$$
\begin{equation}
    H(P) = E_{x \sim P} [-log P(x)]
\end{equation}
$$

In **entropy-regularized reinforcement learning**, the agent gets a bonus reward at each time step proportional to the entropy of the policy at that point. This changes the RL problem to:

$$
\begin{equation}
    \pi^* = arg \max_\pi E_{\tau \sim \pi} [\sum^\infin_{t = 0} \gamma^t (R(s_t, a_t, s_{t+1}) + \alpha H(\pi (\cdot | s_t)))],
\end{equation}
$$
where $\alpha > 0$ is the trade-off coefficient.

We can now define an adjusted value function to include the entropy bonuses:

$$
\begin{equation}
    V^\pi (s) = E_{\tau \sim \pi} [\sum^\infin_{t = 0} \gamma^t (R(s_t, a_t, s_{t+1}) + \alpha H (\pi (\cdot | s_t))) | s_0 = s]
\end{equation}
$$

And an adjusted Q function:

$$
\begin{equation}
    Q^\pi (s, a) = E_{\tau \sim \pi} [\sum^\infin_{t = 0} \gamma^t (R(s_t, a_t, s_{t+1}) + \alpha H (\pi (\cdot | s_t))) | s_0 = s, a_0 = a]
\end{equation}
$$

Given these definitions, $V^\pi$ and $Q^\pi$ are connected by:

$$
\begin{equation}
    V^\pi (s) = E_{a \sim \pi} [Q^\pi(s, a)] + \alpha H(\pi(\cdot|s)),
\end{equation}
$$

and the bellman equation for $Q^\pi$ is:

$$
\begin{equation}
    Q^\pi (s, a) = E_{a' \sim \pi, s\sim P} [R(s, a, s') + \gamma (Q^\pi(s', a') + \alpha H(\pi(\cdot|s')))] \\
\end{equation}
$$

$$
\begin{equation}
    = E_{s' \sim P} [R(s, a, s') + \gamma V^\pi(s')] \\
\end{equation}
$$

Entropy-regularization's purpose is the encouragement of exploration. By empoying entropy-regularization more stochastic/uncertain policies are encouraged during training. Higher entropy policies tend to produce more diverse actions in different situations. This diversity can be beneficial in handling uncertainties in the environment and discovering a wider range of effective strategies. It can lead to more robust policies that perform well across various scenarios. Without entropy regularization, RL algorithms might converge to overly deterministic policies too quickly, especially in the early stages of learning. This premature convergence could result in suboptimal solutions, as the agent might not explore enough to discover better strategies.

*small explanation: According to the Bellman Equation, long-term- reward in a given action is equal to the reward from the current action combined with the expected reward from the future actions taken at the following time. Here, s' and a' denote the next state and action taken in this next state.*

#### Soft Actor Critic

SAC concurrently learns a policy $\pi_\theta$ and two $Q$ functions $Q_{\phi_1}$ $Q_{\phi_2}$. There are currently two standard variations of SAC: one uses a fixed entropy regularization coefficient $\alpha$, and another that enforces an entropy constraint by varying $\alpha$ over the course of training. The entropy-constrained variant is generally preferred. The OpenAI documentation example is the simple variant.

Both $Q$-functions are learned with Mean Squared Bellman Error (MSBE) minimization, thus by regressing to a single shared target. The bellman error measures the difference between the left (value of the state or state-action pair) and the right side (the discounted sum of rewards plus the value of the next state or next state-action pair) of the Bellman equation. The MSBE is the average of the squared Bellman errors over a set of states or state-action pairs. It is often used as a measure how well a value function approximator is performing. MSBE minimization is done via stochastic gradient descent.

The shared target is computed using target Q-networks, and the target Q-networks are obtained by polyak averaging the Q-network parameters over the course of training. The key idea behind Polyak averaging is to maintain a running average of the model parameters over the course of the training iterations. Instead of using the final parameters of the training process, Polyak averaging computes the average of the parameters obtained during the entire training trajectory. Here, new parameters are often weighted more heavily than older parameters. This stabilizes the training process, especially if the learning process is noisy or has significant fluctuations.

The shared target makes use of the clipped double-Q trick. It is used to adress the overestimation bias when estimating the action values in Q-learning algorithms. Overestimation can happen due to the selection of Q-values for the next states in the Bellman equation, where always the *maximum* estimated Q-value is used. The clipped double Q trick uses a pair of Q-value estimators. Instead of always using the max Q-value of one Q-function, the Q-function estimator used is randomly chosen. The idea is to decouple the selection of the best action from the evaluation of that action. The chosen Q-function is used to update the non-chosen Q-function in the update step. When updating one of the Q-functions, the learning is unbiased with respect to the function itself as the estimation is done by the other function. 
In the updated version, we have a model Q and a target model Q', instead of two independent models. Q' is used for action selection and Q for evaluation. Q is updated, and Q' slowly copies Q parameters using Polyak averaging.
The clipped version is once again an update. Here, when computing the update targets, we take the minimum of the two next-state actions values produced by our two Q networks; when the Q estimate from one is greated than the other, we reduce it to the minimum, avoiding overestimation. Fujimoto et al. presents another benefit of this setting: the minimum operator should provide higher value to states with lower variance estimation error. This means that the minimization will lead to a preference for states with low-variance value estimates, leading to safer policy updates with stable learning targets.

In SAC the next-state actions used in the target come from the *current policy* instead of the target policy.

Now we look at the final Q-loss function, but first, lets take a look at the SAC contribution of entropy regularization. The recursive Bellman equation, a little bit rewritten:

$$
\begin{equation}
    Q^\pi (s, a) = E_{a' \sim \pi, s'\sim P} [R(s, a, s') + \gamma (Q^\pi(s', a') - \alpha \log \pi(a'|s'))]
\end{equation}
$$

The right hand side is an expectation over next states (which come from the replay buffer) and next actions (which come from the current policy, and NOT the replay buffer). Since it is an expectation, we can approximate it with samples:

$$
\begin{equation}
    Q^\pi (s, a) \approx r + \gamma (Q^\pi(s', \tilde{a}') - \alpha \log \pi(\tilde{a}'|s')),
\end{equation}
$$

$$
\begin{equation}
    \tilde{a}' \sim \pi(\cdot | s'),
\end{equation}
$$

where $\tilde{a}'$ is freshly determined from the current policy, while $r$ and $s'$ come from the replay buffer.

Putting it all together, the loss functions for the $Q$-networks in SAC are:

$$
\begin{equation}
    L(\phi_i, \mathcal{D}) = E_{(s, a, r, s', d) \sim \mathcal{D}} \left [ \left ( Q_{\phi_i} (s, a) - y(r, s' d)  \right )^2 \right ],
\end{equation}
$$

where the target $y$ is given by:

$$
\begin{equation}
    y(r, s', d) = r + \gamma (1 - d) \left ( \min_{j=1,2} Q_{\phi_{targ, j}} (s', \tilde{a}') - \alpha \log \phi_\theta (\tilde{a}'|s')  \right )
\end{equation}
$$

Learning the policy: the policy should, in each state, act to maximize the expected future return plus expected future entropy. That is, it should maximize $V^\pi(s)$, which is expanded to:

$$
\begin{equation}
    V^\pi(s) = E_{a \sim \pi} [Q^\pi (s,a) - \alpha \log \pi (a | s)]
\end{equation}
$$

The way we optimize the policy makes use of the *reparameterization trick*, in which a sample from $\pi_\theta(\cdot|s)$ (current policy) is drawn by computing a deterministic function of state, policy parameters, and independent noise. The authors of the SAC paper use a squashed Quassian policy, which means that samples are obtained according to:

$$
\begin{equation}
    \tilde{a}_\theta (s, \xi) = \tanh (\mu_\theta(s) + \sigma_\theta(s) \odot \xi)
\end{equation}
$$

$$
\begin{equation}
    \xi \sim \mathcal{N}(0, I)
\end{equation}
$$

to be continued...

---

### Hyperparameters in RL

PAPER: [Hyperparameters in Reinforcement Learning and How To Tune Them](https://arxiv.org/abs/2306.01324)

ABSTRACT: In order to improve reproducibility, deep reinforcement learning (RL) has been adopting better scientific practices such as standardized evaluation metrics and reporting. However, the process of hyperparameter optimization still varies widely across papers, which makes it challenging to compare RL algorithms fairly. In this paper, we show that hyperparameter choices in RL can significantly affect the agent’s final performance and sample efficiency, and that the hyperparameter landscape can strongly depend on the tuning seed which may lead to overfitting. We therefore propose adopting established best practices from AutoML, such as the separation of tuning and testing seeds, as well as principled hyperparameter optimization (HPO) across a broad search space. We support this by comparing multiple state-of-theart HPO tools on a range of RL algorithms and environments to their hand-tuned counterparts, demonstrating that HPO approaches often have higher performance and lower compute overhead. As a result of our findings, we recommend a set of best practices for the RL community, which should result in stronger empirical results with fewer computational costs, better reproducibility, and thus faster progress. In order to encourage the adoption of these practices, we provide plug-andplay implementations of the tuning algorithms used in this paper at https://github.com/facebookresearch/how-to-autorl.



---

### Hyperparameters in SAC


---

### Multi-agent Soft Actor-Critic

[Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2104.06655)

---

### CTDE

[Is Centralized Training with Decentralized Execution Framework Centralized Enough for MARL?](https://arxiv.org/abs/2305.17352)

---

### OVERALL SURVEY VERY COMPLETE AND GOOD

[Multi-agent deep reinforcement learning: a survey](https://link.springer.com/article/10.1007/s10462-021-09996-w)