# Week 14 Notes:

---

#### Categorical distributions and reparameterization

If we sample from a softmax neural network output as a categorical distribution, we cannot backpropagate the gradient past this sampling process. The reason for this is that this is a stochastic sampling process defined by a discrete process, of which both it is impossible to generate a gradient.

A workaround for this is the reparameterization trick. In the case of a continuous distribution, we don't have to deal with the discrete problem. How do we solve the stochasitisity in the sampling?

By reparameterizing we sample by a linear combination of a deterministic and stochastic element. The samples are derived from a sum of the mean of the distribution and some stochastic noise (0-mean centered normal, so no bias). This noise can be seen as a constant, and the gradient can be computed.

Now, in the case of a discrete distribution, a gumbel-max trick was invented by DeepMind researchers. Some stochastic noise generated from a Gumbel Distribution is added to all log probabilities of the discrete categories. The the Gumbel distribution is typically used to model the distribution of the maximums for a number of samples pulled from other distributions. From these new probabilities, the argmax is taken which becomes the target. This approach has been mathematically proven to be equivalent to computing the softmax over a set of stochastically sampled points.

Now, we have a new problem, the argmax function is not differentiable. Next, the deepmind researchers came up with the Gumbel-softmax. Here, the argmax is replaced with a softmax operator.

---

#### Bertsekas Lecture:

Starting point: Every agent have exact state information (all), choose actions.

Model: Discrete-time (possible stochastic) system with state $x$ and control $u$. Parallel actions possible. He shows the Spider-and-Flies Example.

He shows: 5 actions per spider, which gives $5^{15}$ joint move choices. Sequential actions reduce this to $5 \cdot 15 = 75$.

DEC-POMDP: Some spiders can see some of the flies, but some others can not. Becomes notoriously difficult.

RL/Policy gradient approach for these kinds of problems: Forget about dynamic programming, parameterize the agent policies in a way that is consistent with the information pattern, then gradient descend. Problems: strictly off-line (and difficult) training (cannot adapt to on-line changes of problem data (new flies come in)). No solid theory, lack of performance gaurantees.

For Finite-state infinite horizon problems: 

Cost: 
$$
\begin{equation}
    J_\mu(x_0) = \lim_{N\rightarrow \infin} E_{w_k} \left[ \sum_{k=0}^{N-1} \gamma^k g(x_k, \mu_k(x_k), ..., \mu_m(x_k), w_k)  \right],
\end{equation}
$$

where $g$ is cost of stage $k$, and $\mu_1, ..., \mu_m$ are the policies of the agents.

Optimallity condition (minimize RHS of bellman equation):

$$
\begin{equation}
    \mu^*(x) \in \arg \min_{(u_1, ..., u_m)} E \left [  g(x, u_1, ..., u_m, w) + \alpha J^* (f(x, u_1, ..., u_m, w)) \right],
\end{equation}
$$

where $J^*(x_0) = \min_\mu J_\mu(x_0)$, which is very hard to find. A lot of RL methods replace $J^*$ with some approximation, often in the form of a neural network.

Policy iteration:
1. Start with some policy $\mu$.
2. Evaluate policy with $J_\mu$.
3. Policy improvement with Bellman Eq. with $J_mu$ instead of $J^*$.
4. Rollout policy $\tilde{\mu} \in \arg \min_{(u_1, ..., u_m)} E \left [  g(x, u_1, ..., u_m w) + \alpha J_\mu (f(x, u_1, ..., u_m w)) \right],$

With a fundamental policy improvement property: $J_{\tilde{\mu}}(x) \leq J_\mu (x)$, for all $x$.

**Proposed: New form of policy improvement, namely: one-agent-at-a-time policy improvement.**

- Usage of "guesses" to make up for missing information.
- The "guesses" are precomputed, possibly through neural network training.
- Subject of ongoing research.

Eq (2) shows that the standard rollout algorithm has a search space that is exponential in $m$.

Proposed alternative:

$$
\begin{align}
    \tilde{\mu}_1(x) \in \arg \min_{u_1} E \left [  g(x, u_1, \mu_2(x) ,..., \mu_m(x), w) + \alpha J^* (f(x, u_1, \mu_2(x),...,\mu_m(x), w)) \right],\\
    \tilde{\mu}_2(x) \in \arg \min_{u_2} E \left [  g(x, \tilde{\mu_1}(x), u_2, \mu_3(x),..., \mu_m(x), w) + \alpha J^* (f(x, \tilde{\mu_1}(x), u_2, \mu_3(x),...,\mu_m(x) w)) \right],\\
    \tilde{\mu}_m(x) \in \arg \min_{u_m} E \left [  g(x, \tilde{\mu_1}(x), ..., \tilde{\mu}_{m-1}(x), u_m, w) + \alpha J^* (f(x, \tilde{\mu_1},(x),..., \tilde{\mu}_{m-1}(x), u_m, w)) \right],\\
\end{align}
$$

where, $\mu$ is the base policy. Has search space that is LINEAR in $m$. One-agent-at-a-time is just a reformulation of standard rollout. The key theorethical fact: the cost improvement property is maintained (which can be proven).

An example for base policy for the spiders: Move along the shortest path to the closest surviving flies (Manhatten distance).

The agent-by-agent policy iteration converges to an agent-by-agent optimal policy, which may not be optimal.

**Inherently serial computation?** -> Precomputed signaling. 

Use precomputed substitute "guesses" $\hat{\mu_i}(x)$ in place of the preceding rollout controls $\tilde{\mu_i}(x)$. Signalling possibilities: use base policy controls for signaling (this may work poorly). Use a neural net representation of the rollout policy controls for signaling: $\hat{\mu_i}(x) \approx \tilde{\mu_i}(x)$, for all $i=1...m$.

https://arxiv.org/pdf/2002.04175.pdf