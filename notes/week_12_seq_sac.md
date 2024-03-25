# Notes Week 12

---

### Sequential critic: Actor-Critic

For every update step:

$$
\begin{align*}
    &A \sim \pi(\cdot| S; \theta)\\
    &\text{Take action } A \text{ observe } S', R\\
    &\theta \leftarrow R + \gamma V(S'; \mathbf{w}) - V(S; \mathbf{w})\\
    &\mathbf{w} \leftarrow \mathbf{w} + \lambda^\mathbf{w} \delta \nabla V(S; \mathbf{w})\\
    &\theta \leftarrow \theta + \lambda^\theta I \delta \nabla \ln \pi (A | S; \theta)
\end{align*}
$$

Where, if the policy is optimal, the left $(R + \gamma V(S'))$ and right $(V(S))$ hand side should cancel out, resulting in $\theta = 0$. This is by definition of the value function, where the value of the current state is the same as the reward gained from the transition to the next state plus the value of this next state.

#### For an $m$-agent problem, we change this to:

$$
\begin{align*}
    &A_1 \sim \pi_1 (\cdot | S; \theta_1), A_2 \sim \pi_2(\cdot | S, A_1; \theta_2), \ldots, A_m \sim \pi_m(\cdot | S, A_1, \ldots, A_{m-1};\theta_m)\\
    &\text{Take actions } A_1,\ldots,A_m \text{ observe } S', R\\
    &\delta_i \leftarrow R + \gamma V_i(S, A_1, \ldots, A_{i-1}; \mathbf{w}_i) - V_{i-1}(S, A_1, \ldots, A_{i-2}; \mathbf{w}_{i-1}), &\text{for }i = 1, \ldots, m-1 \\
    &\delta_m \leftarrow R(S, A_1, \ldots, A_m) + \gamma V_1(S'; \mathbf{w}_1) - V_{m-1}(S, A_1, \ldots, A_{m-1}; \mathbf{w}_{m-1})\\
    &\mathbf{w}_i \leftarrow \mathbf{w}_i + \lambda^\mathbf{w}_i \delta_i \nabla V_i(S, A_1, \ldots, A_{i-1}; \mathbf{w}_i), &\text{for }i = 1, \ldots, m \\
    &\theta_i \leftarrow \theta_i + \lambda^{\theta_i} I \delta_i \nabla \ln \pi_i(A|S, A_1, \ldots, A_{i-1}), &\text{for }i = 1, \ldots, m \\\\
\end{align*}
$$

Here, we make the comparison between the value given a state and some or no action, and the value of this same state given an extra action. At the end of the sequence, the last V-function with all state action information is compared to the first V-function on the next state.

The relation of the preceding stage and the following stage (state plus one more action), is a powerful comparison on agent actions, where the target value is conditioned on the extra action, therefore, if the policy is optimal, the intermediate values should be equal.  

---

### Sequential critic: Soft Actor-Critic

Given the objective functions:

$$
\begin{align*}
    &y(R, S', d) = R + \gamma(1-d) \left( \min_{i=1,2} Q^{targ}_i(S', \tilde{A}'; \mathbf{w}_{targ}) - \alpha \log \pi (\tilde{A}'| S'; \theta)\right), &\tilde{A}' \sim \pi(\cdot|S'; \theta)\\
    &J_Q (\mathbf{w}) = \frac{1}{2}\sum_{i=1,2}(Q_i(S, A; \mathbf{w}_i) - y(R, S', d))^2 \\
    &J_\pi(\theta) =  \alpha \log \pi_\theta(\tilde{A}(S|\theta) | S; \theta) - \min_{i=1,2} Q_i (S, \tilde{A}(S|\theta); \mathbf{w})  \\
    &J(\alpha) = -\alpha (\log\pi(A|S) + \mathcal{H})\\
\end{align*}
$$
For each gradient step:
$$
\begin{align*}
    &A \sim \pi(\cdot| S; \theta)\\
    &\text{Take action } A \text{ observe } S', R\\
    &\mathbf{w}_i \leftarrow \mathbf{w}_i - \lambda^\mathbf{w_i} \nabla_\mathbf{w_i}J_Q(\mathbf{w}_i)  & \text{ for } i = 1, 2\\
    &\theta \leftarrow \theta - \lambda^\theta \nabla_\theta J_\pi(\theta)\\
    & \alpha \leftarrow \alpha - \lambda^\alpha \nabla_\alpha J(\alpha)\\
    &\mathbf{w}_{targ, i} \leftarrow \rho \mathbf{w}_{targ, i} + (1 - \rho)\mathbf{w}_{i}, & \text{ for } i = 1, 2\\
\end{align*}
$$
where:

- $y(R,S', d)$: Critic target.
- $\mathbf{w}$: Critic parameters (Two critics for clipped double Q, plus two target critics).
- $\theta$: Actor parameters.
- $\alpha$: Entropy temperature. 
- $\tilde{A}(S|\theta)$: a sample from $\pi(\cdot | S; \theta)$ which is differentiable w.r.t. $\theta$ via the reparameterization trick:

$$
\begin{align*}
    &\tilde{A}(S|\theta) = \tanh (\mu_\theta(S) + \sigma(S) \odot \xi), &\xi \sim N(0, I)
\end{align*}
$$

- $\mathcal{H}$: the entropy target, often taken as $-\text{dim} (\mathcal{A})$, where $\mathcal{A}$ is the action space.

#### For an $m$-agent problem, we change this to:
**Intuition:**
Actor-Critic:
$$
\begin{align*}
    &\delta \leftarrow V(S) - (R + \gamma V(S')),  & \text{(for gradient descend instead of ascend)}\\
    &\delta_i \leftarrow V_{i}(S, A_1, \ldots, A_{i-1}) - (R + \gamma V_{i+1}(S, A_1, \ldots, A_{i}))\\
    &\delta_m \leftarrow V_{m}(S, A_1, \ldots, A_{m-1}) - (R(S, A_1, \ldots, A_m) + \gamma V_1(S'))
\end{align*}
$$

Simplified version of Soft Actor-Critic:

$$
\begin{align*}
    & H = \alpha \log \pi (A|S)\\
    & J_Q = Q(S, A) - (R + \gamma(Q(S', A') - H))\\
    & J_\pi =  \alpha \log \pi(\tilde{A}(S|\theta) | S) - Q (S, \tilde{A}(S|\theta)) 
\end{align*}
$$

Therefore:

$$
\begin{align*}
    & H_i = \alpha_i \log \pi_i (A_i|S)\\
    & J_{Q_i} = Q_{i}(S, A_1, \ldots, A_{i}) - (R + \gamma(Q_{i+1}(S, A_1, \ldots, A_{i+1}) - H_{i+1}))\\
    & J_{Q_m} = Q_{m}(S, A_1, \ldots, A_{m}) - (R(S, A_1, \ldots, A_m) + \gamma(Q_1(S', A'_1) - H_1))\\
    & J_{\pi_i} =  \alpha_i \log \pi_i(\tilde{A_i}(S|\theta) | S) - Q_i(S,A_1, \ldots, A_{i-1}, \tilde{A}(S|\theta)) 
\end{align*}
$$
**Therefore:**

For each stage $i$ in the sequential $Q$-calculation, from $i = 1, \ldots,m$, we have a set of two ($k=1,2$) $Q$-functions to perform clipped double $Q$-learning.

$$
\begin{align*}
    &y_i(R_i, S, d) = R_i + \gamma(1 - d) \left ( \min_{k=1,2}Q^{targ}_{i,k}(S, A_1, \ldots, A_{i}; \mathbf{w}^{targ}_{i,k}) - \alpha_i \log \pi_i (A_i|S, A_1, \ldots, A_{i-1}; \theta_i) \right)\\
    &y_m(R, S', d) = R(S, A_1, ..., A_m) + \gamma(1 - d) \left ( \min_{k=1,2}Q^{targ}_{1,k}(S', A'_1; \mathbf{w}^{targ}_{1,k}) - \alpha_1 \log \pi_1 (A'_1|S'; \theta_1) \right), & A'_i \sim \pi_i(\cdot| S', A'_1, \ldots, A'_{i-1})\\
    &J_{Q_i} (\mathbf{w}_i) = \frac{1}{2}\sum_{k=1,2}(Q_{i, k}(S, A_1, \ldots, A_{i}; \mathbf{w}_{i, k}) - y_{i+1}(R_i, S, d))^2 \\
    &J_{Q_m} (\mathbf{w}_m) = \frac{1}{2}\sum_{k=1,2}(Q_{m, k}(S, A_1, \ldots, A_m; \mathbf{w}_m) - y_{m}(R, S', d))^2\\
    &J_{\pi_i}(\theta_i) =  \alpha_i \log \pi_i(\tilde{A}(S|\theta) | S, A_1, \ldots, A_{i-1}; \theta) - \min_{k=1,2} Q_{i,k} (S, A_1, \ldots, A_{i-1}, \tilde{A}_i(S|\theta); \mathbf{w})  \\
    &J(\alpha_i) = -\alpha_i (\log\pi_i(A_i|S, A_1, \ldots, A_{i-1}) + \mathcal{H_i})\\
\end{align*}
$$

where:

- $y$: Critic target.
- $\mathbf{w}$: Critic parameters (Two critics for clipped double Q, plus two target critics).
- $\theta$: Actor parameters.
- $\alpha$: Entropy temperature.
- $\tilde{A_i}(S|\theta_i)$: a sample from $\pi_i(\cdot | S; \theta_i)$ which is differentiable w.r.t. $\theta_i$ via the reparameterization trick:

For each gradient step:
$$
\begin{align*}
    &A_i \sim \pi_i(\cdot | S, A_1, \ldots, A_{i-1};\theta_) & \text{ for } i = 1, \ldots, m-1\\
    &\text{Take actions } A_1,\ldots,A_m \text{ observe } S', R\\
    &A'_i \sim \pi_i(\cdot | S', A'_1, \ldots, A'_{i-1};\theta_i) & \text{ for } i = 1, \ldots, m-1\\
    &\mathbf{w}_{i,k} \leftarrow \mathbf{w}_{i,k} - \lambda^{\mathbf{w}_{i,k}} \nabla_{\mathbf{w}_{i,k}} J_{Q_i}(\mathbf{w}_{i,k})  & \text{ for } k = 1, 2, \text{ and } i = 1, \ldots, m\\
    &\theta_i \leftarrow \theta_i - \lambda^{\theta_i} \nabla_{\theta_i} J_{\pi_i}(\theta_i) & \text{ for } i = 1, \ldots, m\\
    & \alpha_i \leftarrow \alpha_i - \lambda{^\alpha_i} \nabla_{\alpha_i} J(\alpha_i) & \text{ for } i = 1, \ldots, m\\
    &\mathbf{w}^{targ}_{i,k} \leftarrow \rho \mathbf{w}^{targ}_{i,k} + (1 - \rho)\mathbf{w}_{i,k}, & \text{ for } k = 1, 2, \text{ and } i = 1, \ldots, m\\
\end{align*}
$$