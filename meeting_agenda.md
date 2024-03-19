# Meeting 19/03

---

#### Transforming an m-agent infinite horizon problem into a stationary infinite horizon problem with fewer control/action choices available at each state

Transform:

$$
\begin{equation}
    \textbf{u} = \pi(x) = \{u_1, \ldots , u_m\}
\end{equation}
$$

To:

$$
\begin{equation}
    \textbf{u} = \{ \pi_0(x), \pi_1(x,u_1), \ldots , \pi_{m}(x, u_1, \ldots, u_{m-1}) \}
\end{equation}
$$

Sequentially calculated, such that $\pi_0(x) = u_1$, and $\pi_i (x, u_1, \ldots, u_{i-1}) = u_i$ for $i = 1, \ldots , m$.

**Q-learning:**

$$
\begin{equation}
    \textbf{u} = \{ \max_{u_1 \in U_1(x)} Q_1(x), \max_{u_2 \in U_2(x)} Q_2(x, u_1), \ldots , \max_{u_m \in U_m(x)} Q_{m}(x, u_1, \ldots , u_{m-1})  \}
\end{equation}
$$

Sequentially calculated, such that $max_{u_1 \in U_1(x)}Q_0(x) = u_1$, and $\max_{u_i \in U_i(x)}Q_i (x, u_1, \ldots, u_{i-1}) = u_i$ for $i = 1, \ldots, m$.

Lowers control choices compared to one policy that determines all actions at once.

**Assumptions (for online multiagent setting):**

1. All agents have access to the current state $x_k$
2. There is an order in which agents compute and apply their local controls.
3. There is *intercommunication* between the agents, so agent $l$ knows the local controls $u_1^k, \ldots , u_l-1^k$ computed by the predecessor agents $1, \ldots, l-1$ in the given order.

**Actor-Critic:**

All actions are given by Actors. Critics do not determine actions. Therefore we have to use sequalization in the form of Eq (2).

Using policies in the form of Eq (2) instead of:

$$
    \textbf{u} = \{ \pi_1(x), \pi_2(x), \ldots , \pi_m(x) \}
$$

Removes the possibility to execute the system in a decentralized manner.

**What I had implemented:**

- Centralized critic in the form: $Q(x, u_1, \ldots, u_m)$, where $u_1, \ldots, u_m$ is given by the actors $\pi_0(x), \ldots, \pi_m(x)$
- Centralized critics in the form $Q_1 , \ldots, Q_m$ in the same manner as the above.

**What I implemented (but could not test):**

- Sequential actors in the form: $\pi(x) = \{ \pi_0(x), \pi_1(x,u_1), \ldots , \pi_{m}(x, u_1, \ldots, u_{m-1}) \}$
- Sequential critic in the form: $Q_m(x, u_1, \ldots, u_m, Q_1, \ldots, Q_{m-1})$, where $Q_1(x, u_1, \ldots, u_m), Q_2(x, u_1, \ldots, u_m, Q_1), \ldots, Q_m(x, u_1, \ldots, u_m, Q_1, \ldots, Q_{m-1})$. However, this does not reduce control choices but only introduces a larger state space.
- Param sharing: One actor and one critic for multi-agent settings. Identity for now based on one-hot, but can be improved with latent encoder-decoder identity as done in paper (<https://arxiv.org/pdf/2102.07475.pdf>).

**Ideas:**

- Hierarchical control: Using $m$ policies $\pi^1_0(x), \ldots, \pi^1_m(x)$ as base, only conditioned on current state $x$, which approximate the actions $u_1, \ldots, u_m$. Then, following we have policies $\pi^2_1(x, u_1, \ldots, u_m), \ldots, \pi^2_m(x, u_1, \ldots, u_m)$ conditioned on both current state $x$ and approximated controls $u_1, \ldots, u_m$, that map to new approximate controls $u^2_1, \ldots, u^2_1$ using global coordination. Train the networks using normal loss function on $u^2_1, \ldots, u^2_1$, and train policies $\pi^1$ using intermediate MSELoss between $u^1$ and $u^2$. Decentralized execution via $\pi^1$. Seems a too simple approach to mapping/mixing actions, but I'm interested where the error is.
- 
  
<!-- - $Q(x, u_1, \ldots, u_m) = \{Q_1(x, u_1, \ldots, u_m), Q_1(x, u_1, \ldots, u_m, Q_1), \ldots, Q_m(x, u_1, \ldots, u_m, Q_1, \ldots, Q_{m-1}) \}$. However, this does not seem to do much rather than backprop through all Q-networks based on the final Q-value. -->
