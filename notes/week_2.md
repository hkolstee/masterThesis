# Week 2 Notes

---

#### Ramping revisited

**R: Ramping -> Smoothness of the district’s consumption profile where low R means there is gradual increase in consumption even after self-generation is unavailable in the evening and early morning. High R means abrupt change in grid load that may lead to unscheduled strain on grid infrastructure and blackouts caused by supply deficit.**

$$
\begin{equation}
    R = r_t \div r^{baseline}_t,
\end{equation}
$$
where:
$$
\begin{equation}
    r_t = |E_t - E_{t-1}|,
\end{equation}
$$
where:
$E$ = Neighborhood-level net electricity consumption (kWh)

#### Problem:
The baseline is calculated the same way as the agents daily peak. However, the baseline is calculated using value for electricity consumption when no energy storage systems are being used. Therefore, the electricity consumption of the agent might be a lot higher for some hours than the baseline, and a lot lower on other hours.

---

#### All-time peak revisited

**A: all-time peak**
Here, taking the same approach will result in sparse rewards. Rewards will only be given to the agents when a new max energy consumption is achieved over the entire episode. Perhaps because this is a minimization problem, the zero sparse reward still gives the agent enough information as it is the optimal reward. Also, whenever a reward is gained in the episode it hopefully tells the agents to watch their energy consumption.

$$
\begin{equation}
    A = A_{control} \div A_{baseline},
\end{equation}
$$
where:
$$
\begin{equation}
    A_t =
    \begin{cases}
        max(0, E_t), & \text{if } t = 0\\
        max(0, E_{0}, E_1,..., E_t) - A_{t-1} , & \text{otherwise}
    \end{cases}
\end{equation}
$$

#### Problem: 
The baseline consumption will only be larger than the last peak by chance that more electricity should be used for thermal comfort, rather than the actions of the non-baseline agent which can achieve a new peak consumption by charging energy storages. Therefore, we will only use the difference in peak consumption of the control agent. 