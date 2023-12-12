# Notes Week 48
---
### Dependence Diagram:
A dependency graph is a data structure formed by a directed graph that describes the dependency of an entity in the system on the other entities of the same system. The underlying structure of a dependency graph is a directed graph where each node points to the node on which it depends.

In our project we will be looking at the score/reward function variable interdependence. 

---
### Episode Based Score/Reward Function
The reward function as given is:
$$
\begin{equation}
    \begin{split}
        Score_{Control} = 0.3 \cdot Score_{Control}^{Comfort} + 0.1 \cdot Score_{Control}^{Emissions} + 0.3 \cdot Score_{Control}^{Grid} + 0.3 \cdot Score^{Resilience}_{Control},
    \end{split}
\end{equation}
$$

where:

$$
\begin{equation}
    Score_{Control}^{Comfort} = U,
\end{equation}
$$
$$
\begin{equation}
    Score_{Control}^{Emissions} = G,
\end{equation}
$$
$$
\begin{equation}
    Score_{Control}^{Grid} = \overline{R, L, P_{d}, P_{n}},
\end{equation}
$$
$$
\begin{equation}
    Score_{Control}^{Resilience} = \overline{M, S},
\end{equation}
$$

where these 4 scores are made up of 8 key performance indicators (KPIs): carbon emissions (G), discomfort (U), ramping (R), 1 - load factor (L), daily peak (P_d), all-time peak (P_n) 1 - thermal resilience (M), and normalized unserved energy (S). 

The grid control and resilience control scores are averages over their KPIs. 

The KPIs are calculated as follows: 

---

**G: Carbon emissions -> the emissions from imported electricity:**
$$
\begin{equation}
    G = \sum_{i=0}^{b-1}g^{i}_{control} \div \sum_{i=0}^{b-1}g^{i}_{baseline},
\end{equation}
$$
where:
$$
\begin{equation}
    g = \sum^{n-1}_{t=0} max (0, e_{t} \cdot B_t),
\end{equation}
$$
where:
$e_{t}$ = building level net electricity consumption,
$B_{t}$ = Emission rate.

---

**U: Unmet hours -> proportion of time steps when a building is occupied and indoor temperature falls outside a comfort band**

$$
\begin{equation}
    U = \sum_{i=0}^{b-1}u^{i}_{control} \div b,
\end{equation}
$$
where:
$$
\begin{equation}
    u = a \div o,
\end{equation}
$$
where:

$$
\begin{equation}
a = \sum^{n-1}_{t=0}\left\{\begin{matrix}
1\text{, if }|T_{t} - T_{t}^{setpoint}| > T^{band}\text{ and }O_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
where:
$$
\begin{equation}
o = \sum^{n-1}_{t=0}\left\{\begin{matrix}
1\text{, if } O_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$

where:
$T_{t}$ = Indoor dry-bulb temperature (oC),
$T_{t}^{setpoint}$ =  Indoor dry-bulb temperature setpoint (oC)  (Desired temperature for thermal comfort), 
$T^{band}$ = Indoor dry-bulb temperature comfort band (±$T^{setpoint}$),
$O_t$ = Occupant count,
$b$ = Total number of buildings.

---

**R: Ramping -> Smoothness of the district’s consumption profile where low R means there is gradual increase in consumption even after self-generation is unavailable in the evening and early morning. High R means abrupt change in grid load that may lead to unscheduled strain on grid infrastructure and blackouts caused by supply deficit.**

$$
\begin{equation}
    R = r_{control} \div r_{baseline},
\end{equation}
$$
where:
$$
\begin{equation}
    r = \sum^{n-1}_{t=0} |E_t - E_{t-1}|,
\end{equation}
$$
where:
$E$ = Neighborhood-level net electricity consumption (kWh)

---

**L: 1 - Load factor -> Average ratio of daily average and peak consumption. Load factor is the efficiency of electricity consumption and is bounded between 0 (very inefficient) and 1 (highly efficient) thus, the goal is to maximize the load factor or minimize (1 − load factor).**

$$
\begin{equation}
    L = l_{control} \div l_{baseline},
\end{equation}
$$
where:
$$
\begin{equation}
    l = \left ( \sum_{d=0}^{n\div h} 1 - \frac{(\sum^{d \cdot h + h - 1}_{t = d \cdot h} E_{t})\div h}{max(E_{d \cdot h}, ..., E_{d \cdot h + h - 1})} \right ) \div \left ( \frac{n}{h} \right ),
\end{equation}
$$
where:
$E$ = Neighborhood-level net electricity consumption
$n$ = Total number of time steps
$d$ = Day
$h$ = Hours per day

---

**P_d: Daily peak -> Average, maximum consumption at any time step per day.**

NOTE: ESCALATED REWARD

$$
\begin{equation}
    P_{d} = p_{d_{control}} \div p_{d_{baseline}},
\end{equation}
$$
where:
$$
\begin{equation}
    p_{d} = \left ( \sum^{n \div h}_{d = 0} max (E_{d \cdot h},..., E_{d \cdot h + h - 1}) \right ) \div \left ( \frac{n}{h} \right ),
\end{equation}
$$
where: 
$E$ = Neighborhood-level net electricity consumption,
$n$ = Total number of time steps,
$d$ = Day,
$h$ = Hours per day.

---

**P_n : All-time peak -> Maximum consumption at any time step.**

$$
\begin{equation}
    P_n = p_{n_{control}} \div p_{n_{baseline}},
\end{equation}
$$
where:
$$
\begin{equation}
    p_n = max(E_0, ... , E_n),
\end{equation}
$$
where: 
$E$ = Neighborhood-level net electricity consumption,
$n$ = Total number of time steps.

---

**M: 1 - thermal resilience -> Same as unmet hours (thermal comfort) U but only considers time steps when there is power outage.**

$$
\begin{equation}
    M = \sum^{b-1}_{i=0} m^{i}_{control} \div b,
\end{equation}
$$
where:
$$
\begin{equation}
    m = a \div o,
\end{equation}
$$
where:
$$
\begin{equation}
    a = \sum^{n-1}_{t=0}\left\{\begin{matrix}
1\text{, if } |T_t - T_{t}^{setpoint}| > T^{band}\text{ and }O_t > 0\text{ and }F_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
where:
$$
\begin{equation}
    a = \sum^{n-1}_{t=0}\left\{\begin{matrix}
1\text{, if } O_t > 0\text{ and }F_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
where:
$T_{t}$ = Indoor dry-bulb temperature (oC),
$T_{t}^{setpoint}$ =  Indoor dry-bulb temperature setpoint,(oC)  (Desired temperature for thermal comfort),
$T^{band}$ = Indoor dry-bulb temperature comfort band (±$T^{setpoint}$),
$O_t$ = Occupant count,
$F$ = Power outage signal (Yes/No),
$b$ = Total number of buildings.

---

**S: Normalized unserved energy	-> Proportion of unmet demand due to supply shortage e.g. power outage.**

$$
\begin{equation}
    S = \sum_{i=0}^{b-1}s_{control}^{i} \div b,
\end{equation}
$$
where:
$$
\begin{equation}
    s = s^{served} \div s^{expected},
\end{equation}
$$
where:
$$
\begin{equation}
s^{served} = \sum_{t=0}^{n-1}\left\{\begin{matrix}
q_n^{served}\text{, if } F_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
$$
\begin{equation}
s^{expected} = \sum_{t=0}^{n-1}\left\{\begin{matrix}
q_n^{expected}\text{, if } F_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
where:
$q$ = Building-level cooling, domestic hot water and non-shiftable load energy demand (kWh),
$n$ = Total number of time steps,
$b$ = Total number of buildings.

---
# Step-Reward Conversion
We convert the cummulative episode-wise score function to a step-wise reward function.


The reward function as given is:
$$
\begin{equation}
    \begin{split}
        Score_{Control} = 0.3 \cdot Score_{Control}^{Comfort} + 0.1 \cdot Score_{Control}^{Emissions} + 0.3 \cdot Score_{Control}^{Grid} + 0.3 \cdot Score^{Resilience}_{Control},
    \end{split}
\end{equation}
$$

We convert it to:
$$
\begin{equation}
    \begin{split}
        Reward_t = 0.3 \cdot Comfort_t + 0.1 \cdot Emissions_t + 0.3 \cdot Grid_t + 0.3 \cdot Resilience_t,
    \end{split}
\end{equation}
$$


where:

$$
\begin{equation}
    Comfort_t = U_t,
\end{equation}
$$
$$
\begin{equation}
    Emissions_t = G_t,
\end{equation}
$$
$$
\begin{equation}
    Grid_t = \overline{R_t, L_t, Pd_t, Pn_t},
\end{equation}
$$
$$
\begin{equation}
    Resilience_t = \overline{M_t, S_t},
\end{equation}
$$

NOTE FROM MEETINGl TRANSITION INDEPENDENT DECENTRALIZED MDP

where these 4 reward components are made up of 8 key performance indicators (KPIs): carbon emissions (G), discomfort (U), ramping (R), 1 - load factor (L), daily peak (Pd), all-time peak (Pn) 1 - thermal resilience (M), and normalized unserved energy (S). 

The grid and resilience reward components are averages over their KPIs. 

The KPIs are calculated as follows: 

---

**G: Carbon emissions -> the emissions from imported electricity:**
$$
\begin{equation}
    G_t = \sum_{i=0}^{b-1}g^{i}_t \div \sum_{i=0}^{b-1}g^{i}_{baseline_t},
\end{equation}
$$
where:
$$
\begin{equation}
    g_t = max (0, e_{t} \cdot B_t),
\end{equation}
$$
where:
$e_{t}$ = building level net electricity consumption,
$B_{t}$ = Emission rate.

---

**U: Unmet hours -> proportion of time steps when a building is occupied and indoor temperature falls outside a comfort band**

$$
\begin{equation}
    U_t = \left (\sum_{i=0}^{b-1}u^{i}_t \right ) \div b,
\end{equation}
$$
where:
$$
\begin{equation}
    u_t = a_t \div o_t,
\end{equation}
$$
where:

$$
\begin{equation}
a_t = \left\{\begin{matrix}
1\text{, if }|T_{t} - T_{t}^{setpoint}| > T^{band}\text{ and }O_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
where:
$$
\begin{equation}
o_t = \left\{\begin{matrix}
1\text{, if } O_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$

where:
$T_{t}$ = Indoor dry-bulb temperature (oC),
$T_{t}^{setpoint}$ =  Indoor dry-bulb temperature setpoint (oC)  (Desired temperature for thermal comfort), 
$T^{band}$ = Indoor dry-bulb temperature comfort band (±$T^{setpoint}$),
$O_t$ = Occupant count,
$b$ = Total number of buildings.

---

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

---

**L: 1 - Load factor -> Average ratio of daily average and peak consumption. Load factor is the efficiency of electricity consumption and is bounded between 0 (very inefficient) and 1 (highly efficient) thus, the goal is to maximize the load factor or minimize (1 − load factor).**
**-> Ratio of daily average and peak consumption. Load factor is the efficiency of electricity consumption and is bounded between 0 (very inefficient) and 1 (highly efficient) thus, the goal is to maximize the load factor or minimize (1 − load factor).**
$$
\begin{equation}
    L = l_{t} \div l^{baseline}_t,
\end{equation}
$$
where **DUBBLE CHECK**:
$$
\begin{equation}
    l = \left (\sum_{d=0}^{n\div h} 1 - \frac{(\sum^{d \cdot h + h - 1}_{t = d \cdot h} E_{t})\div h}{max(E_{d \cdot h}, ..., E_{d \cdot h + h - 1})} \right ) \div \left ( \frac{n}{h} \right ),
\end{equation}
$$
TO ->
$$
\begin{equation}
    l_t = 1 - \frac{(\sum^{t}_{i = t - h} E_{i})\div h}{max(E_{t-h}, E_{t-h+1} ..., E_{t})}
\end{equation}
$$
where:
$E$ = Neighborhood-level net electricity consumption
$n$ = Total number of time steps
$d$ = Day
$h$ = Hours per day
$t$ = Current time step

---

**Pd: Daily peak -> maximum consumption at any time step of this day.**

$$
\begin{equation}
    Pd = pd_{t} \div pd_{t}^{baseline},
\end{equation}
$$
where **DOUBLE CHECK:**:
$$
\begin{equation}
    p_{d} = \left ( \sum^{n \div h}_{d = 0} max (E_{d \cdot h},..., E_{d \cdot h + h - 1}) \right ) \div \left ( \frac{n}{h} \right ),
\end{equation}
$$
TO ->
$$
\begin{equation}
    pd_t = max (E_{t-h}, E_{t - h + 1}..., E_{t}),
\end{equation}
$$

where: 
$E$ = Neighborhood-level net electricity consumption,
$n$ = Total number of time steps,
$d$ = Day,
$h$ = Hours per day.
$t$ = Current timestep

---

**P_n : All-time peak -> Maximum consumption at any time step.**

How will this be incoorporated step-wise?

$$
\begin{equation}
    Pn = p_{n} \div p_{n}^{baseline},
\end{equation}
$$

To?:

**-> Current consumption**
$$
\begin{equation}
    P_t = p_{t} \div p_{t}^{baseline},
\end{equation}
$$

$$
\begin{equation}
    P_t = E_t,
\end{equation}
$$

where: 
$E$ = Neighborhood-level net electricity consumption,
$n$ = Total number of time steps.
$t$ = Current time step

---

**M: 1 - thermal resilience -> Same as unmet hours (thermal comfort) U but only considers time steps when there is power outage.**

$$
\begin{equation}
    M_t = \left ( \sum^{b-1}_{i=0} m^{i}_{t} \right ) \div b,
\end{equation}
$$
where:
$$
\begin{equation}
    m_t = a_t \div o_t,
\end{equation}
$$
where:
$$
\begin{equation}
    a_t = \left\{\begin{matrix}
1\text{, if } |T_t - T_{t}^{setpoint}| > T^{band}\text{ and }O_t > 0\text{ and }F_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
where:
$$
\begin{equation}
    o_t = \left\{\begin{matrix}
1\text{, if } O_t > 0\text{ and }F_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
where:
$T_{t}$ = Indoor dry-bulb temperature (oC),
$T_{t}^{setpoint}$ =  Indoor dry-bulb temperature setpoint,(oC)  (Desired temperature for thermal comfort),
$T^{band}$ = Indoor dry-bulb temperature comfort band (±$T^{setpoint}$),
$O_t$ = Occupant count,
$F$ = Power outage signal (Yes/No),
$b$ = Total number of buildings.

---

**S: Normalized unserved energy	-> Proportion of unmet demand due to supply shortage e.g. power outage.**
$$
\begin{equation}
    S_t = \left ( \sum_{i=0}^{b-1}s_t^{i} \right ) \div b,
\end{equation}
$$
where:
$$
\begin{equation}
    s_t = s_t^{served} \div s_t^{expected},
\end{equation}
$$
where:
$$
\begin{equation}
s^{served}_t = \left\{\begin{matrix}
q_t^{served}\text{, if } F_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
$$
\begin{equation}
s^{expected}_t = \left\{\begin{matrix}
q_t^{expected}\text{, if } F_t > 0
\\ 
0\text{, else}
\end{matrix},\right.
\end{equation}
$$
where:
F = Power outage signal (Yes/No),
$q$ = Building-level cooling, domestic hot water and non-shiftable load energy demand (kWh),
$n$ = Total number of time steps,
$b$ = Total number of buildings.