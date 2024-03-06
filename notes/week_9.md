# Week 9 Notes

---

### Literature review research:

- Variance in continuous policies is exaggerated in multi-agent settings.
- Environment more unstable in discrete (?)
- Big challenge in CTDE is how to represent the centralized action-value Q function
    - Simple: forgo centralised Q, learn individual Q function. However, this approach can not explicitly represent interactions between agents and may not converge
    - Another extreme: learn fully centralised Q function, and use it in actor critic framework (COMA). However, this requires on-policy learning, which is sample inefficient, and training the fully centralized critic becomes inpractical when there are more than a handful of agents. 
    - In between: Centralised but factored Q-function (VDN), by representing Qtot as a sum of individual value functions that condition only on individual observations and actions, a decentralised policy arises simply from each agent selecting actions greedily with respect to its own Q. However, VDN severely limits the complexity of the central Q funcs that can be represented and ignores any extra state info during training.
- Another challenge is the multi-agent credit assignment problem
     