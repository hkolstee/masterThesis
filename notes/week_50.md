# Notes Week 50
---

### Paper: [Solving Transition Independent Decentralized Markov Decision Processes](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1207&context=cs_faculty_pubs)

**Abstract:** Formal treatment of collaborative multi-agent systems has been lagging behind the rapid progress in sequential decision making by individual agents. Recent work in the area of decentralized Markov Decision Processes (MDPs) has contributed to closing this gap, but the computational complexity of these models remains a serious obstacle. To overcome this complexity barrier, we identify a specific class of decentralized MDPs in which the agents’ transitions are independent. The class consists of independent collaborating agents that are tied together through a structured global reward function that depends on all of their histories of states and actions. We present a novel algorithm for solving this class of problems and examine its properties, both as an optimal algorithm and as an anytime algorithm. To the best of our knowledge, this is the first algorithm to optimally solve a non-trivial subclass of decentralized MDPs. It lays the foundation for further work in this area on both exact and approximate algorithms.

#### Summary:
##### Introduction
- The Multi-agent Markov Decision Process (**MMDP**) is a straightforward extension of the Markov Decision Process (**MDP**) to multiple agents by **factoring the action space** into actions for each of the agents. 
- **Centralized** approach is easier to solve optimally than a general **decentralized** approach. 
- Other researchers have focussed similarly to the authors on the **interaction between agents through the reward function**.
- Differently than the authors, these researchers use an observation model that resembles **full observability** of the MMDP rather than the **partial observability** of the authors model.
- Case of study: Situations where each agent has a **partial** and different view of the global state.
- MMDP assumption: every agent has the same **complete** world state view.
- **Zero cost communication** can satisfy this assumption.
- A paper (Xuan and Lesser (2002)) has agents only communicate when there is **ambiguity**.
- ***The problem the authors examine, is one where communication has a very high cost or is not possible***.
- For the general **DEC-POMDP**, **the only known optimal algorithm** is a new dynamic programming algorithm developed by Hansen, Bernstein, and Zilberstein (2004).
- The authors approach computes the **optimal policy** goal as part of an **optimal joint policy**.

Problem description: 

- The class of problems studied in this paper is characterized by two or more cooperative agents solving (mostly) independent local problems. **The actions taken by one agent can not affect any other agents' observation or local state**.
- ***An agent can not observe the other agents’ states and actions and can not communicate with them***
- **The interaction between the agents happens through a global value function that is not simply a sum of the values obtained through each of the agents’ local problems**
- The non-linear rewards combined with the decentralized view of the agents make the problem more difficult to solve than the MMDP, while the independence of the local problems make it easier to solve than the general DEC-MDP.
- This class of problem is called Transition Independent Dec-MDPs.
- Problems where the value of a single action performed by one agent may depend on the actions of other agents. 
- Here the example is given where complementary and redundant actions are possible in a two agent setting. The global *utility* function (quantifies the preference of agents action in a given state (not a value function as it is not involved with a policy)) is no longer additive over the agents. 