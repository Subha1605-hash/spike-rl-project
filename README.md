## Learning Summary

### What I Knew Before Starting

This project builds on my prior experience with:

Statistical ML & Data Mining

* Model selection, inference, and classification techniques
* Algorithms like logistic regression, decision trees, and ensemble methods

Advanced ML

* Deep learning (CNNs, RNNs), probabilistic models, and generative models like VAEs and GANs

Generative AI / LLMs

* Transformers, decoding strategies, and agentic AI pipelines

Reinforcement Learning (RL)

* Core ideas like policy/value functions, exploration vs. exploitation, and reward feedback

---

### Improvements & Deeper Understanding

Through Objectives 1–3 and maze navigation, I reinforced and extended my knowledge of:

* Q-Learning : Implemented from scratch and tuned for stability and convergence
* RL Agent Design : Built agents to operate in grid-based environments like FrozenLake
* SNNs (Spiking Neural Networks) : Gained practical experience using `snntorch` and working with spike-based computation
* Policy Gradient Methods : Implemented REINFORCE algorithm from scratch for CartPole-v1; understood the use of stochastic policies and gradient ascent in RL.

---

### What I Learned From Scratch

Spiking Neural Networks (SNNs)

* Concepts of membrane potential, thresholds, and time-dependent spiking
* Built and trained SNNs for binary classification (MNIST: digits 0 vs 1)

STDP (Spike-Timing-Dependent Plasticity)

* Implemented biologically plausible learning rules
* Combined STDP with reinforcement signals (reward-modulated STDP)

Maze Navigation with SNN Agents

* Created a simple custom maze environment
* Trained an SNN agent using feedback from environment
* Visualized step-by-step movement with `matplotlib`

Meta-Learning (Objectives 1–3)

* Objective 1: Introduced to meta-learning paradigms and applications

  * Meta-learning overview and real-world use cases
  * Few-shot learning with Hebbian-style updates
  * Explored cross-domain use cases (e.g., mobile OCR)
  * Key resources:

    * [https://www.comet.com/site/blog/meta-learning-in-machine-learning/](https://www.comet.com/site/blog/meta-learning-in-machine-learning/)
    * [https://www.geeksforgeeks.org/machine-learning/meta-learning-in-machine-learning/](https://www.geeksforgeeks.org/machine-learning/meta-learning-in-machine-learning/)
    * [https://www.ibm.com/think/topics/meta-learning](https://www.ibm.com/think/topics/meta-learning)
    * [https://rbcborealis.com/research-blogs/tutorial-2-few-shot-learning-and-meta-learning-i/](https://rbcborealis.com/research-blogs/tutorial-2-few-shot-learning-and-meta-learning-i/)
    * [http://anyline.com/news/cross-domain-few-shot-learning-mobile-ocr](http://anyline.com/news/cross-domain-few-shot-learning-mobile-ocr)
    * [https://proceedings.neurips.cc/paper/2020/file/ee23e7ad9b473ad072d57aaa9b2a5222-Paper.pdf](https://proceedings.neurips.cc/paper/2020/file/ee23e7ad9b473ad072d57aaa9b2a5222-Paper.pdf)

* Objective 2: Implemented Meta-Learning in Maze Navigation

  * Used the Maze Navigation GitHub repo ([https://github.com/JustinHeaton/Maze-Navigation](https://github.com/JustinHeaton/Maze-Navigation))
  * Built a MAML-based meta-learner for few-shot task adaptation
  * Resources studied:

    * [https://arxiv.org/pdf/1807.05076](https://arxiv.org/pdf/1807.05076)
    * [https://github.com/tristandeleu/pytorch-maml](https://github.com/tristandeleu/pytorch-maml)
    * [https://github.com/learnables/learn2learn/tree/master/examples](https://github.com/learnables/learn2learn/tree/master/examples)

* Objective 3: Compared with a Q-learning Baseline

  * Built a simple Q-learning agent to solve the same maze task
  * Resources used:

    * [https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/](https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/)
    * [https://www.youtube.com/watch?v=ZhoIgo3qqLU\&list=PL58zEckBH8fBW\_XLPtIPlQ-mkSNNx0tLS](https://www.youtube.com/watch?v=ZhoIgo3qqLU&list=PL58zEckBH8fBW_XLPtIPlQ-mkSNNx0tLS)
    * [https://github.com/simoninithomas/Deep\_reinforcement\_learning\_Course/tree/master/Q%20learning/FrozenLake](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Q%20learning/FrozenLake)

Policy Gradient Methods

* Learned the REINFORCE algorithm from scratch
* Built and trained an agent on CartPole-v1
* Understood the use of log-likelihoods, discounted rewards, and policy parameter updates
* References:

  * [https://www.geeksforgeeks.org/machine-learning/policy-gradient-methods-in-reinforcement-learning/](https://www.geeksforgeeks.org/machine-learning/policy-gradient-methods-in-reinforcement-learning/)
  * [https://en.wikipedia.org/wiki/Policy\_gradient\_method](https://en.wikipedia.org/wiki/Policy_gradient_method)
  * [https://youtu.be/e20EY4tFC\_Q](https://youtu.be/e20EY4tFC_Q)
  * [https://medium.com/intro-to-artificial-intelligence/reinforce-a-policy-gradient-based-reinforcement-learning-algorithm-84bde440c816](https://medium.com/intro-to-artificial-intelligence/reinforce-a-policy-gradient-based-reinforcement-learning-algorithm-84bde440c816)
  * [https://proceedings.neurips.cc/paper\_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
  * RL Course by David Silver - Lecture 7: Policy Gradient Methods

---

### Summary

> I explored the intersection of **neuroscience-inspired learning**, **meta-learning**, and **reinforcement learning**. I built agents that combine both **biologically plausible spiking mechanisms** and **adaptive learning across tasks**, covering techniques like Q-learning, MAML, and policy gradients.

This project gives me a strong foundation to pursue more complex research and applications in:

* Neuro-symbolic AI
* Embodied learning
* Brain-inspired RL systems
* Adaptive autonomous agents
* Meta-learners for real-world environments

