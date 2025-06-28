

##  Learning Summary

### What I Knew Before Starting

This project builds on my prior experience with:

Statistical ML & Data Mining

  * Model selection, inference, and classification techniques
  * Algorithms like logistic regression, decision trees, and ensemble methods

Advanced ML

  * Deep learning (CNNs, RNNs), probabilistic models, and generative models like VAEs and GANs

Generative AI / LLMs

  * Transformers, decoding strategies, prompt engineering, and agentic AI pipelines

Reinforcement Learning (RL)

  * Core ideas like policy/value functions, exploration vs. exploitation, and reward feedback

---

### Improvements & Deeper Understanding

Through Objectives 1–3 and maze navigation, I reinforced and extended my knowledge of:

* Q-Learning : Implemented from scratch and tuned for stability and convergence
* RL Agent Design : Built agents to operate in grid-based environments like FrozenLake
* SNNs (Spiking Neural Networks) : Gained practical experience using `snntorch` and working with spike-based computation

---

###  What I Learned From Scratch

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

Debugging + Tooling

  * Resolved Gym API shifts (`reset()`/`step()` in new API)
  * Managed `torch` device mismatches (CPU vs CUDA)
  * Dealt with dependency conflicts and versioning (numpy, gym, dopamine)

---

### Summary

> I explored the intersection of **neuroscience-inspired learning** and **RL agents** — building agents that combine **spiking neuron dynamics** with **decision-making in interactive environments**.

This project gives me a strong foundation to pursue more complex research and applications in:

* Neuro-symbolic AI
* Embodied learning
* Brain-inspired RL systems
* Adaptive autonomous agents

---

