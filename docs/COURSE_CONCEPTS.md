# Course Concepts in This Project

Mapping of **2025 course** (Optimization, PyTorch, SGD, Adam) to this repo.

---

## Table of Contents → Code

| Course Topic | Where in This Repo |
|--------------|--------------------|
| **0. Introduction** (PyTorch, spam classification) | PyTorch policy network; IPO “context” = features; “labels” = participate/entry/exit/size. |
| **1. Optimization and PyTorch Basics in 1D** | We optimize a scalar objective (risk-adjusted score) w.r.t. policy parameters; `policy_network.py` uses `nn.Module` and autodiff. |
| **2. Stochastic Optimization Basics in 1D** | REINFORCE uses stochastic rewards (minibatches of episodes); step-size and validation in `train_policy.py`. |
| **3. Linear Regression: Gradient Descent** | Feature extraction in `features.py` (linear-ish inputs); policy MLP is a nonlinear regression from features to action logits. |
| **4. How to compute gradients in PyTorch** | `policy_network.py`: `nn.Module`, `forward()`, `loss.backward()`, `optimizer.step()` in `train_policy.py`. |
| **5. How to think about derivatives (best linear approximation)** | Features in `features.py` are chosen so the policy can approximate a “best linear” (or nonlinear) map from state to action. |
| **6. Stochastic gradient descent: first look** | `train_policy.py`: minibatches of episodes, per-batch reward, REINFORCE gradient estimate = stochastic gradient. |
| **7. SGD: Noisy Quadratic Model (momentum, preconditioning)** | Optional: we use Adam (adaptive step) and gradient clipping instead of raw SGD; can add momentum/SGD later. |
| **8. SGD: general problem and implementation** | `train_reinforce()`: empirical risk = mean negative reward; minibatch loop; PyTorch `optimizer.zero_grad()`, `backward()`, `step()`. |
| **9. Adaptive Optimization (Adagrad, Adam, AdamW)** | `train_policy.py`: `torch.optim.Adam(policy.parameters(), lr=lr)`. |
| **10. Benchmarking Optimizers** | Compare random search (`run_week3.py`) vs PyTorch/Adam (`run_pytorch.py`) via same `score()` and validation. |
| **11. Playbook for Tuning Deep Learning Models** | `run_pytorch.py`: `--lr`, `--lr_schedule`, `--batch_size`, `--n_epochs`; validation set and train/val score logging. |
| **12. Scaling Transformers** | Not used (single GPU / CPU policy); batching in `train_policy.py` is the only “scale” we use. |
| **Recap** | This doc + `WEEK3_DELIVERABLE.md` tie the project back to objectives, constraints, and course topics. |

---

## Concepts in Short

- **Autodiff (Lectures 1, 4, 5)**  
  Policy outputs logits → we compute log-prob of actions → `loss = -(log_prob * advantage)`. `loss.backward()` gives gradients for the policy parameters.

- **Empirical risk / SGD (Lectures 2, 6, 8)**  
  “Risk” = negative mean reward over episodes. We optimize it with minibatches of episodes and a stochastic policy (REINFORCE), i.e. stochastic gradient updates.

- **Step-size and schedules (Lectures 1, 2, 8, 11)**  
  `train_policy.py` supports `lr_schedule`: `"constant"`, `"cosine"`, `"step"`. You can extend with more schedules from the course.

- **Validation (Lectures 2, 8, 11)**  
  We hold out `val_frac` of episodes and log `val_score` every epoch to detect overfitting and tune hyperparameters.

- **Adam (Lecture 9)**  
  Default optimizer in `train_policy.py` is `torch.optim.Adam`.

- **Tuning playbook (Lecture 11)**  
  Use `run_pytorch.py` flags (`--lr`, `--batch_size`, `--n_epochs`, `--lr_schedule`) and validation curves to tune the policy.

---

## File Guide

| File | Course Concepts |
|------|------------------|
| `src/features.py` | Feature extraction (linear/context); input to “regression” (policy). |
| `src/policy_network.py` | PyTorch `nn.Module`, `forward()`, autodiff, sampling (REINFORCE). |
| `src/train_policy.py` | SGD/Adam, minibatches, step-size schedule, validation, gradient clipping. |
| `run_pytorch.py` | Entry point for training; CLI for lr, schedule, batch size (playbook). |
| `run_week3.py` | Black-box (random) optimization; compare with PyTorch in benchmarking. |
