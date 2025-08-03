# Optimizing Day-Ahead Energy Trading with Proximal Policy Optimization and Blockchain

This repository contains code for training a reinforcement learning agent using Stable-Baselines3 with a dynamic curriculum learning schedule. The focus is on optimizing energy market performance, especially in scenarios involving imbalance penalties and dynamic settlement costs. The system also integrates the Algorand blockchain SDK for transaction handling and record keeping.

---

## Overview

This project explores the integration of Reinforcement Learning (RL) and Blockchain technology as a foundational approach for building intelligent, decentralized, and secure energy trading systems. RL provides a data-driven framework capable of learning optimal control policies through interaction with a dynamic environment, making it highly suitable for energy systems that involve variable renewables, shifting demand patterns, and uncertain market conditions.

---

##  Installation

### Python Version
Ensure you are using **Python 3.12.0**.

### Install dependencies
For CPU-only training:
pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu  
pip install -r requirements.txt
