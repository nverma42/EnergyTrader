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

###  Project Structure
The file energy_trader.py python source file contains the main driver program.  
The file rl_model.py contains the code for training the reinforcement learning agent.    
The file smart_contract.py contains the code for creating Algorand smart contract.  
The data folder holds Algorand keys, mnemonics, and ERCOT data files.

### Approach
1. Get ERCOT hourly demand forecast and price forecast data as the input.  
2. Train reinforcement learning model.  
3. Distribute hourly demand among k consumers based imbalance least cost targets.  
4. Get the optimal allocations from the model.  
5. Create asset transactions to transfer energy asset from matcher to sellers.  
6. Create application call transaction to update the completed status in the smart contract.  
7. Trigger payments from matcher to sellers. 
