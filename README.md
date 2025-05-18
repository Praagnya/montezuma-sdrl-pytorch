# Montezuma's Revenge - Symbolic Deep Reinforcement Learning (SDRL)

This repository contains a PyTorch-based implementation of **Symbolic Deep Reinforcement Learning (SDRL)** for the Atari game **Montezuma's Revenge**. SDRL combines classical symbolic planning with hierarchical reinforcement learning to address environments with sparse rewards.

## 📁 Project Structure

<code>
montezuma-sdrl-pytorch/
├── agents/                                     # Implementation of SDRL agents
├── environments/                               # Environment wrappers and utilities
├── skills/                                     # Learned skills and option policies
├── utils/                                      # Helper functions and common utilities
├── hybrid_asp_dqn_training_more_dqns_new.py    # Entry point for training and evaluation
├── config.yaml                                 # Configuration file for hyperparameters
├── requirements.txt                            # Dependency list
└── README.md                                   # Project documentation
</code>
</pre>

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Praagnya/montezuma-sdrl-pytorch.git
   cd montezuma-sdrl-pytorch

2.	**Create and Activate a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

3.	**Install Python Dependencies**
    ```bash
    pip install -r requirements.txt

4.  **Install Clingo (for symbolic planning)**
    ```bash
    brew install clingo       # macOS
    sudo apt install clingo   # Ubuntu/Debian

## 🚀 Running the Agent

    ```bash
    python hybrid_asp_dqn_training_more_dqns_new.py

## ⚡️ Key Features

- **Symbolic planner integration using Clingo**  
  Leverages ASP (Answer Set Programming) for interpretable goal planning.

- **Hierarchical DQN agent with skill discovery**  
  Breaks complex tasks into manageable subtasks using options.

- **Modular design for flexibility and reproducibility**  
  Easily configure, debug, and extend the codebase for new environments or planners.

- **Supports goal decomposition and state abstraction**  
  Enables reasoning over symbolic subgoals and abstract state representations.

## 🧾 Requirements

	•	Python 3.8+
	•	PyTorch ≥ 1.10
	•	Clingo ≥ 5.5

## 📄 License

This project is licensed under the MIT License.


