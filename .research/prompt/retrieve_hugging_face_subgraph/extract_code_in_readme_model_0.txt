
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
env_name: LunarLander-v3
tags:
- LunarLander-v3
- dueling dqn
- reinforcement-learning
- custom-implementation
- deep-q-learning
- pytorch
model-index:
- name: DuelingDQN-1d-LunarLander-v3
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: LunarLander-v3
      type: LunarLander-v3
    metrics:
    - type: mean_reward
      value: 274.78 +/- 16.60
      name: mean_reward
      verified: false
---

# **Dueling DQN** Agent playing **LunarLander-v3**
This is a trained model of a **Dueling DQN** agent playing **LunarLander-v3**.

## Usage
### create the conda env in https://github.com/GeneHit/drl_practice
```bash
conda create -n drl python=3.10
conda activate drl
python -m pip install -r requirements.txt
```

### play with full model
```python
# load the full model
model = load_from_hub(repo_id="winkin119/DuelingDQN-1d-LunarLander-v3", filename="full_model.pt")

# Create the environment. 
env = gym.make("LunarLander-v3")
state, _ = env.reset()
action = model.action(state)
...
```
There is also a state dict version of the model, you can check the corresponding chapter in the repo.

Output:
{
    "extracted_code": "# load the full model\nmodel = load_from_hub(repo_id=\"winkin119/DuelingDQN-1d-LunarLander-v3\", filename=\"full_model.pt\")\n\n# Create the environment.\nenv = gym.make(\"LunarLander-v3\")\nstate, _ = env.reset()\naction = model.action(state)\n..."
}
