from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import numpy as np

class RewardImputation:
    def __init__(self, start_patch):
        self.model = LinearRegression()
        self.start_patch = start_patch
        self.trained = False
    
    def train(self, policy_outputs, reward_outputs):
        print(f"Training reward imputation model with {len(policy_outputs)} steps")
        assert len(policy_outputs) == len(reward_outputs)
        assert len(policy_outputs) == self.start_patch

        # policy_outputs: shape (start_patch, n_prompts, n_generations, policy_hidden_size)
        # reward_outputs: shape (start_patch, n_prompts, n_generations)
        policy_hidden_states = np.stack([output['hidden_states'] for output in policy_outputs])
        rewards = np.stack([output['reward_aggregated'] for output in reward_outputs])
        
        # flatten the hidden states and rewards
        policy_hidden_states = policy_hidden_states.reshape(-1, policy_hidden_states.shape[-1])
        rewards = rewards.reshape(-1, 1)
        
        # fit the model
        self.model.fit(policy_hidden_states, rewards)
        self.trained = True

    def impute(self, hidden_states):
        if not self.trained:
            raise ValueError("Reward imputation model not trained")
        predicted_rewards = self.model.predict(hidden_states)
        return predicted_rewards[:, 0]


class CorrImputation:
    def __init__(self, start_pre_patch, start_patch, alpha=0.1):
        self.model = Lasso(alpha)
        self.start_pre_patch = start_pre_patch 
        self.start_patch = start_patch
        self.trained = False

    # 
    def train(self, policy_outputs, reward_outputs):
        print(f"Training correlation prediction model with {len(policy_outputs)} steps")
        # assert len(policy_outputs) == len(reward_outputs)
        # assert len(policy_outputs) == self.start_patch

        # policy_outputs: shape (start_pre_patch, n_prompts, n_generations, policy_hidden_size)
        policy_hidden_states = np.stack([output['hidden_states'] for output in policy_outputs])
         # policy_outputs_avg: shape (start_pre_patch, n_prompts, policy_hidden_size)
        policy_hidden_states_avg = policy_hidden_states.mean(-2)
        
        # correlations: shape (start_pre_patch, n_prompts, 1)
        correlations = np.stack([output['correlation'] for output in reward_outputs])
        
        # flatten the hidden states and rewards
        policy_hidden_states_avg = policy_hidden_states_avg.reshape(-1, policy_hidden_states_avg.shape[-1])
        correlations = correlations.reshape(-1, 1)
        
        # fit the model
        self.model.fit(policy_hidden_states_avg, correlations)
        self.trained = True

    def impute(self, hidden_states):
        if not self.trained:
            raise ValueError("Reward imputation model not trained")
        hidden_states_avg = hidden_states_avg.mean(0)
        predicted_rewards = self.model.predict(hidden_states_avg)
        return predicted_rewards[:, 0]

