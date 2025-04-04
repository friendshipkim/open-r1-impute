from sklearn.linear_model import LinearRegression
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

