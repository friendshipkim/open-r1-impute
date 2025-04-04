from sklearn.linear_model import LinearRegression

class RewardImputation:
    def __init__(self, start_patch):
        self.model = LinearRegression()
        self.start_patch = start_patch
    
    def train(self, policy_outputs, reward_outputs):
        breakpoint()
        self.model.fit(policy_outputs, reward_outputs)

    def impute(self, hidden_states):
        return self.model.predict(hidden_states)
