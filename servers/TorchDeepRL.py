import components.builders as builders

class TorchDeepRL():
    def __init__(self, config) -> None:
        self.learner = builders.build_learner(config)
    

    def step(self, state, reward, terminal):
        return self.learner.step(state, reward, terminal)
    

    def experience_and_optimize(self, state, reward, terminal):
        self.learner.experience(state, reward, terminal)
        self.learner.optimize()