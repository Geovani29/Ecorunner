import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = [self.get_q(state, a) for a in self.actions]
            return self.actions[np.argmax(q_values)]

    def learn(self, state, action, reward, next_state):
        q_current = self.get_q(state, action)
        q_max_next = max([self.get_q(next_state, a) for a in self.actions])
        self.q_table[(state, action)] = q_current + self.alpha * (reward + self.gamma * q_max_next - q_current)

        # 🔁 Disminuir epsilon gradualmente (hasta un mínimo)
        self.epsilon = max(0.05, self.epsilon * 0.995)


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
