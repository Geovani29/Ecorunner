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

    def save(self, path):
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data.get('epsilon', self.epsilon)  # Usa el valor guardado o mantiene el actual
