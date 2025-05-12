import numpy as np
import random
import pickle
from rl.q_learning import QLearningAgent

ACTIONS = [0, 1, 2, 3]  # 0=nada, 1=subir, 2=bajar, 3=disparar

SCREEN_HEIGHT = 800
PLAYER_STEP = 60
TRASH_SPEED = 5

class SimEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_y = SCREEN_HEIGHT // 2
        self.trash_y = random.randint(0, SCREEN_HEIGHT)
        self.trash_x = 800

        # Obstacle spawns less often
        self.obstacle_y = random.randint(0, SCREEN_HEIGHT)
        self.obstacle_x = random.choice([900, 1100, 1300])

        self.steps_alive = 0
        self.last_y = self.player_y
        self.same_y_count = 0

        return self.get_state()

    def get_state(self):
        return (
            self.player_y // 60,
            self.trash_y // 60,
            self.trash_x // 100,
            self.obstacle_y // 60,
            self.obstacle_x // 100
        )

    def step(self, action):
        if action == 1:
            self.player_y -= PLAYER_STEP
        elif action == 2:
            self.player_y += PLAYER_STEP

        self.player_y = max(0, min(SCREEN_HEIGHT, self.player_y))
        self.trash_x -= TRASH_SPEED
        self.obstacle_x -= TRASH_SPEED
        self.steps_alive += 1

        # Detect inactividad vertical
        if self.player_y == self.last_y:
            self.same_y_count += 1
        else:
            self.same_y_count = 0
        self.last_y = self.player_y

        reward = 0
        done = False

        # Recompensa por disparar bien
        if action == 3 and abs(self.player_y - self.trash_y) <= 40 and self.trash_x < 120:
            reward = 20
            done = True

        # Penalización por choque
        elif abs(self.player_y - self.obstacle_y) <= 40 and self.obstacle_x < 50:
            reward = -30
            done = True

        # Penalización por dejar pasar basura
        elif self.trash_x < 0:
            reward = -10
            done = True

        # Penalizar inactividad prolongada
        elif self.same_y_count >= 5:
            reward -= 5

        # Recompensa por supervivencia cada 5 pasos
        if self.steps_alive % 5 == 0:
            reward += 1

        return self.get_state(), reward, done

if __name__ == "__main__":
    agent = QLearningAgent(actions=ACTIONS, alpha=0.1, gamma=0.9, epsilon=0.2)
    env = SimEnvironment()
    episodes = 40000

    for ep in range(episodes):
        state = env.reset()
        for step in range(300):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            if done:
                break

    agent.save("data/q_table.pkl")
    print("IA entrenada y q_table.pkl guardado.")

class RealTimeIAController:
    def __init__(self):
        self.agent = QLearningAgent(actions=ACTIONS)
        self.agent.load("data/q_table.pkl")

    def get_state(self, player_y, trash_y, trash_x, obstacle_y, obstacle_x):
        return (
            player_y // 60,
            trash_y // 60,
            trash_x // 100,
            obstacle_y // 60,
            obstacle_x // 100
        )

    def decide(self, player_y, trash_list, obstacle_list):
        if not trash_list or not obstacle_list:
            return 0

        nearest_trash = min(trash_list, key=lambda t: t.x)
        nearest_obs = min(obstacle_list, key=lambda o: o.x)

        state = self.get_state(
            player_y,
            nearest_trash.y,
            nearest_trash.x,
            nearest_obs.y,
            nearest_obs.x
        )
        return self.agent.choose_action(state)