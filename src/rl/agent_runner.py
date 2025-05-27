import os
import csv
import pickle
import random
from collections import defaultdict

try:
    from rl.q_learning import QLearningAgent  # Para cuando se ejecuta desde main.py
except ImportError:
    from q_learning import QLearningAgent

# ——— RUTAS Y DIRECTORIOS ———
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
QTABLE_PATH  = os.path.join(DATA_DIR, "q_table.pkl")

# ——— CONFIGURACIÓN ———
ACTIONS          = [0, 1, 2, 3]    # 0=nada, 1=subir, 2=bajar, 3=disparar
SCREEN_HEIGHT    = 800
PLAYER_STEP      = 60
TRASH_SPEED      = 5

TRAIN_EPISODES   = 80000
MAX_STEPS_PER_EP = 300

# Hiperparámetros RL
ALPHA            = 0.1    # tasa de aprendizaje
GAMMA            = 0.9    # factor de descuento
EPSILON_START    = 1.0    # prob. inicial de exploración
EPSILON_MIN      = 0.05   # prob. mínima de exploración
# Ajuste de decay para que epsilon caiga de 1.0 a 0.05 en TRAIN_EPISODES
DECAY_RATE       = (EPSILON_MIN / EPSILON_START) ** (1.0 / TRAIN_EPISODES)

# ——— ENTORNO SIMULADO ———
class SimEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_y     = SCREEN_HEIGHT // 2
        self.trash_y      = random.randint(0, SCREEN_HEIGHT)
        self.trash_x      = 800
        self.obstacle_y   = random.randint(0, SCREEN_HEIGHT)
        self.obstacle_x   = random.choice([900, 1100, 1300])
        self.steps_alive  = 0
        self.last_y       = self.player_y
        self.same_y_count = 0
        return self.get_state()

    def get_state(self):
        return (
            self.player_y   // PLAYER_STEP,
            self.trash_y    // PLAYER_STEP,
            self.trash_x    // 100,
            self.obstacle_y // PLAYER_STEP,
            self.obstacle_x // 100
        )

    def step(self, action):
        if action == 1:
            self.player_y -= PLAYER_STEP
        elif action == 2:
            self.player_y += PLAYER_STEP
        self.player_y = max(0, min(SCREEN_HEIGHT, self.player_y))

        self.trash_x    -= TRASH_SPEED
        self.obstacle_x -= TRASH_SPEED
        self.steps_alive += 1

        if self.player_y == self.last_y:
            self.same_y_count += 1
        else:
            self.same_y_count = 0
        self.last_y = self.player_y

        reward, done = 0, False
        if action == 3 and abs(self.player_y - self.trash_y) <= 40 and self.trash_x < 120:
            reward += 20
            self.trash_x = 800
            self.trash_y = random.randint(0, SCREEN_HEIGHT)
        elif abs(self.player_y - self.obstacle_y) <= 40 and self.obstacle_x < 50:
            reward, done = -30, True
        elif self.trash_x < 0:
            reward, done = -10, True
        elif self.same_y_count >= 5:
            reward -= 5

        if self.steps_alive % 5 == 0:
            reward += 1

        return self.get_state(), reward, done

# ——— ENTRENAMIENTO OFFLINE ———
def train():
    # Inicializar agente y cargar Q-Table previa si existe
    agent = QLearningAgent(actions=ACTIONS, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START)
    if os.path.exists(QTABLE_PATH):
        data = pickle.load(open(QTABLE_PATH, "rb"))
        agent.q_table = data.get("q_table", {})
        print("Resumiendo entrenamiento desde tabla previa…")

    env = SimEnvironment()

    # Logging
    metrics = []              # (ep, total_reward, steps_survived, success_flag, epsilon)
    state_counts = defaultdict(int)

    for ep in range(1, TRAIN_EPISODES + 1):
        state = env.reset()
        state_counts[state] += 1

        total_reward = 0
        success_flag = 0

        # Entrenamiento de un episodio
        for _ in range(MAX_STEPS_PER_EP):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)

            state_counts[next_state] += 1
            total_reward += reward
            if done and reward == 20:
                success_flag = 1

            state = next_state
            if done:
                break

        steps_survived = env.steps_alive

        # Decaimiento de epsilon
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * DECAY_RATE)

        # Evaluación pura cada 5000 episodios
        if ep % 5000 == 0:
            eps_backup = agent.epsilon
            agent.epsilon = 0.0
            eval_rewards = []
            for _ in range(100):
                s = env.reset()
                total_eval = 0
                while True:
                    a = agent.choose_action(s)
                    s, r, d = env.step(a)
                    total_eval += r
                    if d:
                        break
                eval_rewards.append(total_eval)
            avg_eval = sum(eval_rewards) / len(eval_rewards)
            print(f">>> Evaluación pura ep={ep}: recompensa media = {avg_eval:.2f}")
            agent.epsilon = eps_backup

        # Guardar métricas del episodio
        metrics.append((ep, total_reward, steps_survived, success_flag, agent.epsilon))

        if ep % 1000 == 0:
            last1000 = metrics[-1000:]
            avg_reward = sum(m[1] for m in last1000) / 1000
            print(f"Epi {ep}: recompensa media últimos 1k = {avg_reward:.2f}")

    # Guardar Q-Table (solo q_table)
    pickle.dump({"q_table": agent.q_table}, open(QTABLE_PATH, "wb"))
    print("Q-Table guardada en", QTABLE_PATH)

    data_dir = os.path.dirname(QTABLE_PATH)
    # 1) metrics.csv
    with open(os.path.join(data_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "total_reward", "steps_survived", "success_flag", "epsilon"])
        w.writerows(metrics)
    # 2) rewards.csv
    with open(os.path.join(data_dir, "rewards.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "total_reward"])
        w.writerows((m[0], m[1]) for m in metrics)
    # 3) state_counts.csv
    with open(os.path.join(data_dir, "state_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "count"])
        for st, cnt in state_counts.items():
            w.writerow([st, cnt])

    print("Metrics, rewards y state_counts guardados en", data_dir)

# ——— CONTROLADOR ONLINE ———
class OnlineQLearningController:
    def __init__(self):
        self.agent = QLearningAgent(
            actions=ACTIONS,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON_START
        )
        try:
            self.agent.load(QTABLE_PATH)
            print(f"[IA] Q-Table cargada. Epsilon guardado: {self.agent.epsilon:.3f}")
        except FileNotFoundError:
            print("Warning: Q-Table no encontrada, jugando sin cargar.")

        # Arranca en vivo con exploración moderada para seguir aprendiendo
        self.agent.epsilon = 0.10  # explora un 10 % cada partida
        self.last_state  = None
        self.last_action = None

    def get_state(self, player_y, trash, obs):
        return (
            player_y   // PLAYER_STEP,
            trash.y    // PLAYER_STEP,
            trash.x    // 100,
            obs.y      // PLAYER_STEP,
            obs.x      // 100
        )

    def choose(self, player_y, trash_list, obstacle_list):
        player_y = max(0, min(player_y, SCREEN_HEIGHT))
        if trash_list:
            trash = min(trash_list, key=lambda t: t.x)
        else:
            trash = type('Trash', (), {'x':800,'y':SCREEN_HEIGHT//2})()
        if obstacle_list:
            obs = min(obstacle_list, key=lambda o: o.x)
        else:
            obs = type('Obstacle', (), {'x':900,'y':SCREEN_HEIGHT//2})()

        state  = self.get_state(player_y, trash, obs)
        action = self.agent.choose_action(state)
        self.last_state, self.last_action = state, action
        return action

    def learn(self, reward, player_y, trash_list, obstacle_list, done):
        if trash_list and obstacle_list:
            trash = min(trash_list, key=lambda t: t.x)
            obs   = min(obstacle_list, key=lambda o: o.x)
            next_state = self.get_state(player_y, trash, obs)
        else:
            next_state = self.last_state

        self.agent.learn(self.last_state, self.last_action, reward, next_state)

        if done:
            self.agent.epsilon = max(EPSILON_MIN, self.agent.epsilon * DECAY_RATE)
            self.agent.save(QTABLE_PATH)
            print(f"[IA] Q-Table guardada en {QTABLE_PATH} - eps={self.agent.epsilon:.3f}")

if __name__ == "__main__":
    train()