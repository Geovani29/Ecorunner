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

TRAIN_EPISODES   = 320000 * 4
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
        self.player_y = SCREEN_HEIGHT // 2
        self.trash_y = random.randint(0, SCREEN_HEIGHT) # Considera un rango menor si es muy dificil
        self.trash_x = 800
        self.obstacle_y = random.randint(0, SCREEN_HEIGHT) # Considera un rango menor si es muy dificil
        # Para aumentar la variabilidad y dificultad gradual, podrías hacer que la X inicial del obstáculo varíe más
        # o que aparezcan más cerca a medida que el agente mejora. Por ahora, lo dejamos así.
        self.obstacle_x = random.choice([900, 1000, 1100, 1200, 1300]) # Un poco más de variedad en la X del obstáculo
        self.steps_alive = 0
        self.last_y = self.player_y
        self.same_y_count = 0
        return self.get_state()

    def get_state(self):
        # La discretización actual es bastante granular. Si el agente tiene problemas,
        # podrías probar con PLAYER_STEP más pequeño o una discretización más fina para X.
        # Por ahora, la mantenemos.
        return (
            self.player_y // PLAYER_STEP,
            self.trash_y // PLAYER_STEP,
            self.trash_x // 100, # Discretización de X de la basura
            self.obstacle_y // PLAYER_STEP,
            self.obstacle_x // 100 # Discretización de X del obstáculo
        )

    def step(self, action):
        if action == 1:  # Subir
            self.player_y -= PLAYER_STEP
        elif action == 2:  # Bajar
            self.player_y += PLAYER_STEP
        # action == 0 (nada) y action == 3 (disparar) no mueven al jugador verticalmente aquí.
        
        self.player_y = max(0, min(SCREEN_HEIGHT - PLAYER_STEP, self.player_y)) # Ajuste para que el jugador no se salga por abajo con su propia altura

        self.trash_x -= TRASH_SPEED
        self.obstacle_x -= TRASH_SPEED
        self.steps_alive += 1

        if self.player_y == self.last_y:
            self.same_y_count += 1
        else:
            self.same_y_count = 0
        self.last_y = self.player_y
        
        reward, done = 0, False

        # --- RECOMPENSAS Y PENALIZACIONES AJUSTADAS ---

        # Objetivo Principal: Disparar Basura
        # Hacemos esta recompensa más destacada.
        if action == 3 and abs(self.player_y - self.trash_y) <= 40 and self.trash_x < 120:
            reward += 50  # Aumentamos recompensa por disparar basura
            self.trash_x = random.choice([750, 800, 850]) # Reset con ligera variabilidad
            self.trash_y = random.randint(0, SCREEN_HEIGHT - PLAYER_STEP)
            # Podríamos añadir un pequeño bonus por disparos rápidos o rachas, pero simplifiquemos por ahora.
        
        # Penalización Mayor: Chocar con Obstáculo
        # Ya la tienes alta, lo cual es bueno.
        elif abs(self.player_y - self.obstacle_y) <= 40 and self.obstacle_x < 50: # Asumimos que 40 es la mitad de la altura del jugador/obstáculo
            reward, done = -100, True  # Mantenemos penalización alta
            
        # Penalización Secundaria: Basura se Escapa
        # Ya la tienes considerable, lo cual está bien.
        elif self.trash_x < 0:
            reward, done = -25, True # Mantenemos penalización considerable, ligeramente ajustada
            
        # Penalización por Inactividad (quedarse en la misma Y)
        # Esto ya lo tienes.
        elif self.same_y_count >= 7: # Aumentamos ligeramente el umbral para no penalizar ajustes finos
            reward -= 5  # Mantenemos penalización por inactividad
            self.same_y_count = 0 # Resetear para que no penalice continuamente si la acción es no moverse.

        # Penalización por Disparar al Vacío (NUEVO)
        # Para desincentivar que el agente haga "spam" de disparos.
        if action == 3 and not (abs(self.player_y - self.trash_y) <= 40 and self.trash_x < 120):
            reward -= 2 # Pequeña penalización por disparar y fallar (o disparar sin objetivo claro)

        # Bonus de Supervivencia (MODIFICADO SIGNIFICATIVAMENTE)
        # Recompensa muy pequeña por cada paso, o una un poco mayor cada más pasos.
        # La idea es que no compita con la recompensa de disparar basura.
        # Opción 1: Pequeña recompensa por paso.
        # reward += 0.1 # Recompensa muy pequeña por simplemente estar vivo.
        # Opción 2: Recompensa modesta cada N pasos.
        if self.steps_alive % 10 == 0: # Cada 10 pasos
            reward += 1  # Recompensa de supervivencia mucho menor

        # Asegurar que el episodio termine si alcanza el máximo de pasos (si no lo hace ya el bucle de entrenamiento)
        if self.steps_alive >= MAX_STEPS_PER_EP and not done:
            # done = True # Opcional: si quieres que termine aquí y reciba la recompensa acumulada.
            # reward += 0 # O una pequeña recompensa/penalización por llegar al final.
            pass


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