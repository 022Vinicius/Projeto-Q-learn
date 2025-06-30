from connection import connect, get_state_reward
import numpy as np

Q = np.zeros((96, 3))  # Q-table zerada
actions = ["left","right", "jump"]
alpha = 0.2  # taxa de aprendizado
gamma = 0.9  # fator de desconto
epsilon = 0.5  # taxa de exploração TESTE1 = 0.5, TESTE2 = 0.6, TESTE3 = 0.7

def state_to_index(estado_binario):
    plataforma = int(estado_binario[:5], 2)
    direcao = int(estado_binario[5:], 2)
    return plataforma * 4 + direcao

port = 2037
s = connect(port)
if s == 0:
    exit()

# Loop principal do agente
for episode in range(6000):
    acao_inicial = "jump"
    estado, recompensa = get_state_reward(s, acao_inicial)
    state_idx = state_to_index(estado)
    done = False
    while not done:
        # Escolhe a ação com base na política epsilon-greedy
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(3) # ação aleatória
        else:
            action_idx = np.argmax(Q[state_idx])
        action = actions[action_idx]
        # Evita pular se estiver em uma direção perigosa no início
       
        #direcao = estado[-2:]
        #if action == "jump" and direcao in ["01", "10", "11"]:  # Exemplo: só pula se estiver virado para Leste ou Oeste
            # Escolhe girar para a direita para evitar cair
        #    action = "right"

        estado, recompensa = get_state_reward(s, action)
        #print(f"Episódio: {episode}, Estado: {estado}, Ação: {action}, Recompensa: {recompensa}")  # <-- Adicione esta linha
        next_state_idx = state_to_index(estado)
        # Atualize Q-table
        Q[state_idx, action_idx] = Q[state_idx, action_idx] + alpha * (
            recompensa + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action_idx]
        )

        state_idx = next_state_idx
        #condição de parada :
        if recompensa == -1 or recompensa == -14:
            done = True

        print(f"Episódio: {episode}, Estado: {estado}, Ação: {action}, Recompensa: {recompensa}")
# Salve a Q-table ao final
np.savetxt("resultado.txt", Q, fmt="%.6f")
s.close()