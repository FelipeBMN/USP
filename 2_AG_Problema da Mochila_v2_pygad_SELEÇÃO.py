#%% PyGAD - Diferentes Tipos de Seleção

#Instalação: pip install pygad

import pygad
import pandas as pd

#%% Dados da mochila
mochila = pd.DataFrame({
    'item': ["barra de cereal", "casaco", "tênis", "celular", "água", "protetor solar", "protetor labial", "garrafas de oxigênio", "máquina fotográfica"],
    'pontos': [6, 7, 3, 2, 9, 5, 2, 10, 6],
    'peso': [200, 400, 400, 100, 1000, 200, 30, 3000, 500]
})

def fitness_mochila(ga_instance, solution, solution_idx):
    pontos = sum(mochila['pontos'][i] for i in range(9) if solution[i] == 1)
    peso = sum(mochila['peso'][i] for i in range(9) if solution[i] == 1)
    return pontos if peso <= 5000 else 0

#%%============================================================================
# 1. SELEÇÃO POR TORNEIO (Tournament Selection)
# ============================================================================

print(" 1) SELEÇÃO POR TORNEIO")
ga_torneio = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    parent_selection_type="tournament",  # ← TORNEIO
    K_tournament=3                       # ← Tamanho do torneio (3 competidores)
)

ga_torneio.run()
solution_torneio, fitness_torneio, _ = ga_torneio.best_solution()
print(f"Resultado Torneio: Fitness = {fitness_torneio}")

#%%============================================================================
# 2. SELEÇÃO POR ROLETA (Roulette Wheel Selection)
# ============================================================================

print("\n 2) SELEÇÃO POR ROLETA")
ga_roleta = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    parent_selection_type="rws"          # ← ROLETA (Roulette Wheel Selection)
)

ga_roleta.run()
solution_roleta, fitness_roleta, _ = ga_roleta.best_solution()
print(f"Resultado Roleta: Fitness = {fitness_roleta}")

#%%============================================================================
# 3. SELEÇÃO STEADY-STATE (Steady-State Selection)
# ============================================================================

print("\n 3) SELEÇÃO STEADY-STATE")
ga_steady = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    parent_selection_type="sss"          # ← STEADY-STATE
)

ga_steady.run()
solution_steady, fitness_steady, _ = ga_steady.best_solution()
print(f"Resultado Steady-State: Fitness = {fitness_steady}")

#%%============================================================================
# 4. SELEÇÃO ALEATÓRIA (Random Selection)
# ============================================================================

print("\n 4) SELEÇÃO ALEATÓRIA")
ga_random = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    parent_selection_type="random"       # ← ALEATÓRIA
)

ga_random.run()
solution_random, fitness_random, _ = ga_random.best_solution()
print(f"Resultado Aleatório: Fitness = {fitness_random}")

#%%============================================================================
# 5. SELEÇÃO POR RANK (Rank Selection)
# ============================================================================

print("\n 5) SELEÇÃO POR RANK")
ga_rank = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    parent_selection_type="rank"         # ← RANK
)

ga_rank.run()
solution_rank, fitness_rank, _ = ga_rank.best_solution()
print(f"Resultado Rank: Fitness = {fitness_rank}")

#%%============================================================================
# COMPARAÇÃO DOS RESULTADOS
# ============================================================================

print(f"\n --> COMPARAÇÃO DOS MÉTODOS DE SELEÇÃO:")
print(f"Torneio:     {fitness_torneio} pontos")
print(f"Roleta:      {fitness_roleta} pontos")
print(f"Steady-State: {fitness_steady} pontos")
print(f"Aleatório:   {fitness_random} pontos")
print(f"Rank:        {fitness_rank} pontos")

# Encontrar o melhor método
metodos = {
    "Torneio": fitness_torneio,
    "Roleta": fitness_roleta,
    "Steady-State": fitness_steady,
    "Aleatório": fitness_random,
    "Rank": fitness_rank
}

melhor_metodo = max(metodos, key=metodos.get)
melhor_fitness = metodos[melhor_metodo]

print(f"\n ----> MELHOR MÉTODO: {melhor_metodo} com {melhor_fitness} pontos")

# ============================================================================
# EXPLICAÇÃO DOS MÉTODOS
# ============================================================================

print(f"\n --> EXPLICAÇÃO DOS MÉTODOS:")
print("1) TORNEIO: Seleciona k indivíduos aleatórios, escolhe o melhor")
print("2) ROLETA: Probabilidade proporcional ao fitness")
print("3) STEADY-STATE: Substitui apenas os piores indivíduos")
print("4) ALEATÓRIO: Seleção completamente aleatória")
print("5) RANK: Baseado na posição no ranking, não no fitness absoluto")