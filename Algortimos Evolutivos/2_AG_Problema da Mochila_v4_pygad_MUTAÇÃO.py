#%% PyGAD - Configuração da Mutação

#Instalação: pip install pygad

import pygad
import pandas as pd
import matplotlib.pyplot as plt

# Dados da mochila
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
# 1. MUTAÇÃO MUITO BAIXA (Pouca Exploração)
#  ============================================================================

print(" - MUTAÇÃO MUITO BAIXA (1%)")
ga_mut_baixa = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # MUTAÇÃO MUITO BAIXA
    mutation_type="random",              # Tipo de mutação
    mutation_probability=0.01,           # ← 1% de chance de mutação 
    mutation_percent_genes=5,            # ← 5% dos genes podem sofrer mutação
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_mut_baixa.run()
solution_baixa, fitness_baixa, _ = ga_mut_baixa.best_solution()
print(f"Resultado MUTAÇÃO BAIXA: Fitness = {fitness_baixa}")

#%% ============================================================================
# 2. MUTAÇÃO NORMAL (Equilibrada)
# ============================================================================

print("\n - MUTAÇÃO NORMAL (10%)")
ga_mut_normal = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # MUTAÇÃO NORMAL
    mutation_type="random",              # Tipo de mutação
    mutation_probability=0.1,            # ← 10% de chance de mutação 
    mutation_percent_genes=15,           # ← 15% dos genes podem sofrer mutação
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_mut_normal.run()
solution_normal, fitness_normal, _ = ga_mut_normal.best_solution()
print(f"Resultado MUTAÇÃO NORMAL: Fitness = {fitness_normal}")

#%% ============================================================================
# 3. MUTAÇÃO ALTA (Muita Exploração)
# ============================================================================

print("\n - MUTAÇÃO ALTA (25%)")
ga_mut_alta = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # MUTAÇÃO ALTA
    mutation_type="random",              # Tipo de mutação
    mutation_probability=0.25,           # ← 25% de chance de mutação 
    mutation_percent_genes=30,           # ← 30% dos genes podem sofrer mutação
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_mut_alta.run()
solution_alta, fitness_alta, _ = ga_mut_alta.best_solution()
print(f"Resultado MUTAÇÃO ALTA: Fitness = {fitness_alta}")

#%% ============================================================================
# 4. SEM MUTAÇÃO (Só Cruzamento)
# ============================================================================

print("\n - QUASE SEM MUTAÇÃO (0.1%)")
ga_sem_mut = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # SEM MUTAÇÃO (simulada)
    mutation_type="random",              
    mutation_probability=0.001,          # ← 0.1% = Quase sem mutação
    mutation_percent_genes=1,            # ← Mínimo permitido (1%)
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_sem_mut.run()
solution_sem_mut, fitness_sem_mut, _ = ga_sem_mut.best_solution()
print(f"Resultado QUASE SEM MUTAÇÃO: Fitness = {fitness_sem_mut}")

#%% ============================================================================
# TIPOS DE MUTAÇÃO DISPONÍVEIS
# ============================================================================

print("\n --> TESTANDO DIFERENTES TIPOS DE MUTAÇÃO:")

# MUTAÇÃO ADAPTATIVA
print("\n - MUTAÇÃO ADAPTATIVA")
ga_mut_adaptativa = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # MUTAÇÃO ADAPTATIVA
    mutation_type="adaptive",            # ← Adapta taxa automaticamente
    mutation_probability=[0.25, 0.05],   # ← [inicial, final]
    
    parent_selection_type="tournament"
)

ga_mut_adaptativa.run()
solution_adaptativa, fitness_adaptativa, _ = ga_mut_adaptativa.best_solution()
print(f"Resultado MUTAÇÃO ADAPTATIVA: Fitness = {fitness_adaptativa}")

#%% ============================================================================
# COMPARAÇÃO DOS RESULTADOS
# ============================================================================

print(f"\n --> COMPARAÇÃO - EFEITO DA MUTAÇÃO:")
print(f"Quase sem mutação (0.1%): {fitness_sem_mut} pontos")
print(f"Mutação baixa (1%):      {fitness_baixa} pontos")
print(f"Mutação normal (10%):    {fitness_normal} pontos")
print(f"Mutação alta (25%):      {fitness_alta} pontos")
print(f"Mutação adaptativa:      {fitness_adaptativa} pontos")

# Encontrar melhor configuração
configuracoes = {
    "Quase sem mutação": fitness_sem_mut,
    "Mutação baixa": fitness_baixa,
    "Mutação normal": fitness_normal,
    "Mutação alta": fitness_alta,
    "Mutação adaptativa": fitness_adaptativa
}

melhor_config = max(configuracoes, key=configuracoes.get)
melhor_fitness = configuracoes[melhor_config]

print(f"\n --> MELHOR CONFIGURAÇÃO: {melhor_config} com {melhor_fitness} pontos")

#%% ============================================================================
# ANÁLISE DA CONVERGÊNCIA
# ============================================================================

print(f"\n --> COMPARANDO CONVERGÊNCIA:")

plt.figure(figsize=(15, 3))

plt.subplot(1, 5, 1)
plt.plot(ga_sem_mut.best_solutions_fitness, 'r-', linewidth=2)
plt.title('Quase Sem Mutação')
plt.ylabel('Fitness')

plt.subplot(1, 5, 2)
plt.plot(ga_mut_baixa.best_solutions_fitness, 'b-', linewidth=2)
plt.title('Mutação Baixa (1%)')

plt.subplot(1, 5, 3)
plt.plot(ga_mut_normal.best_solutions_fitness, 'g-', linewidth=2)
plt.title('Mutação Normal (10%)')

plt.subplot(1, 5, 4)
plt.plot(ga_mut_alta.best_solutions_fitness, 'orange', linewidth=2)
plt.title('Mutação Alta (25%)')

plt.subplot(1, 5, 5)
plt.plot(ga_mut_adaptativa.best_solutions_fitness, 'purple', linewidth=2)
plt.title('Mutação Adaptativa')

plt.tight_layout()
plt.show()

#%% ============================================================================
# CONFIGURAÇÕES DETALHADAS DA MUTAÇÃO
# ============================================================================

print(f"\n --> TIPOS DE MUTAÇÃO DISPONÍVEIS NO PYGAD:")

tipos_mutacao = {
    "random": "Aleatória - troca genes por valores aleatórios",
    "swap": "Troca - troca posições de dois genes", 
    "inversion": "Inversão - inverte sequência de genes",
    "scramble": "Embaralha - embaralha genes aleatórios",
    "adaptive": "Adaptativa - ajusta taxa automaticamente"
}

for tipo, descricao in tipos_mutacao.items():
    print(f"• {tipo:12}: {descricao}")

#%% ============================================================================
#   CONFIGURAÇÃO RECOMENDADA
#   ============================================================================

print(f"\n --> CONFIGURAÇÃO RECOMENDADA:")

ga_ideal = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # MUTAÇÃO EQUILIBRADA
    mutation_type="random",              # Tipo simples e compreensível
    mutation_probability=0.1,            # 10% - taxa moderada
    mutation_percent_genes=15,           # 15% dos genes por indivíduo
    
    # CRUZAMENTO
    crossover_type="single_point",       # Cruzamento de um ponto
    crossover_probability=0.8,           # 80% de chance de cruzamento
    
    parent_selection_type="tournament",
    K_tournament=3
)

print("\n --> TEORIA:")
print("• Mutação baixa (1-5%): Convergência rápida, pode ficar preso")
print("• Mutação média (10-15%): Equilibrio entre exploração e exploitação")
print("• Mutação alta (20-30%): Muita exploração, convergência lenta")
print("• Quase sem mutação (0.1%): Convergência limitada, pouca diversidade")
print("• Adaptativa: Mostra como algoritmos podem se auto-ajustar")

print(f"\n --> EXPERIMENTO SUGERIDO:")
print("Execute este código várias vezes e compare:")
print("- Qual configuração é mais consistente?")
print("- Qual encontra soluções melhores?")
print("- Como a convergência varia?")