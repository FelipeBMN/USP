#%% PyGAD - Configuração do Crossover (Cruzamento)

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
# 1. CROSSOVER BAIXO (Pouco Cruzamento)
# ============================================================================

print(" -  CROSSOVER BAIXO (30%)")
ga_cross_baixo = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # CROSSOVER BAIXO
    crossover_type="single_point",       # Tipo de cruzamento
    crossover_probability=0.3,           # ← 30% de chance de cruzamento
    
    # Mutação padrão
    mutation_probability=0.1,
    mutation_percent_genes=15,
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_cross_baixo.run()
solution_baixo, fitness_baixo, _ = ga_cross_baixo.best_solution()
print(f"Resultado CROSSOVER BAIXO: Fitness = {fitness_baixo}")

#%%============================================================================
# 2. CROSSOVER NORMAL (Padrão Recomendado)
# ============================================================================

print("\n -  CROSSOVER NORMAL (70%)")
ga_cross_normal = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # CROSSOVER NORMAL
    crossover_type="single_point",       # Tipo de cruzamento
    crossover_probability=0.7,           # ← 70% de chance de cruzamento
    
    # Mutação padrão
    mutation_probability=0.1,
    mutation_percent_genes=15,
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_cross_normal.run()
solution_normal, fitness_normal, _ = ga_cross_normal.best_solution()
print(f"Resultado CROSSOVER NORMAL: Fitness = {fitness_normal}")

#%%============================================================================
# 3. CROSSOVER ALTO (Muito Cruzamento)
# ============================================================================

print("\n -  CROSSOVER ALTO (95%)")
ga_cross_alto = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # CROSSOVER ALTO
    crossover_type="single_point",       # Tipo de cruzamento
    crossover_probability=0.95,          # ← 95% de chance de cruzamento
    
    # Mutação padrão
    mutation_probability=0.1,
    mutation_percent_genes=15,
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_cross_alto.run()
solution_alto, fitness_alto, _ = ga_cross_alto.best_solution()
print(f"Resultado CROSSOVER ALTO: Fitness = {fitness_alto}")

#%%============================================================================
# 4. SEM CROSSOVER (Só Mutação + Seleção)
# ============================================================================

print("\n -  SEM CROSSOVER (0%)")
ga_sem_cross = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # SEM CROSSOVER
    crossover_type="single_point",       
    crossover_probability=0.0,           # ← 0% = SEM cruzamento
    
    # Mutação padrão
    mutation_probability=0.1,
    mutation_percent_genes=15,
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_sem_cross.run()
solution_sem_cross, fitness_sem_cross, _ = ga_sem_cross.best_solution()
print(f"Resultado SEM CROSSOVER: Fitness = {fitness_sem_cross}")

#%%============================================================================
# TIPOS DE CROSSOVER DISPONÍVEIS
# ============================================================================

print("\n --> TESTANDO DIFERENTES TIPOS DE CROSSOVER:")

# CROSSOVER DE DOIS PONTOS
print("\n - CROSSOVER DOIS PONTOS")
ga_dois_pontos = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # CROSSOVER DOIS PONTOS
    crossover_type="two_points",         # ← Cruzamento de dois pontos
    crossover_probability=0.8,           
    
    mutation_probability=0.1,
    parent_selection_type="tournament"
)

ga_dois_pontos.run()
solution_dois_pontos, fitness_dois_pontos, _ = ga_dois_pontos.best_solution()
print(f"Resultado DOIS PONTOS: Fitness = {fitness_dois_pontos}")

# CROSSOVER UNIFORME
print("\n - CROSSOVER UNIFORME")
ga_uniforme = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    keep_elitism=1,
    
    # CROSSOVER UNIFORME
    crossover_type="uniform",            # ← Cruzamento uniforme
    crossover_probability=0.8,           
    
    mutation_probability=0.1,
    parent_selection_type="tournament"
)

ga_uniforme.run()
solution_uniforme, fitness_uniforme, _ = ga_uniforme.best_solution()
print(f"Resultado UNIFORME: Fitness = {fitness_uniforme}")

#%%============================================================================
# COMPARAÇÃO DOS RESULTADOS
# ============================================================================

print(f"\n --> COMPARAÇÃO - EFEITO DA PROBABILIDADE DE CROSSOVER:")
print(f"Sem crossover (0%):      {fitness_sem_cross} pontos")
print(f"Crossover baixo (30%):   {fitness_baixo} pontos")
print(f"Crossover normal (70%):  {fitness_normal} pontos")
print(f"Crossover alto (95%):    {fitness_alto} pontos")

print(f"\n --> COMPARAÇÃO - TIPOS DE CROSSOVER:")
print(f"Um ponto (single_point):  {fitness_normal} pontos")
print(f"Dois pontos (two_points): {fitness_dois_pontos} pontos")
print(f"Uniforme (uniform):       {fitness_uniforme} pontos")

# Encontrar melhor configuração
configuracoes_prob = {
    "Sem crossover": fitness_sem_cross,
    "Crossover baixo": fitness_baixo,
    "Crossover normal": fitness_normal,
    "Crossover alto": fitness_alto
}

melhor_prob = max(configuracoes_prob, key=configuracoes_prob.get)
melhor_fitness_prob = configuracoes_prob[melhor_prob]

configuracoes_tipo = {
    "Um ponto": fitness_normal,
    "Dois pontos": fitness_dois_pontos,
    "Uniforme": fitness_uniforme
}

melhor_tipo = max(configuracoes_tipo, key=configuracoes_tipo.get)
melhor_fitness_tipo = configuracoes_tipo[melhor_tipo]

print(f"\n -->  MELHOR PROBABILIDADE: {melhor_prob} com {melhor_fitness_prob} pontos")
print(f" -->  MELHOR TIPO: {melhor_tipo} com {melhor_fitness_tipo} pontos")

#%%============================================================================
# ANÁLISE DA CONVERGÊNCIA
# ============================================================================

print(f"\n -->  COMPARANDO CONVERGÊNCIA POR PROBABILIDADE:")

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.plot(ga_sem_cross.best_solutions_fitness, 'r-', linewidth=2)
plt.title('Sem Crossover (0%)')
plt.ylabel('Fitness')

plt.subplot(1, 4, 2)
plt.plot(ga_cross_baixo.best_solutions_fitness, 'b-', linewidth=2)
plt.title('Crossover Baixo (30%)')

plt.subplot(1, 4, 3)
plt.plot(ga_cross_normal.best_solutions_fitness, 'g-', linewidth=2)
plt.title('Crossover Normal (70%)')

plt.subplot(1, 4, 4)
plt.plot(ga_cross_alto.best_solutions_fitness, 'orange', linewidth=2)
plt.title('Crossover Alto (95%)')

plt.tight_layout()
plt.show()

#%%============================================================================
# TIPOS DE CROSSOVER DETALHADOS
# ============================================================================

print(f"\n -  TIPOS DE CROSSOVER DISPONÍVEIS NO PYGAD:")

tipos_crossover = {
    "single_point": "Um ponto - corta em uma posição",
    "two_points": "Dois pontos - corta em duas posições", 
    "uniform": "Uniforme - cada gene escolhido aleatoriamente",
    "scattered": "Espalhado - similar ao uniforme"
}

for tipo, descricao in tipos_crossover.items():
    print(f"• {tipo:12}: {descricao}")

print(f"\n - EXEMPLO VISUAL DE CROSSOVER:")
print("Pai 1:    [1, 0, 1, 1, 0, 1, 0, 1, 0]")
print("Pai 2:    [0, 1, 0, 1, 1, 0, 1, 0, 1]")
print("          ↓")
print("Um ponto: [1, 0, 1, |1, 1, 0, 1, 0, 1]  (corte na posição 3)")
print("Uniforme: [1, 1, 1, 1, 1, 1, 0, 1, 1]  (cada gene escolhido)")


print("\n - CONCEITOS PRINCIPAIS: ")
print("• Crossover baixo (30-50%): Pouca recombinação, busca mais local")
print("• Crossover normal (70-80%): Equilibrio ideal exploração/exploitação")
print("• Crossover alto (90-95%): Muita recombinação, mais exploração")
print("• Sem crossover: Demonstra importância da recombinação")

print(f"\n - EXPERIMENTOS SUGERIDOS:")
print("1. Varie crossover_probability: 0.3, 0.5, 0.8, 0.95")
print("2. Compare tipos: single_point vs two_points vs uniform")
print("3. Combine com diferentes taxas de mutação")
print("4. Observe como afeta a convergência")

print(f"\n - SINTESE:")
print("• Crossover = EXPLOITAÇÃO (combina soluções boas)")
print("• Mutação = EXPLORAÇÃO (cria diversidade)")
print("• Equilibrio entre os dois é fundamental!")