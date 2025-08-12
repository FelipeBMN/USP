#%% PyGAD - Configuração do Elitismo

#Instalação: pip install pygad

import pygad
import pandas as pd

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

#%% ============================================================================
# 1. COM ELITISMO (Padrão - Recomendado)
# ============================================================================

print(" - ALGORITMO COM ELITISMO")
ga_com_elitismo = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    
    # CONFIGURAÇÃO DO ELITISMO
    keep_elitism=2,                    # ← Manter os 2 melhores indivíduos
                                       # Valores típicos: 1-5 ou 1-10% da população
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_com_elitismo.run()
solution_elitismo, fitness_elitismo, _ = ga_com_elitismo.best_solution()
print(f"Resultado COM elitismo: Fitness = {fitness_elitismo}")

#%% ============================================================================
# 2. SEM ELITISMO
# ============================================================================

print("\n - ALGORITMO SEM ELITISMO")
ga_sem_elitismo = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    
    # SEM ELITISMO
    keep_elitism=0,                    # ← 0 = Sem elitismo
                                       # Nenhum indivíduo é preservado automaticamente
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_sem_elitismo.run()
solution_sem_elitismo, fitness_sem_elitismo, _ = ga_sem_elitismo.best_solution()
print(f"Resultado SEM elitismo: Fitness = {fitness_sem_elitismo}")

#%% ============================================================================
# 3. ELITISMO ALTO (Muitos Indivíduos Preservados)
# ============================================================================

print("\n - ALGORITMO COM ELITISMO ALTO")
ga_elitismo_alto = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    
    # ELITISMO ALTO
    keep_elitism=5,                    # ← Manter os 5 melhores (25% da população)
                                       # Mais conservador, menos diversidade
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_elitismo_alto.run()
solution_elitismo_alto, fitness_elitismo_alto, _ = ga_elitismo_alto.best_solution()
print(f"Resultado ELITISMO ALTO: Fitness = {fitness_elitismo_alto}")

#%% ============================================================================
# COMPARAÇÃO DOS RESULTADOS
# ============================================================================

print(f"\n - COMPARAÇÃO - EFEITO DO ELITISMO:")
print(f"Sem elitismo (keep_elitism=0):  {fitness_sem_elitismo} pontos")
print(f"Elitismo normal (keep_elitism=2): {fitness_elitismo} pontos")
print(f"Elitismo alto (keep_elitism=5):   {fitness_elitismo_alto} pontos")

# PyGAD - Configuração do Elitismo

import pygad
import pandas as pd

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

#%% ============================================================================
# 1. COM ELITISMO (Padrão - Recomendado)
# ============================================================================

print(" - ALGORITMO COM ELITISMO")
ga_com_elitismo = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    
    # CONFIGURAÇÃO DO ELITISMO
    keep_elitism=2,                    # ← Manter os 2 melhores indivíduos
                                       # Valores típicos: 1-5 ou 1-10% da população
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_com_elitismo.run()
solution_elitismo, fitness_elitismo, _ = ga_com_elitismo.best_solution()
print(f"Resultado COM elitismo: Fitness = {fitness_elitismo}")

#%% ============================================================================
# 2. SEM ELITISMO
# ============================================================================

print("\n - ALGORITMO SEM ELITISMO")
ga_sem_elitismo = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    
    # SEM ELITISMO
    keep_elitism=0,                    # ← 0 = Sem elitismo
                                       # Nenhum indivíduo é preservado automaticamente
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_sem_elitismo.run()
solution_sem_elitismo, fitness_sem_elitismo, _ = ga_sem_elitismo.best_solution()
print(f"Resultado SEM elitismo: Fitness = {fitness_sem_elitismo}")

#%% ============================================================================
# 3. ELITISMO ALTO (Muitos Indivíduos Preservados)
# ============================================================================

print("\n - ALGORITMO COM ELITISMO ALTO")
ga_elitismo_alto = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    
    # ELITISMO ALTO
    keep_elitism=5,                    # ← Manter os 5 melhores (25% da população)
                                       # Mais conservador, menos diversidade
    
    parent_selection_type="tournament",
    K_tournament=3
)

ga_elitismo_alto.run()
solution_elitismo_alto, fitness_elitismo_alto, _ = ga_elitismo_alto.best_solution()
print(f"Resultado ELITISMO ALTO: Fitness = {fitness_elitismo_alto}")

#%% ============================================================================
# COMPARAÇÃO DOS RESULTADOS
# ============================================================================

print(f"\n - COMPARAÇÃO - EFEITO DO ELITISMO:")
print(f"Sem elitismo (keep_elitism=0):  {fitness_sem_elitismo} pontos")
print(f"Elitismo normal (keep_elitism=2): {fitness_elitismo} pontos")
print(f"Elitismo alto (keep_elitism=5):   {fitness_elitismo_alto} pontos")

# ============================================================================
# ANÁLISE DA CONVERGÊNCIA - CORREÇÃO DO PLOT
# ============================================================================

print(f"\n - ANÁLISE DAS CURVAS DE CONVERGÊNCIA:")

# SOLUÇÃO: Plotar individualmente em vez de subplots
import matplotlib.pyplot as plt

# Extrair dados de fitness manualmente para garantir que funcione
def extrair_fitness_data(ga_instance, titulo):
    """Extrai dados de fitness do histórico do GA"""
    if hasattr(ga_instance, 'best_solutions_fitness') and ga_instance.best_solutions_fitness:
        return ga_instance.best_solutions_fitness
    else:
        # Fallback caso não tenha histórico
        print(f"⚠️  Dados de convergência não disponíveis para {titulo}")
        return []

# Criar figura com subplots
plt.figure(figsize=(12, 4))

# Plot 1: Sem Elitismo
plt.subplot(1, 3, 1)
fitness_data_1 = extrair_fitness_data(ga_sem_elitismo, "Sem Elitismo")
if fitness_data_1:
    plt.plot(range(1, len(fitness_data_1) + 1), fitness_data_1, 'b-', linewidth=2)
    plt.title("Sem Elitismo")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
else:
    # Se não tem dados, criar um plot básico
    plt.text(0.5, 0.5, f'Resultado Final:\n{fitness_sem_elitismo} pontos', 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue'))
    plt.title("Sem Elitismo")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")

# Plot 2: Elitismo Normal
plt.subplot(1, 3, 2)
fitness_data_2 = extrair_fitness_data(ga_com_elitismo, "Elitismo Normal")
if fitness_data_2:
    plt.plot(range(1, len(fitness_data_2) + 1), fitness_data_2, 'g-', linewidth=2)
    plt.title("Elitismo Normal")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, f'Resultado Final:\n{fitness_elitismo} pontos', 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen'))
    plt.title("Elitismo Normal")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")

# Plot 3: Elitismo Alto
plt.subplot(1, 3, 3)
fitness_data_3 = extrair_fitness_data(ga_elitismo_alto, "Elitismo Alto")
if fitness_data_3:
    plt.plot(range(1, len(fitness_data_3) + 1), fitness_data_3, 'r-', linewidth=2)
    plt.title("Elitismo Alto")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, f'Resultado Final:\n{fitness_elitismo_alto} pontos', 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcoral'))
    plt.title("Elitismo Alto")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")

plt.tight_layout()
plt.show()


#%% ============================================================================
# CONFIGURAÇÕES RECOMENDADAS
# ============================================================================

print(f"\n --> RECOMENDAÇÕES PARA ELITISMO:")
print("• keep_elitism=1-2: Para populações pequenas (10-50)")
print("• keep_elitism=2-5: Para populações médias (50-200)")
print("• keep_elitism=5-10: Para populações grandes (200+)")
print("• Regra geral: 5-10% da população")
print(f"• Para população de 20: keep_elitism=1-2 é ideal")

#%% ============================================================================
# EXEMPLO PRÁTICO FINAL - CONFIGURAÇÃO RECOMENDADA
# ============================================================================

print(f"\n --> CONFIGURAÇÃO RECOMENDADA:")

ga_ideal = pygad.GA(
    num_generations=20,
    sol_per_pop=30,
    num_parents_mating=15,
    fitness_func=fitness_mochila,
    num_genes=9,
    gene_type=int,
    gene_space=[0, 1],
    
    # ELITISMO EQUILIBRADO
    keep_elitism=1,                    # ← 1 indivíduo (5% da população)
                                       # Garante que não perde a melhor solução
                                       # Mas permite diversidade
    
    parent_selection_type="tournament",
    K_tournament=3,
    
    # OUTROS PARÂMETROS IMPORTANTES
    mutation_probability=0.05,          # Taxa de mutação
    crossover_probability=0.8          # Taxa de cruzamento
)
