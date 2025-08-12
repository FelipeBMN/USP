#%% ============================================================================
#   PyGAD - ALGORITMO GENÉTICO COMPLETO COM TODOS OS HIPERPARÂMETROS
#   Problema da Mochila 
#   ============================================================================

#Instalação: pip install pygad

import pygad
import pandas as pd

#%% ============================================================================
# DADOS DO PROBLEMA DA MOCHILA
# ============================================================================

mochila = pd.DataFrame({
    'item': ["barra de cereal", "casaco", "tênis", "celular", "água", "protetor solar", "protetor labial", "garrafas de oxigênio", "máquina fotográfica"],
    'pontos': [6, 7, 3, 2, 9, 5, 2, 10, 6],
    'peso': [200, 400, 400, 100, 1000, 200, 30, 3000, 500]
})

# ============================================================================
# FUNÇÃO DE FITNESS (AVALIAÇÃO DA QUALIDADE DA SOLUÇÃO)
# ============================================================================

def fitness_mochila(ga_instance, solution, solution_idx):
    """
    Função que avalia a qualidade de uma solução
    Retorna: pontos totais se peso ≤ 5000g, senão 0
    """
    pontos = sum(mochila['pontos'][i] for i in range(9) if solution[i] == 1)
    peso = sum(mochila['peso'][i] for i in range(9) if solution[i] == 1)
    return pontos if peso <= 5000 else 0

#%% ============================================================================
#   CONFIGURAÇÃO DO ALGORITMO GENÉTICO
#   ============================================================================

ga_instance = pygad.GA(
    
    # ========================================================================
    # PARÂMETROS DE POPULAÇÃO E EVOLUÇÃO
    # ========================================================================
    
    num_generations=100,             # Número de gerações (iterações)
                                    # Valores típicos: 50-500
                                    # Mais gerações = busca mais longa
    
    sol_per_pop=10,                 # Tamanho da população (indivíduos)
                                    # Valores típicos: 10-200
                                    # Maior = mais diversidade, mais lento
                                    # Menor = convergência rápida, risco mínimo local
    
    num_parents_mating=5,           # Número de pais selecionados para reprodução
                                    # Valores típicos: 25-75% da população
                                    # Aqui: 5/10 = 50% se reproduzem
    
    # ========================================================================
    # DEFINIÇÃO DO PROBLEMA
    # ========================================================================
    
    fitness_func=fitness_mochila,   # Função que avalia cada solução
    
    num_genes=9,                    # Número de genes por cromossomo
                                    # Aqui: 9 itens da mochila
    
    gene_type=int,                  # Tipo dos genes
                                    # Opções: int, float, [int, float, int, ...]
    
    gene_space=[0, 1],              # Espaço de valores possíveis para cada gene
                                    # [0, 1] = binário (não leva, leva)
                                    # Alternativas:
                                    # [(0, 10)] = inteiros de 0 a 10
                                    # [0, 1, 2, 3] = valores discretos específicos
    
    # ========================================================================
    # SELEÇÃO DE PAIS (PARENT SELECTION)
    # ========================================================================
    
    parent_selection_type="rws",    # Tipo de seleção de pais
                                    # Opções:
                                    # "sss" = Steady-State Selection
                                    # "rws" = Roulette Wheel Selection (roleta)
                                    # "tournament" = Tournament Selection (torneio)
                                    # "rank" = Rank Selection
                                    # "random" = Random Selection
    
    # K_tournament=3,               # ← Use apenas com "tournament"
                                    # Número de competidores no torneio
                                    # Valores típicos: 2-5
    
    # ========================================================================
    # ELITISMO
    # ========================================================================
    
    keep_elitism=1,                 # Número de melhores indivíduos preservados
                                    # Valores típicos: 0-10% da população
                                    # 0 = sem elitismo
                                    # 1 = preserva apenas o melhor
                                    # 2-3 = elitismo moderado
                                    # Aqui: 1/10 = 10% elite
    
    # ========================================================================
    # CRUZAMENTO (CROSSOVER)
    # ========================================================================
    
    crossover_type="single_point",  # Tipo de cruzamento
                                    # Opções:
                                    # "single_point" = Um ponto de corte
                                    # "two_points" = Dois pontos de corte
                                    # "uniform" = Uniforme (cada gene escolhido)
                                    # "scattered" = Espalhado (similar uniforme)
    
    crossover_probability=0.8,      # Probabilidade de cruzamento (80%)
                                    # Valores típicos: 0.6-0.9
                                    # Alto = muita recombinação
                                    # Baixo = pouca mistura de genes
                                    # 0.8 = 80% dos casais fazem cruzamento
    
    # ========================================================================
    # MUTAÇÃO (MUTATION)
    # ========================================================================
    
    mutation_type="random",         # Tipo de mutação
                                    # Opções:
                                    # "random" = Aleatória (mais comum)
                                    # "swap" = Troca posições de genes
                                    # "inversion" = Inverte sequência
                                    # "scramble" = Embaralha genes
                                    # "adaptive" = Taxa adaptativa
    
    mutation_probability=0.1,       # Probabilidade de mutação por gene (10%)
                                    # Valores típicos: 0.01-0.2
                                    # Alto = muita exploração, convergência lenta
                                    # Baixo = pouca diversidade, exploitação
                                    # 0.1 = 10% chance de cada gene sofrer mutação
    
    mutation_percent_genes=15       # Porcentagem de genes que podem sofrer mutação
                                    # Valores típicos: 5-25%
                                    # Controla quantos genes por indivíduo podem mutar
                                    # 15% = até 15% dos genes de um indivíduo
    
    # ========================================================================
    # OUTROS PARÂMETROS AVANÇADOS (OPCIONAIS)
    # ========================================================================
    
    # keep_parents=-1,              # Número de pais mantidos na próxima geração
                                    # -1 = todos os pais são substituídos
                                    # 0 = nenhum pai mantido
                                    # >0 = número específico mantido
    
    # init_range_low=0,             # Valor mínimo para inicialização aleatória
    # init_range_high=1,            # Valor máximo para inicialização aleatória
    
    # random_seed=None,             # Semente para reprodutibilidade
                                    # None = aleatório
                                    # Número = resultados reproduzíveis
    
    # parallel_processing=None,     # Processamento paralelo
                                    # None = sequencial
                                    # ['thread', n] = n threads
                                    # ['process', n] = n processos
    
    # stop_criteria=None            # Critérios de parada
                                    # "reach_xxx" = para quando fitness atinge xxx
                                    # "saturate_xxx" = para após xxx gerações sem melhoria
)

#%% ============================================================================
#   EXECUÇÃO DO ALGORITMO GENÉTICO
#   ============================================================================

print(" --> EXECUTANDO ALGORITMO GENÉTICO...")
print(f"População: {ga_instance.sol_per_pop} indivíduos")
print(f"Gerações: {ga_instance.num_generations}")
print(f"Seleção: {ga_instance.parent_selection_type}")
print(f"Crossover: {ga_instance.crossover_probability*100}%")
print(f"Mutação: {ga_instance.mutation_probability*100}%")
print(f"Elitismo: {ga_instance.keep_elitism} indivíduo(s)")
print()

ga_instance.run()

#============================================================================
# ANÁLISE DOS RESULTADOS
# ============================================================================

solution, fitness_value, _ = ga_instance.best_solution()

print("="*50)
print("RESULTADOS FINAIS")
print("="*50)
print("Solução binária:", solution)
print("Fitness (pontos):", fitness_value)

print("\nItens selecionados:")
peso_total = 0
for i, sel in enumerate(solution):
    if sel: 
        print(f"- {mochila['item'][i]} ({mochila['pontos'][i]} pts, {mochila['peso'][i]}g)")
        peso_total += mochila['peso'][i]

print(f"\nResumo:")
print(f"Pontos totais: {fitness_value}")
print(f"Peso total: {peso_total}g / 5000g")
print(f"Utilização: {peso_total/5000*100:.1f}%")

#%% ============================================================================
# VISUALIZAÇÃO DA EVOLUÇÃO
# ============================================================================

print("\n --> Gerando gráfico de convergência...")
ga_instance.plot_fitness(
    title="Evolução do Algoritmo Genético - Problema da Mochila",
    xlabel="Geração",
    ylabel="Fitness (Pontos)",
    linewidth=2
)

#%% ============================================================================
#  EXPERIMENTOS SUGERIDOS 
#  ============================================================================

print("\n" + "="*50)
print("EXPERIMENTOS:")
print("="*50)

print("\n1. TAMANHO DA POPULAÇÃO:")
print("   sol_per_pop = 5, 10, 20, 50")
print("   Observe: velocidade vs qualidade da solução")

print("\n2. TIPOS DE SELEÇÃO:")
print("   parent_selection_type = 'rws', 'tournament', 'rank'")
print("   Compare: qual encontra melhores soluções?")

print("\n3. PROBABILIDADE DE CROSSOVER:")
print("   crossover_probability = 0.3, 0.6, 0.8, 0.95")
print("   Analise: efeito na convergência")

print("\n4. TAXA DE MUTAÇÃO:")
print("   mutation_probability = 0.01, 0.05, 0.1, 0.2")
print("   Observe: exploração vs exploitação")

print("\n5. ELITISMO:")
print("   keep_elitism = 0, 1, 2, 3")
print("   Compare: estabilidade da convergência")

print("\n6. NÚMERO DE GERAÇÕES:")
print("   num_generations = 20, 50, 100, 200")
print("   Veja: quando para de melhorar?")

print("\n --> DICA: Execute múltiplas vezes para ver variabilidade!")
print(" --> OBJETIVO: Entender como cada parâmetro afeta o resultado!")

#%% ============================================================================
#   CONFIGURAÇÕES ALTERNATIVAS PRÉ-DEFINIDAS
#   ============================================================================

print("\n" + "="*50)
print("CONFIGURAÇÕES ALTERNATIVAS PARA TESTAR")
print("="*50)

configs_alternativas = {
    "Conservador": {
        "sol_per_pop": 20,
        "mutation_probability": 0.05,
        "crossover_probability": 0.9,
        "keep_elitism": 2
    },
    "Explorador": {
        "sol_per_pop": 30,
        "mutation_probability": 0.2,
        "crossover_probability": 0.6,
        "keep_elitism": 1
    },
    "Rápido": {
        "sol_per_pop": 10,
        "num_generations": 30,
        "mutation_probability": 0.15,
        "keep_elitism": 1
    },
    "Intensivo": {
        "sol_per_pop": 50,
        "num_generations": 200,
        "mutation_probability": 0.08,
        "keep_elitism": 3
    }
}

for nome, config in configs_alternativas.items():
    print(f"\n{nome}:")
    for param, valor in config.items():
        print(f"  {param} = {valor}")

print(f"\n - Use essas configurações para demonstrar diferentes estratégias!")
print(f" - Compare os resultados!")