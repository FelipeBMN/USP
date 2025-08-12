#%% Algoritmo Genético Manual - Sem Bibliotecas Externas

import random
import matplotlib.pyplot as plt
import pandas as pd

#%% AG com Seleção por TORNEIO
mochila = pd.DataFrame({
    'item': ["barra de cereal", "casaco", "tênis", "celular", "água", "protetor solar", "protetor labial", "garrafas de oxigênio", "máquina fotográfica"],
    'pontos': [6, 7, 3, 2, 9, 5, 2, 10, 6],
    'peso': [200, 400, 400, 100, 1000, 200, 30, 3000, 500]
})

def fitness(individuo):
    pontos = sum(mochila['pontos'][i] for i in range(9) if individuo[i])
    peso = sum(mochila['peso'][i] for i in range(9) if individuo[i])
    return pontos if peso <= 5000 else 0

def criar_individuo():
    return [random.randint(0, 1) for _ in range(9)]

def criar_populacao(tamanho):
    return [criar_individuo() for _ in range(tamanho)]

def selecao_torneio(populacao, fitness_values, k=3):
    competidores = random.sample(list(zip(populacao, fitness_values)), k)
    return max(competidores, key=lambda x: x[1])[0]

def cruzamento_um_ponto(pai1, pai2):
    ponto = random.randint(1, 8)
    return pai1[:ponto] + pai2[ponto:]

def mutacao(individuo, taxa=0.1):
    return [1-gene if random.random() < taxa else gene for gene in individuo]

def algoritmo_genetico(tam_pop=10, geracoes=50, taxa_mut=0.1):
    populacao = criar_populacao(tam_pop)
    historico = []
    
    for geracao in range(geracoes):
        fitness_values = [fitness(ind) for ind in populacao]
        melhor_fitness = max(fitness_values)
        historico.append(melhor_fitness)
        
        nova_populacao = []
        melhor_idx = fitness_values.index(melhor_fitness)
        nova_populacao.append(populacao[melhor_idx])
        
        while len(nova_populacao) < tam_pop:
            pai1 = selecao_torneio(populacao, fitness_values)
            pai2 = selecao_torneio(populacao, fitness_values)
            filho = cruzamento_um_ponto(pai1, pai2)
            filho = mutacao(filho, taxa_mut)
            nova_populacao.append(filho)
        
        populacao = nova_populacao
    
    fitness_values = [fitness(ind) for ind in populacao]
    melhor_idx = fitness_values.index(max(fitness_values))
    return populacao[melhor_idx], max(fitness_values), historico

solucao, melhor_fitness, historico = algoritmo_genetico()


#%% Visualizando a solução
print("Solução:", solucao)
print("Fitness:", melhor_fitness)
for i, sel in enumerate(solucao):
    if sel: print(f"- {mochila['item'][i]} ({mochila['pontos'][i]} pts, {mochila['peso'][i]}g)")

plt.plot(historico, 'b-', linewidth=2)
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Convergência - AG Manual")
plt.grid(True)
plt.show()