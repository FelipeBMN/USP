#%% --- Importação das Bibliotecas Necessárias ---

#Instalação: pip install pygad
#Instalação: pip install math

import pygad          # A biblioteca principal para o Algoritmo Genético
import numpy as np    # Usada para cálculos numéricos eficientes
import matplotlib.pyplot as plt # Para gerar os gráficos
import math           # Para a função de raiz quadrada (cálculo da distância)

#%% --- 1. Parâmetros do Algoritmo e do Problema ---

# Hiperparâmetros do Algoritmo Genético que podem ser ajustados
num_generations = 50        # Quantas "gerações" ou iterações o algoritmo irá executar
sol_per_pop = 80            # Quantas soluções (rotas) existirão em cada geração
num_parents_mating = 20      # Quantas das melhores soluções de uma geração serão selecionadas como "pais"
mutation_percent_genes = 5   # A chance (em %) de um gene sofrer uma mutação aleatória

# Coordenadas das Cidades (baseado no Exemplo 6.4 do livro de Belfiore e Fávero)
city_coordinates = {1: (10, 30), 2: (20, 50), 3: (50, 90), 4: (70, 30), 5: (90, 50)}
coords_list = list(city_coordinates.values())
city_labels = list(city_coordinates.keys())
num_cities = len(coords_list)

#%% --- 2. Preparação dos Dados ---

# Cada solução (cromossomo) será uma lista de 20 genes binários (0 ou 1).
# Mapeamos cada posição (gene) a uma rota específica. Ex: gene 0 = rota da cidade 1 para 2.
gene_to_arc = [(i, j) for i in range(num_cities) for j in range(num_cities) if i != j]
num_genes = len(gene_to_arc) # Total de 20 genes para 5 cidades

# Para otimizar, pré-calculamos a distância entre todas as cidades e guardamos em uma matriz.
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        dist = math.sqrt((coords_list[i][0] - coords_list[j][0])**2 + 
                         (coords_list[i][1] - coords_list[j][1])**2)
        distance_matrix[i, j] = dist

#%% --- 3. Função de Aptidão (Fitness Function) ---

def fitness_func(ga_instance, solution, solution_idx):
    """
    Esta é a função mais importante. Ela avalia a "qualidade" de cada solução candidata.
    A aptidão (fitness) é o inverso do custo. Um custo menor resulta em uma aptidão maior.
    Custo Total = Distância da Rota + Penalidades por violação das regras do problema.
    """
    total_distance = 0
    penalty = 0
    
    # Decodifica a solução (lista de 0s e 1s) para uma lista de rotas ativas.
    active_arcs = [gene_to_arc[i] for i, gene in enumerate(solution) if gene == 1]
    for arc in active_arcs:
        total_distance += distance_matrix[arc[0], arc[1]]

    # PENALIDADE 1: Verifica se cada cidade tem exatamente UMA SAÍDA e UMA CHEGADA.
    # Esta é uma regra fundamental do caixeiro-viajante.
    departures = np.zeros(num_cities)
    arrivals = np.zeros(num_cities)
    for arc in active_arcs:
        departures[arc[0]] += 1
        arrivals[arc[1]] += 1
    # Adiciona uma grande penalidade por cada violação, "ensinando" o algoritmo.
    penalty += (np.sum(np.abs(departures - 1)) + np.sum(np.abs(arrivals - 1))) * 1000

    # PENALIDADE 2: Verifica se a solução forma SUB-ROTAS (ciclos menores).
    # Uma solução válida deve ser um único ciclo passando por todas as cidades.
    if len(active_arcs) != num_cities:
        # Penalidade se o número de rotas ativas for diferente do número de cidades.
        penalty += 10000
    elif active_arcs:
        # Usamos uma busca em grafo para ver se todas as cidades estão conectadas.
        visited, queue = set(), [active_arcs[0][0]]
        adj_list = {i: [] for i in range(num_cities)}
        for start, end in active_arcs: adj_list[start].append(end)
        
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                if node in adj_list: queue.extend(adj_list[node])
        # Se a busca não conseguiu visitar todas as cidades, é porque há sub-rotas.
        if len(visited) != num_cities:
            penalty += 10000

    total_cost = total_distance + penalty
    # Retorna o inverso do custo. Se o custo for 0, retorna um valor negativo.
    return 1.0 / total_cost if total_cost != 0 else -1

#%% --- 4. Callback para Imprimir o Fitness de Cada Geração ---
def on_generation(ga_instance):
    """
    Esta função é executada ao final de cada geração para mostrar o progresso.
    """
    print(f"Geração: {ga_instance.generations_completed:3} | Fitness: {ga_instance.best_solution()[1]:.4f}")

#%% --- 5. Configuração e Execução do Algoritmo Genético ---

# Inicializa a instância do algoritmo genético com os parâmetros definidos.
ga_instance = pygad.GA(
    num_generations=num_generations,
    sol_per_pop=sol_per_pop,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    num_genes=num_genes,
    gene_type=int,
    gene_space=[0, 1], # Cada gene só pode ser 0 ou 1
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=mutation_percent_genes,
    on_generation=on_generation # Define a função a ser chamada a cada geração
)

print("Executando o Algoritmo Genético...")
# Este comando inicia o processo de evolução.
ga_instance.run()
print("Execução finalizada.")

#%% --- 6. Análise e Visualização do Resultado Final ---

# Pega a melhor solução encontrada após todas as gerações.
solution, solution_fitness, _ = ga_instance.best_solution()
final_distance = 1.0 / solution_fitness if solution_fitness != 0 else float('inf')
final_arcs = [gene_to_arc[i] for i, gene in enumerate(solution) if gene == 1]

# Imprime o resultado final de forma legível.
print("\n--- Melhor Solução Encontrada ---")
print(f"Distância Total: {final_distance:.2f}")
print(f"Arcos Ativos: {[(city_labels[arc[0]], city_labels[arc[1]]) for arc in final_arcs]}")

# Cria e exibe os gráficos com o resultado final.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Resultado Final do Algoritmo Genético para o PCV", fontsize=16)

# Subplot 1: Gráfico da Melhor Rota Encontrada
ax1.scatter([c[0] for c in coords_list], [c[1] for c in coords_list], c='red', zorder=2)
for i, city_coord in enumerate(coords_list):
    ax1.text(city_coord[0] + 1, city_coord[1] + 1, str(city_labels[i]))
for start_node, end_node in final_arcs:
    start_coord, end_coord = coords_list[start_node], coords_list[end_node]
    ax1.arrow(start_coord[0], start_coord[1], end_coord[0] - start_coord[0], 
              end_coord[1] - start_coord[1], head_width=2, head_length=2, 
              fc='blue', ec='blue', length_includes_head=True)
ax1.set_title(f"Melhor Rota Encontrada (Distância: {final_distance:.2f})")
ax1.set_xlabel("Coordenada X")
ax1.set_ylabel("Coordenada Y")
ax1.grid(True)

# Subplot 2: Gráfico do Histórico de Aptidão
ax2.plot(ga_instance.best_solutions_fitness)
ax2.set_title("Evolução da Aptidão por Geração")
ax2.set_xlabel("Geração")
ax2.set_ylabel("Fitness")
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()