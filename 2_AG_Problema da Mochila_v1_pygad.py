#%% Instalação: pip install pygad

import pygad
import pandas as pd


#%%
mochila = pd.DataFrame({
    'item': ["barra de cereal", "casaco", "tênis", "celular", "água", "protetor solar", "protetor labial", "garrafas de oxigênio", "máquina fotográfica"],
    'pontos': [6, 7, 3, 2, 9, 5, 2, 10, 6],
    'peso': [200, 400, 400, 100, 1000, 200, 30, 3000, 500]
})

def fitness_mochila(ga_instance, solution, solution_idx):
    pontos = sum(mochila['pontos'][i] for i in range(9) if solution[i] == 1)
    peso = sum(mochila['peso'][i] for i in range(9) if solution[i] == 1)
    return pontos if peso <= 5000 else 0

ga_instance = pygad.GA(
    num_generations=20, sol_per_pop=10, num_parents_mating=5, fitness_func=fitness_mochila,
    num_genes=9, gene_type=int, gene_space=[0, 1], parent_selection_type="rws",
    keep_elitism=1, crossover_type="single_point", crossover_probability=0.8,
    mutation_type="random", mutation_probability=0.05, mutation_percent_genes=10
)

ga_instance.run()
solution, fitness_value, _ = ga_instance.best_solution()

#%% Apresentar os resultados
print("Solução:", solution)
print("Fitness:", fitness_value)
for i, sel in enumerate(solution):
    if sel: print(f"- {mochila['item'][i]} ({mochila['pontos'][i]} pts, {mochila['peso'][i]}g)")

ga_instance.plot_fitness()