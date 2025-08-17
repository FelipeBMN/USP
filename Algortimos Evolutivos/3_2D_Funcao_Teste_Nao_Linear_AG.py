#%% --- Importação das Bibliotecas Necessárias ---

import numpy as np         # Biblioteca para cálculos numéricos eficientes (arrays, operações matemáticas)
import matplotlib.pyplot as plt  # Para criação de gráficos e visualizações

#%% --- 1. Configuração dos Parâmetros do Algoritmo Genético ---

# Define a semente aleatória para garantir reprodutibilidade dos resultados
np.random.seed(1234)

# Hiperparâmetros do Algoritmo Genético que podem ser ajustados para melhorar performance
POP = 14        # Tamanho da população: número de indivíduos (soluções candidatas) em cada geração
BITS = 14       # Número de bits por cromossomo: precisão da codificação binária (2^14 = 16384 valores possíveis)
GENS = 20       # Número de gerações: quantas iterações o algoritmo executará para evoluir as soluções

# Probabilidades dos operadores genéticos (valores típicos que funcionam bem na prática)
PC = 0.8        # Taxa de Crossover: 80% de chance de dois pais gerarem filhos por recombinação
PM = 0.05       # Taxa de Mutação: 5% de chance de cada bit sofrer mutação (inversão 0↔1)

# Domínio da função objetivo que queremos otimizar
XMIN, XMAX = -1.0, 2.0   # Intervalo onde procuraremos o máximo da função f(x) = x*sin(10πx) + 1

#%% --- 2. Definição da Função Objetivo e Conversão Binário-Real ---

# Função matemática que queremos maximizar: f(x) = x * sin(10π*x) + 1
# Esta é uma função multimodal (vários picos) que testa a capacidade do AG de encontrar o máximo global
f = lambda x: x * np.sin(10 * np.pi * x) + 1

def bits_to_real(bits):
    """
    Converte um cromossomo binário (lista de 0s e 1s) para um valor real no domínio [XMIN, XMAX].
    
    Processo:
    1. Converte array binário para string: [1,0,1,1] → "1011"
    2. Interpreta como número binário: "1011" → 11 (decimal)
    3. Normaliza para [0,1]: 11 / (2^14-1) = 11/16383
    4. Mapeia para domínio desejado: XMIN + proporção * (XMAX - XMIN)
    
    Args:
        bits: Array numpy com valores 0 e 1 representando o cromossomo
    
    Returns:
        float: Valor real correspondente no intervalo [XMIN, XMAX]
    """
    binary_string = ''.join(map(str, bits))  # Converte array para string binária
    decimal_value = int(binary_string, 2)    # Converte binário para decimal
    # Mapeia linearmente do intervalo [0, 2^BITS-1] para [XMIN, XMAX]
    return XMIN + (XMAX - XMIN) * decimal_value / (2**BITS - 1)

#%% --- 3. Implementação dos Operadores Genéticos ---

def evolve_population(pop):
    """
    Implementa um ciclo completo de evolução: Avaliação → Seleção → Reprodução → Mutação → Elitismo
    
    Esta função é o coração do algoritmo genético. Ela pega uma população atual e 
    produz a próxima geração aplicando os princípios da evolução natural.
    
    Args:
        pop: Array numpy (POP x BITS) representando a população atual
        
    Returns:
        tuple: (nova_população, valores_x, fitness_valores)
    """
    
    # ETAPA 1: AVALIAÇÃO - Calcula o fitness (aptidão) de cada indivíduo
    x_vals = np.array([bits_to_real(ind) for ind in pop])  # Converte todos os cromossomos para valores reais
    fitness = np.array([f(x) for x in x_vals])             # Avalia a função objetivo para cada valor
    
    # ETAPA 2: SELEÇÃO POR ROLETA VICIADA (Roulette Wheel Selection)
    # Indivíduos com maior fitness têm maior probabilidade de serem selecionados como pais
    # Adiciona pequena constante para evitar divisão por zero e garantir que todos tenham alguma chance
    probs = (fitness - fitness.min() + 0.001)  # Normaliza fitness para valores não-negativos
    probs /= probs.sum()                       # Converte para probabilidades (soma = 1)
    # Seleciona POP indivíduos com reposição baseado nas probabilidades
    selected = pop[np.random.choice(POP, POP, p=probs)]
    
    # ETAPA 3: REPRODUÇÃO (Crossover + Mutação)
    # Gera POP-2 novos indivíduos (reservamos 2 vagas para elitismo)
    new_pop = []
    for i in range(0, POP-2, 2):  # Processa pares de pais
        # Seleciona dois pais da população selecionada
        p1, p2 = selected[i], selected[(i+1) % POP]
        
        # CROSSOVER DE UM PONTO: Recombina material genético dos pais
        if np.random.random() < PC:  # Apenas se sorteio < taxa de crossover
            pt = np.random.randint(1, BITS)  # Escolhe ponto de corte aleatório (1 a BITS-1)
            # Cria filhos trocando segmentos após o ponto de corte
            c1 = np.r_[p1[:pt], p2[pt:]]     # Filho 1: início do pai1 + fim do pai2
            c2 = np.r_[p2[:pt], p1[pt:]]     # Filho 2: início do pai2 + fim do pai1
        else:
            # Se não há crossover, filhos são cópias exatas dos pais
            c1, c2 = p1.copy(), p2.copy()
        
        # MUTAÇÃO BIT-FLIP: Cada bit pode ser invertido independentemente
        for c in [c1, c2]:  # Aplica mutação a ambos os filhos
            # Cria máscara booleana: True onde deve ocorrer mutação
            mask = np.random.random(BITS) < PM
            # Inverte bits selecionados: 0→1, 1→0
            c[mask] = 1 - c[mask]
        
        new_pop.extend([c1, c2])  # Adiciona filhos à nova população
    
    # ETAPA 4: ELITISMO - Preserva os melhores indivíduos
    # Garante que as melhores soluções não sejam perdidas durante a evolução
    elite_idx = np.argsort(fitness)[-2:]  # Índices dos 2 melhores indivíduos
    new_pop.extend([pop[elite_idx[-1]], pop[elite_idx[-2]]])  # Adiciona à nova população
    
    return np.array(new_pop[:POP]), x_vals, fitness

#%% --- 4. Execução Principal do Algoritmo Genético ---

print("AG: Painel 4 Gerações")
print(f"População: {POP} | Gerações: {GENS} | Bits: {BITS}")
print(" -> Elitismo: Preserva os 2 melhores indivíduos")
print("-" * 60)

# INICIALIZAÇÃO: Cria população inicial aleatória
# Cada indivíduo é um cromossomo binário de BITS genes, cada gene pode ser 0 ou 1
pop = np.random.randint(0, 2, (POP, BITS))

# Dicionário para armazenar "snapshots" da evolução em gerações específicas
snapshots = {}

# LOOP PRINCIPAL DE EVOLUÇÃO
for gen in range(GENS):
    # Evolui a população por uma geração completa
    pop, x_vals, fitness = evolve_population(pop)
    best_idx = np.argmax(fitness)  # Encontra o índice do melhor indivíduo
    
    # Captura snapshot e imprime detalhes apenas nas gerações de interesse
    if (gen + 1) in [1, 5, 10, 20]:
        # Armazena estado completo da população para posterior visualização
        snapshots[gen + 1] = (x_vals.copy(), fitness.copy(), x_vals[best_idx], fitness[best_idx])
        
        # Imprime estatísticas resumidas da geração
        print(f"Gen {gen+1:2d}: Melhor = {fitness[best_idx]:.4f} (x={x_vals[best_idx]:.4f}) | "
              f"Média = {np.mean(fitness):.4f}")
        
        # Exibe detalhes completos de cada indivíduo da população
        print("    Cromossomo (binário)    →  Decimal  →     x     →   f(x)")
        print("    " + "-"*56)
        for i, (individual, x_val, fit_val) in enumerate(zip(pop, x_vals, fitness)):
            binary_str = ''.join(map(str, individual))  # Converte cromossomo para string
            decimal_val = int(binary_str, 2)            # Valor decimal correspondente
            marker = " ★" if i == best_idx else "  "    # Marca o melhor indivíduo
            print(f"   {marker}{binary_str}  →   {decimal_val:5d}   → {x_val:7.4f} → {fit_val:7.4f}")
        print()

#%% --- 5. Criação do Painel de Visualização (2x2) ---

# Preparação dos dados para plotagem
x_func = np.linspace(XMIN, XMAX, 500)  # 500 pontos para curva suave da função
y_func = f(x_func)                     # Valores da função objetivo
colors = ['red', 'orange', 'green', 'purple']  # Cores para cada geração
target_gens = [1, 5, 10, 20]         # Gerações que serão visualizadas

# Criação da figura com 4 subplots (2 linhas × 2 colunas)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Evolução do Algoritmo Genético - Painel de Convergência', 
             fontsize=16, fontweight='bold')

# Converte matriz 2x2 de axes em lista linear para facilitar iteração
axes_flat = axes.flatten()

# Cria um gráfico para cada geração selecionada
for i, (gen, color) in enumerate(zip(target_gens, colors)):
    ax = axes_flat[i]  # Subplot atual
    x_vals, fitness, best_x, best_f = snapshots[gen]  # Dados da geração
    
    # ELEMENTO 1: Curva da função objetivo (linha preta contínua)
    # Mostra o "landscape" completo que o AG está explorando
    ax.plot(x_func, y_func, 'k-', linewidth=2.5, alpha=0.7, label='f(x)')
    
    # ELEMENTO 2: População atual (pontos coloridos)
    # Cada ponto representa um indivíduo da população nesta geração
    ax.scatter(x_vals, fitness, c=color, s=70, alpha=0.8, 
              edgecolors='black', linewidth=0.8, label=f'População')
    
    # ELEMENTO 3: Melhor indivíduo (estrela dourada)
    # Destaca visualmente a melhor solução encontrada até agora
    ax.scatter(best_x, best_f, c='gold', s=180, marker='*', 
              edgecolors='red', linewidth=2.5, label=f'Melhor: {best_f:.3f}')
    
    # Configurações estéticas e informativas do subplot
    ax.set_xlim(XMIN, XMAX)               # Limites do eixo X
    ax.set_ylim(-1, 3.2)                  # Limites do eixo Y  
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
    
    # Título colorido indicando a geração
    ax.set_title(f'Geração {gen}', fontsize=14, fontweight='bold', 
                color=color, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.grid(True, alpha=0.4, linestyle='--')  # Grade sutil para facilitar leitura
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)  # Legenda
    
    # Caixa informativa com estatísticas da população
    ax.text(0.02, 0.95, f'Pop: {len(x_vals)} indivíduos\nMédia: {np.mean(fitness):.3f}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Ajusta layout para evitar sobreposição de elementos
plt.tight_layout()
plt.show()

#%% --- 6. Relatório Final de Convergência ---

print("\n --> RESUMO DA EVOLUÇÃO:")
print("Geração  |  Melhor x  |  Melhor f(x)  |  Fitness Médio")
print("-" * 50)

# Imprime tabela resumo da evolução
for gen in target_gens:
    x_vals, fitness, best_x, best_f = snapshots[gen]
    print(f"   {gen:2d}    | {best_x:8.4f} | {best_f:10.4f}  | {np.mean(fitness):10.4f}")

print("✅ Painel 2x2 criado com sucesso!")

# Análise dos resultados:
# - Geração 1: População dispersa explorando todo o espaço de busca
# - Geração 5: Início da convergência, população se concentrando em regiões promissoras  
# - Geração 10: Convergência mais acentuada, exploração refinada dos picos
# - Geração 20: População convergiu para o máximo global (ou local muito bom)