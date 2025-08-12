#%% ALGORITMO GENÉTICO COM PYGAD vs PROGRAMAÇÃO LINEAR
# Problema da Ração Animal

#Instalação: pip install pygad

import numpy as np
import matplotlib.pyplot as plt
import pygad
import pulp

print(" - ALGORITMO GENÉTICO (PyGAD) vs PROGRAMAÇÃO LINEAR")
print("=" * 65)
print("Biblioteca PyGAD: Implementação profissional de AG")
print("=" * 65)

#%%=============================================================================
# CONFIGURAÇÃO DO PROBLEMA
# =============================================================================

# Dados do problema
CUSTOS = np.array([0.56, 0.81, 0.46])  # osso, soja, peixe
PROTEINA_COEF = np.array([0.2, 0.5, 0.4])
CALCIO_COEF = np.array([0.6, 0.4, 0.4])
PROTEINA_MIN = 0.3
CALCIO_MIN = 0.5

print(f" - DADOS DO PROBLEMA:")
print(f"   Custos ($/kg): Osso={CUSTOS[0]}, Soja={CUSTOS[1]}, Peixe={CUSTOS[2]}")
print(f"   Restrições: Proteína ≥ {PROTEINA_MIN*100}%, Cálcio ≥ {CALCIO_MIN*100}%")

#%%# =============================================================================
# FUNÇÃO DE FITNESS PARA O PYGAD
# =============================================================================

def calcular_fitness(ga_instance, solution, solution_idx):
    # Normalizar para que a soma seja 1
    x = np.array(solution)
    if np.sum(x) == 0:
        return 0.001  # Evitar divisão por zero
    
    x_norm = x / np.sum(x)
    x1, x2, x3 = x_norm
    
    # Calcular custo
    custo = np.dot(CUSTOS, x_norm)
    
    # Calcular restrições
    proteina = np.dot(PROTEINA_COEF, x_norm)
    calcio = np.dot(CALCIO_COEF, x_norm)
    
    # Aplicar penalidades por violação de restrições
    penalidade = 0
    
    if proteina < PROTEINA_MIN:
        penalidade += (PROTEINA_MIN - proteina) * 100  # Penalidade alta
        
    if calcio < CALCIO_MIN:
        penalidade += (CALCIO_MIN - calcio) * 100
    
    # Fitness = 1 / (custo + penalidade)
    custo_penalizado = custo + penalidade
    fitness = 1.0 / (custo_penalizado + 0.001)  # +0.001 evita divisão por zero
    
    return fitness

def callback_geracao(ga_instance):
    """Callback chamado a cada geração para acompanhar evolução"""
    geracao = ga_instance.generations_completed
    fitness_atual = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    custo_atual = 1.0 / fitness_atual - 0.001
    
    if geracao % 50 == 0 or geracao == 1:
        print(f"Geração {geracao:3d}: Melhor custo = ${custo_atual:.4f}")

#%%=============================================================================
# SOLUÇÃO COM PROGRAMAÇÃO LINEAR (REFERÊNCIA)
# =============================================================================

def resolver_programacao_linear():
    """Resolve usando PuLP para ter a solução ótima de referência"""
    print("\n - RESOLVENDO COM PROGRAMAÇÃO LINEAR")
    print("-" * 45)
    
    prob = pulp.LpProblem("Racao_Otima", pulp.LpMinimize)
    
    x1 = pulp.LpVariable("osso", lowBound=0)
    x2 = pulp.LpVariable("soja", lowBound=0) 
    x3 = pulp.LpVariable("peixe", lowBound=0)
    
    # Função objetivo
    prob += 0.56*x1 + 0.81*x2 + 0.46*x3
    
    # Restrições
    prob += 0.2*x1 + 0.5*x2 + 0.4*x3 >= 0.3  # Proteína
    prob += 0.6*x1 + 0.4*x2 + 0.4*x3 >= 0.5  # Cálcio
    prob += x1 + x2 + x3 == 1  # Soma
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    solucao = np.array([x1.varValue, x2.varValue, x3.varValue])
    custo = pulp.value(prob.objective)
    
    print(f" - SOLUÇÃO ÓTIMA:")
    print(f"   Custo: ${custo:.4f}")
    print(f"   Osso: {solucao[0]:.3f} ({solucao[0]:.1%})")
    print(f"   Soja: {solucao[1]:.3f} ({solucao[1]:.1%})")
    print(f"   Peixe: {solucao[2]:.3f} ({solucao[2]:.1%})")
    
    return solucao, custo

#%%=============================================================================
# EXECUÇÃO DO ALGORITMO GENÉTICO COM PYGAD
# =============================================================================

def executar_pygad(num_execucoes=3):
    """Executa o PyGAD múltiplas vezes para mostrar variabilidade"""
    print("\n - RESOLVENDO COM PYGAD")
    print("-" * 45)
    
    resultados = []
    
    for execucao in range(num_execucoes):
        print(f"\n - Execução {execucao + 1}:")
        
        # Configuração do PyGAD
        ga_instance = pygad.GA(
            # Genes e espaço de busca
            num_generations=300,
            num_parents_mating=10,
            sol_per_pop=50,
            num_genes=3,
            gene_space=[{'low': 0.0, 'high': 1.0} for _ in range(3)],
            
            # Função de fitness
            fitness_func=calcular_fitness,
            
            # Operadores genéticos
            parent_selection_type="tournament",
            K_tournament=3,
            crossover_type="single_point",
            mutation_type="random",  
            mutation_percent_genes=20,
            mutation_by_replacement=False,
            random_mutation_min_val=-0.1,
            random_mutation_max_val=0.1,
            
            # Callback e configurações
            on_generation=callback_geracao if execucao == 0 else None,  # Só mostrar na primeira
            suppress_warnings=True,
            random_seed=42 + execucao  # Seed diferente para cada execução
        )
        
        # Executar o algoritmo
        ga_instance.run()
        
        # Obter melhor solução
        melhor_solucao, melhor_fitness, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        
        # Normalizar solução
        melhor_solucao_norm = melhor_solucao / np.sum(melhor_solucao)
        custo = np.dot(CUSTOS, melhor_solucao_norm)
        
        # Verificar restrições
        proteina = np.dot(PROTEINA_COEF, melhor_solucao_norm)
        calcio = np.dot(CALCIO_COEF, melhor_solucao_norm)
        
        resultado = {
            'solucao': melhor_solucao_norm,
            'custo': custo,
            'fitness': melhor_fitness,
            'proteina': proteina,
            'calcio': calcio,
            'proteina_ok': proteina >= PROTEINA_MIN - 0.001,
            'calcio_ok': calcio >= CALCIO_MIN - 0.001,
            'convergencia': ga_instance.best_solutions_fitness
        }
        
        resultados.append(resultado)
        
        # Mostrar resultado da execução
        print(f"   Custo: ${custo:.4f}")
        print(f"   Osso: {melhor_solucao_norm[0]:.3f} ({melhor_solucao_norm[0]:.1%})")
        print(f"   Soja: {melhor_solucao_norm[1]:.3f} ({melhor_solucao_norm[1]:.1%})")
        print(f"   Peixe: {melhor_solucao_norm[2]:.3f} ({melhor_solucao_norm[2]:.1%})")
        
        status_prot = "✅" if resultado['proteina_ok'] else "❌"
        status_calc = "✅" if resultado['calcio_ok'] else "❌"
        
        print(f"   Proteína: {proteina:.3f} {status_prot} (≥{PROTEINA_MIN})")
        print(f"   Cálcio: {calcio:.3f} {status_calc} (≥{CALCIO_MIN})")
    
    return resultados

#%%=============================================================================
# ANÁLISE E VISUALIZAÇÃO
# =============================================================================

def analisar_resultados(solucao_pl, custo_pl, resultados_ag):
    """Analisa e compara os resultados"""
    print("\n" + "=" * 65)
    print(" - ANÁLISE COMPARATIVA")
    print("=" * 65)
    
    custos_ag = [r['custo'] for r in resultados_ag]
    melhor_custo_ag = min(custos_ag)
    pior_custo_ag = max(custos_ag)
    custo_medio_ag = np.mean(custos_ag)
    desvio_padrao_ag = np.std(custos_ag)
    
    print(f"\n - RESUMO DOS CUSTOS:")
    print(f"   Programação Linear (ótimo): ${custo_pl:.4f}")
    print(f"   PyGAD - Melhor:             ${melhor_custo_ag:.4f}")
    print(f"   PyGAD - Médio:              ${custo_medio_ag:.4f} (±{desvio_padrao_ag:.4f})")
    print(f"   PyGAD - Pior:               ${pior_custo_ag:.4f}")
    
    # Calcular desvios
    desvio_melhor = ((melhor_custo_ag - custo_pl) / custo_pl) * 100
    desvio_medio = ((custo_medio_ag - custo_pl) / custo_pl) * 100
    desvio_pior = ((pior_custo_ag - custo_pl) / custo_pl) * 100
    
    print(f"\n - DESVIOS DO ÓTIMO:")
    print(f"   Melhor PyGAD: +{desvio_melhor:.2f}%")
    print(f"   Médio PyGAD:  +{desvio_medio:.2f}%")
    print(f"   Pior PyGAD:   +{desvio_pior:.2f}%")
    
    # Verificar viabilidade das soluções
    solucoes_viaveis = sum(1 for r in resultados_ag if r['proteina_ok'] and r['calcio_ok'])
    taxa_viabilidade = (solucoes_viaveis / len(resultados_ag)) * 100
    
    print(f"\n - VIABILIDADE DAS SOLUÇÕES:")
    print(f"   Programação Linear: 100% (sempre viável)")
    print(f"   PyGAD: {taxa_viabilidade:.0f}% ({solucoes_viaveis}/{len(resultados_ag)} soluções viáveis)")
    
    return {
        'desvio_melhor': desvio_melhor,
        'desvio_medio': desvio_medio,
        'taxa_viabilidade': taxa_viabilidade
    }

def plotar_convergencia(resultados_ag):
    """Plota a convergência do algoritmo genético"""
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Convergência do fitness
    plt.subplot(2, 2, 1)
    for i, resultado in enumerate(resultados_ag):
        convergencia_custo = [1.0/f - 0.001 for f in resultado['convergencia']]
        plt.plot(convergencia_custo, label=f'Execução {i+1}')
    
    plt.axhline(y=custo_pl, color='red', linestyle='--', linewidth=2, label='Ótimo PL')
    plt.xlabel('Geração')
    plt.ylabel('Custo ($)')
    plt.title('Convergência do PyGAD vs Ótimo PL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Distribuição dos custos finais
    plt.subplot(2, 2, 2)
    custos_finais = [r['custo'] for r in resultados_ag]
    plt.hist(custos_finais, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=custo_pl, color='red', linestyle='--', linewidth=2, label='Ótimo PL')
    plt.xlabel('Custo Final ($)')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Resultados PyGAD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Comparação das composições
    plt.subplot(2, 2, (3, 4))
    ingredientes = ['Osso', 'Soja', 'Peixe']
    x_pos = np.arange(len(ingredientes))
    
    # Solução PL
    plt.bar(x_pos - 0.2, solucao_pl, 0.4, label='Programação Linear (Ótimo)', 
            color='red', alpha=0.7)
    
    # Melhor solução AG
    melhor_ag = min(resultados_ag, key=lambda x: x['custo'])
    plt.bar(x_pos + 0.2, melhor_ag['solucao'], 0.4, label='Melhor PyGAD', 
            color='blue', alpha=0.7)
    
    plt.xlabel('Ingredientes')
    plt.ylabel('Proporção (kg)')
    plt.title('Comparação das Composições')
    plt.xticks(x_pos, ingredientes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#%%=============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    try:
        # Resolver com Programação Linear
        solucao_pl, custo_pl = resolver_programacao_linear()
        
        # Resolver com PyGAD
        resultados_ag = executar_pygad(num_execucoes=3)
        
        # Análise comparativa
        estatisticas = analisar_resultados(solucao_pl, custo_pl, resultados_ag)
        
        # Visualização
        plotar_convergencia(resultados_ag)
        
        # Conclusões
        print("\n" + "=" * 65)
        print("🎓 CONCLUSÕES:")
        print("=" * 65)
        
        conclusoes = f"""
🔍 VANTAGENS DO PYGAD:
 * Interface muito simples e intuitiva
 * Muitos parâmetros configuráveis
 * Callbacks para monitoramento
 * Documentação excelente
 * Adequado para problemas complexos não-lineares

1)  LIMITAÇÕES OBSERVADAS NESTE PROBLEMA:
• Desvio médio do ótimo: +{estatisticas['desvio_medio']:.1f}%
• Taxa de soluções viáveis: {estatisticas['taxa_viabilidade']:.0f}%
• Variabilidade entre execuções
• Tempo computacional maior que PL

2) LIÇÃO PRINCIPAL:
PyGAD é uma ferramenta EXCELENTE, mas para problemas 
lineares simples como este, Programação Linear ainda 
é superior em precisão, velocidade e garantia de ótimo.

3) USE PYGAD QUANDO:
• Problema não-linear
• Múltiplos objetivos
• Restrições complexas
• Função objetivo não diferenciável
• Espaço de busca discreto/combinatório

4) USE PROGRAMAÇÃO LINEAR QUANDO:
• Problema linear (como este!)
• Precisa de solução ótima garantida
• Eficiência é importante
• Restrições lineares bem definidas
        """
        
        print(conclusoes)
        
    except ImportError:
        print(" PyGAD não instalado!")
        print(" Para instalar: pip install pygad")