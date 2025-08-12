#%%=============================================================================
# OPÇÃO 1: PULP 
# =============================================================================
print("\n - OPÇÃO 1: PULP")

#Instalação: pip install pulp

import pulp
    
#%% # Criar problema de minimização
prob = pulp.LpProblem("Racao_Animal", pulp.LpMinimize)
    
# Variáveis de decisão
x1 = pulp.LpVariable("Osso", lowBound=0)
x2 = pulp.LpVariable("Soja", lowBound=0) 
x3 = pulp.LpVariable("Peixe", lowBound=0)
    
#%% # Função objetivo
prob += 0.56*x1 + 0.81*x2 + 0.46*x3, "Custo_Total"
    
# Restrições 
prob += 0.2*x1 + 0.5*x2 + 0.4*x3 >= 0.3, "Proteina_Minima"
prob += 0.6*x1 + 0.4*x2 + 0.4*x3 >= 0.5, "Calcio_Minimo"
prob += x1 + x2 + x3 == 1, "Soma_Ingredientes"
    
# Resolver
prob.solve(pulp.PULP_CBC_CMD(msg=0))

#%% #--> Apresentar os resultados    
print(" - PULP - RESULTADO:")
print(f"Status: {pulp.LpStatus[prob.status]}")
print(f"Custo mínimo: ${pulp.value(prob.objective):.3f}")
print(f"Osso: {x1.varValue:.3f} kg ({x1.varValue:.1%})")
print(f"Soja: {x2.varValue:.3f} kg ({x2.varValue:.1%})")
print(f"Peixe: {x3.varValue:.3f} kg ({x3.varValue:.1%})")
    

