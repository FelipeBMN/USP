#%%=============================================================================
# OPÇÃO 2: CVXPY 
#  =============================================================================
print("\n - OPÇÃO 2: CVXPY")
#Instalação: pip install cvxpy

import cvxpy as cp
import numpy as np
    
#%% Variáveis de decisão
x = cp.Variable(3, nonneg=True)  # x[0]=osso, x[1]=soja, x[2]=peixe
    
# Função objetivo
custos = np.array([0.56, 0.81, 0.46])
objetivo = cp.Minimize(custos @ x)
    
#%% Restrições 
restricoes = [
np.array([0.2, 0.5, 0.4]) @ x >= 0.3,  # Proteína
np.array([0.6, 0.4, 0.4]) @ x >= 0.5,  # Cálcio
        cp.sum(x) == 1                          # Soma = 1
        ]
    
#%% Criar e resolver problema
prob = cp.Problem(objetivo, restricoes)
prob.solve()

#%% Apresentar os resultados    
print("✅ CVXPY - RESULTADO:")
print(f"Status: {prob.status}")
print(f"Custo mínimo: ${prob.value:.3f}")
print(f"Osso: {x.value[0]:.3f} kg ({x.value[0]:.1%})")
print(f"Soja: {x.value[1]:.3f} kg ({x.value[1]:.1%})")
print(f"Peixe: {x.value[2]:.3f} kg ({x.value[2]:.1%})")

