import os
import glob
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


import leer_y_cargar
import limpieza
import modelo_markowitz
from testing import testing
import frontera_eficiente

carpeta = r"C:\Users\Gustavo\Documents\U\Semestre Actual\Capstone\universo_300_acciones\universo_300_acciones"

returns = leer_y_cargar.leer_archivos(carpeta)
returns = limpieza.limpiar_datos(returns)

train_days = 252 * 4   
test_days = 252        

train_returns = returns.iloc[:train_days]
test_returns = returns.iloc[train_days:train_days + test_days]

perfiles = {
    "muy conservador": {
        "perdida_max_anual": 0.00,
        "lambda": 100000
    },
    "conservador": {
        "perdida_max_anual": 0.05,
        "lambda": 100
    },
    "neutro": {
        "perdida_max_anual": 0.15,
        "lambda": 20
    },
    "arriesgado": {
        "perdida_max_anual": 0.30,
        "lambda": 7.6
    },
    "muy arriesgado": {
        "perdida_max_anual": 0.40,
        "lambda": 1.3
    }
}

print("\nPerfiles disponibles:")
for p in perfiles:
    print("-", p)

perfil = input("\nIngrese perfil de usuario: ").lower().strip()

if perfil not in perfiles:
    raise ValueError("Perfil no válido. Debe ser: " + ", ".join(perfiles.keys()))

R_objetivo_anual = perfiles[perfil]["perdida_max_anual"]
lam = perfiles[perfil]["lambda"]

print("\nPerfil seleccionado:", perfil)
print("Pérdida máxima anual aceptada:", R_objetivo_anual)
print("Lambda asignado:", lam)

model, w = modelo_markowitz.model_markowitz(
    train_returns,
    lam,
    R_objetivo_anual,
    perfil
)

resultado_test = testing(
    model,
    w,
    train_returns,
    test_returns,
    perfil,
    R_objetivo_anual
)

print(resultado_test)

frontera_eficiente.graficar_frontera_eficiente(train_returns)