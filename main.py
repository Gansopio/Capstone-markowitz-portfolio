import os
import glob
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import frontera_eficiente

import leer_y_cargar
import limpieza
import modelo_markowitz
from testing import testing

carpeta = r"C:\Users\Gustavo\Documents\U\Semestre Actual\Capstone\universo_300_acciones\universo_300_acciones"

returns = leer_y_cargar.leer_archivos(carpeta)
returns = limpieza.limpiar_datos(returns)

train_days = 252 * 4   
test_days = 252        

train_returns = returns.iloc[:train_days]
test_returns = returns.iloc[train_days:train_days + test_days]

R_objetivo_anual = 0.40  
perfil = "muy agresivo"
lam = 4.0

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