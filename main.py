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

carpeta = r"C:\Users\Gustavo\Documents\U\Semestre Actual\Capstone\universo_300_acciones\universo_300_acciones"
returns = leer_y_cargar.leer_archivos(carpeta)
returns = limpieza.limpiar_datos(returns)

train_returns = returns.iloc[:-504]
test_returns = returns.iloc[-504:]

R_objetivo_anual = 0.05
perfil = "conservador"
model, w = modelo_markowitz.model_markowitz(train_returns, R_objetivo_anual, perfil)

resultado_test = testing(
    model,
    w,
    train_returns,
    test_returns,
    perfil,
    R_objetivo_anual
)

print(resultado_test)

