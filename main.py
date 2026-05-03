import os
import glob
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


import backtesting_mensual
import leer_y_cargar
import limpieza
import modelo_markowitz
import perfiles
from testing import testing
import frontera_eficiente

carpeta = r"C:\Users\Gustavo\Documents\U\Semestre Actual\Capstone\universo_300_acciones\universo_300_acciones"

returns = leer_y_cargar.leer_archivos(carpeta)
returns = limpieza.limpiar_datos(returns)

train_days = 252 * 4   
test_days = 252        

train_returns = returns.iloc[:train_days]
test_returns = returns.iloc[train_days:train_days + test_days]

perfil, R_objetivo_anual, lam = perfiles.seleccionar_perfil()

model, w = modelo_markowitz.model_markowitz(
    train_returns,
    lam,
    R_objetivo_anual,
    perfil
)

#resultado_test = testing(
#    model,
#    w,
#    train_returns,
#    test_returns,
#    perfil,
#    R_objetivo_anual
#)


resultado_mensual = backtesting_mensual.backtesting_mensual_con_decision(
    returns,
    perfil,
    R_objetivo_anual,
    lam,
    train_days=252*4,
    test_months=12
)

print(resultado_mensual)
# print(resultado_test)

#frontera_eficiente.graficar_frontera_eficiente(train_returns)