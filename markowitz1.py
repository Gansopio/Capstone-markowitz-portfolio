import os
import glob
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# 1. Leer archivos
# ============================================================

carpeta = r"C:\Users\Gustavo\Documents\U\Semestre Actual\Capstone\universo_300_acciones\universo_300_acciones"

print("Carpeta existe:", os.path.exists(carpeta))
print("Carpeta:", carpeta)

archivos = glob.glob(os.path.join(carpeta, "*.csv"))

print("Archivos encontrados:", len(archivos))
print("Primeros archivos:", archivos[:5])

if len(archivos) == 0:
    raise ValueError("No se encontraron CSV. Revisa la ruta.")

lista_retornos = []

for archivo in archivos:
    ticker = os.path.splitext(os.path.basename(archivo))[0]

    df = pd.read_csv(archivo)

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df["Date"] = df["Date"].dt.date
    df = df.sort_values("Date")
    df = df.set_index("Date")

    close = df["Close"]
    dividends = df["Dividends"]

    # Retorno total
    retorno = (close - close.shift(1) + dividends) / close.shift(1)

    retorno.name = ticker
    lista_retornos.append(retorno)

# ============================================================
# 2. Matriz de retornos
# ============================================================

returns = pd.concat(lista_retornos, axis=1)
returns = returns.dropna()

print("Dimensión total:", returns.shape)

# ============================================================
# Separar última fecha como testing
# ============================================================

train_returns = returns.iloc[:-1]
test_returns = returns.iloc[-1:]

print("Dimensión train:", train_returns.shape)
print("Dimensión test:", test_returns.shape)
print("Fecha de test:", test_returns.index[0])


# ============================================================
# 3. Parámetros Markowitz
# ============================================================

mu = train_returns.mean().values
Sigma = train_returns.cov().values
tickers = list(train_returns.columns)

n = len(tickers)

# ============================================================
# 4. Modelo Gurobi
# ============================================================

model = gp.Model("Markowitz_simple")

# Variables (pesos)
w = model.addVars(n, lb=0, name="w")

# Restricción: suma de pesos = 1
model.addConstr(
    gp.quicksum(w[i] for i in range(n)) == 1,
    name="presupuesto"
)

# Varianza del portafolio
portfolio_variance = gp.quicksum(
    w[i] * Sigma[i, j] * w[j]
    for i in range(n)
    for j in range(n)
)

# Función objetivo: minimizar riesgo
model.setObjective(portfolio_variance, GRB.MINIMIZE)

# ============================================================
# 5. Resolver
# ============================================================

model.optimize()

# ============================================================
# 6. Resultados
# ============================================================

if model.status == GRB.OPTIMAL:

    pesos = np.array([w[i].X for i in range(n)])

    retorno_esperado = np.dot(mu, pesos)
    varianza = pesos @ Sigma @ pesos
    volatilidad = np.sqrt(varianza)
    

    resultado = pd.DataFrame({
        "Ticker": tickers,
        "Peso": pesos
    })

    resultado = resultado[resultado["Peso"] > 1e-6]
    resultado = resultado.sort_values("Peso", ascending=False)

    print("\n📊 PORTAFOLIO ÓPTIMO")
    print(resultado)

    print("\n📈 Retorno esperado:", retorno_esperado)
    print("📉 Varianza:", varianza)
    print("📊 Volatilidad:", volatilidad)
    # Resultado en la última fecha dejada fuera
    retorno_test = float(test_returns.values.flatten() @ pesos)

    print("\n🧪 TESTING ÚLTIMA FECHA")
    print("Fecha test:", test_returns.index[0])
    print("Retorno real del portafolio en test:", retorno_test)

else:
    print("No se encontró solución óptima")