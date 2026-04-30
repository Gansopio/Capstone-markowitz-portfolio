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

# ============================================================
# 4. Modelo Gurobi por perfiles
# ============================================================

retornos_objetivo_anual = {
    "muy_conservador": 0.00,
    "conservador": 0.05,
    "neutro": 0.10,
    "arriesgado": 0.15,
    "muy_arriesgado": 0.20
}

resumen = []

for perfil, R_objetivo_anual in retornos_objetivo_anual.items():

    R_objetivo_diario = R_objetivo_anual / 252

    model = gp.Model(f"Markowitz_{perfil}")
    model.Params.OutputFlag = 0

    w = model.addVars(n, lb=0, name="w")

    model.addConstr(
        gp.quicksum(w[i] for i in range(n)) == 1,
        name="presupuesto"
    )

    portfolio_return = gp.quicksum(
        mu[i] * w[i]
        for i in range(n)
    )

    model.addConstr(
        portfolio_return >= -R_objetivo_diario,
        name="retorno_minimo_perfil"
    )

    portfolio_variance = gp.quicksum(
        w[i] * Sigma[i, j] * w[j]
        for i in range(n)
        for j in range(n)
    )

    model.setObjective(portfolio_variance, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:

        pesos = np.array([w[i].X for i in range(n)])

        retorno_esperado = np.dot(mu, pesos)
        varianza = pesos @ Sigma @ pesos
        volatilidad = np.sqrt(varianza)

        retorno_test = float(test_returns.to_numpy().flatten() @ pesos)

        resultado = pd.DataFrame({
            "Ticker": tickers,
            "Peso": pesos
        })

        resultado = resultado[resultado["Peso"] > 1e-6]
        resultado = resultado.sort_values("Peso", ascending=False)

        print("\n" + "="*80)
        print(f"👤 PERFIL: {perfil}")
        print("="*80)

        print("\n📊 PORTAFOLIO ÓPTIMO")
        print(resultado)

        print("\n🎯 Retorno objetivo anual:", R_objetivo_anual)
        print("🎯 Retorno objetivo diario:", R_objetivo_diario)

        print("\n📈 Retorno esperado train:", retorno_esperado)
        print("📉 Varianza train:", varianza)
        print("📊 Volatilidad train:", volatilidad)

        print("\n🧪 TESTING ÚLTIMA FECHA")
        print("Fecha test:", test_returns.index[0])
        print("Retorno real test:", retorno_test)

        resumen.append({
            "perfil": perfil,
            "retorno_objetivo_anual": R_objetivo_anual,
            "retorno_objetivo_diario": R_objetivo_diario,
            "retorno_esperado_train": retorno_esperado,
            "varianza_train": varianza,
            "volatilidad_train": volatilidad,
            "fecha_test": test_returns.index[0],
            "retorno_test": retorno_test,
            "num_activos_usados": np.sum(pesos > 1e-6),
            "peso_maximo": pesos.max()
        })

    else:
        print(f"\n❌ No se encontró solución óptima para perfil: {perfil}")

        resumen.append({
            "perfil": perfil,
            "retorno_objetivo_anual": R_objetivo_anual,
            "error": "No óptimo"
        })

# ============================================================
# 5. Resumen final
# ============================================================

resumen_df = pd.DataFrame(resumen)

print("\n\n📌 RESUMEN POR PERFIL")
print(resumen_df)

resumen_df.to_csv("resumen_perfiles_markowitz.csv", index=False)
print("\nArchivo guardado: resumen_perfiles_markowitz.csv")