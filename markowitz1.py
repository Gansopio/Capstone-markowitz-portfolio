import os
import glob
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

#### Lee archivos ####

carpeta = r"C:\Users\Gustavo\Documents\U\Semestre Actual\Capstone\universo_300_acciones\universo_300_acciones"

archivos = glob.glob(os.path.join(carpeta, "*.csv"))

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

    retorno = (close - close.shift(1) + dividends) / close.shift(1)

    retorno.name = ticker
    lista_retornos.append(retorno)


##### Matriz de retornos ####

returns = pd.concat(lista_retornos, axis=1)
returns = returns.dropna()

# 🚨 Eliminamos acciones con datos irreales / no invertibles
acciones_a_eliminar = [
    "stock_return_NCPL",
    "stock_return_PROP",
    "stock_return_ABTS"
]

returns = returns.drop(columns=acciones_a_eliminar, errors="ignore")

print("Dimensión total después de limpiar:", returns.shape)

print("Dimensión total:", returns.shape)


### Testing último año ###

lambdas = [0.1,0.25,0.5,0.75,1,2, 2.5, 5,7.5, 10, 20, 50, 100, 200, 300, 400, 500, 1000]
test_days = 252
min_train_days = 1000

resumen = []

# Ventanas anuales: 2021, 2022, 2023, etc. según tu data
for test_start in range(min_train_days, len(returns) - test_days + 1, test_days):

    train_returns = returns.iloc[:test_start]
    test_returns = returns.iloc[test_start:test_start + test_days]

    print("\n" + "="*80)
    print("VENTANA TEST")
    print("Train:", train_returns.index[0], "a", train_returns.index[-1])
    print("Test :", test_returns.index[0], "a", test_returns.index[-1])
    print("="*80)

    mu = train_returns.mean().values
    Sigma = train_returns.cov().values
    tickers = list(train_returns.columns)
    n = len(tickers)

    for lam in lambdas:

        model = gp.Model(f"Markowitz_lambda_{lam}")
        model.Params.OutputFlag = 0  # apagar log de Gurobi

        w = model.addVars(n, lb=0, name="w")

        model.addConstr(
            gp.quicksum(w[i] for i in range(n)) == 1,
            name="presupuesto"
        )

        portfolio_return = gp.quicksum(mu[i] * w[i] for i in range(n))

        portfolio_variance = gp.quicksum(
            w[i] * Sigma[i, j] * w[j]
            for i in range(n)
            for j in range(n)
        )

        model.setObjective(
            portfolio_return - lam * portfolio_variance,
            GRB.MAXIMIZE
        )

        model.optimize()

        if model.status == GRB.OPTIMAL:

            pesos = np.array([w[i].X for i in range(n)])

            retorno_esperado = np.dot(mu, pesos)
            varianza_train = pesos @ Sigma @ pesos
            volatilidad_train = np.sqrt(varianza_train)

            portfolio_test_returns = test_returns @ pesos

            retorno_acumulado_test = (1 + portfolio_test_returns).prod() - 1
            retorno_promedio_diario_test = portfolio_test_returns.mean()
            volatilidad_diaria_test = portfolio_test_returns.std()
            volatilidad_anual_test = volatilidad_diaria_test * np.sqrt(252)

            resumen.append({
                "lambda": lam,
                "train_inicio": train_returns.index[0],
                "train_fin": train_returns.index[-1],
                "test_inicio": test_returns.index[0],
                "test_fin": test_returns.index[-1],
                "retorno_esperado_train": retorno_esperado,
                "volatilidad_train": volatilidad_train,
                "retorno_acumulado_test": retorno_acumulado_test,
                "retorno_promedio_diario_test": retorno_promedio_diario_test,
                "volatilidad_anual_test": volatilidad_anual_test,
                "num_activos_usados": np.sum(pesos > 1e-6),
                "peso_maximo": pesos.max()
            })

        else:
            resumen.append({
                "lambda": lam,
                "train_inicio": train_returns.index[0],
                "train_fin": train_returns.index[-1],
                "test_inicio": test_returns.index[0],
                "test_fin": test_returns.index[-1],
                "error": "No óptimo"
            })

# ============================================================
# Resultados finales
# ============================================================

resumen_df = pd.DataFrame(resumen)

print("\n\n📊 RESUMEN COMPLETO")
print(resumen_df)

print("\n📈 PROMEDIO POR LAMBDA")
print(
    resumen_df.groupby("lambda")[[
        "retorno_acumulado_test",
        "retorno_promedio_diario_test",
        "volatilidad_anual_test",
        "num_activos_usados",
        "peso_maximo"
    ]].mean()
)

resumen_df.to_csv("resumen_backtesting_markowitz.csv", index=False)
print("\nArchivo guardado: resumen_backtesting_markowitz.csv")
print(returns.max().sort_values(ascending=False).head(10))
print("\n🔴 PEORES RETORNOS (mínimos)")
print(returns.min().sort_values().head(10))
