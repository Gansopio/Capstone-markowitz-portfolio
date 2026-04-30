import os
import pandas as pd

carpeta = r"C:\Users\Gustavo\Documents\U\Semestre Actual\Capstone\universo_300_acciones\universo_300_acciones"

acciones = ["stock_return_PROP.csv", "stock_return_ABTS.csv"]

for accion in acciones:
    archivo = os.path.join(carpeta, accion)

    df = pd.read_csv(archivo)

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df["Date"] = df["Date"].dt.date
    df = df.sort_values("Date")

    df["ret"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)

    print("\n" + "=" * 80)
    print(f"ACCIÓN: {accion}")
    print("=" * 80)

    print("\n🟢 Mayores retornos")
    print(df.sort_values("ret", ascending=False).head(10))

    print("\n🔴 Peores retornos")
    print(df.sort_values("ret", ascending=True).head(10))