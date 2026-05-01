import os
import glob
import pandas as pd

def leer_archivos(carpeta):
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

        # Retorno diario total
        retorno_diario = (close - close.shift(1) + dividends) / close.shift(1)

        retorno_diario.name = ticker
        lista_retornos.append(retorno_diario)

    returns = pd.concat(lista_retornos, axis=1)
    returns = returns.dropna()

    return returns