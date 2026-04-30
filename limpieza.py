import pandas as pd
import numpy as np  

def limpiar_datos(returns):
    # 🚨 Eliminamos acciones con datos irreales / no invertibles
    acciones_a_eliminar = [
        "stock_return_NCPL",
        "stock_return_PROP",
        "stock_return_ABTS"
    ]
    
    returns = returns.drop(columns=acciones_a_eliminar, errors="ignore")
    
    return returns