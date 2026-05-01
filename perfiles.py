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

def seleccionar_perfil():

    print("\nPerfiles disponibles:")

    for p in perfiles:
        print("-", p)

    perfil = input("\nIngrese perfil de usuario: ").lower().strip()

    if perfil not in perfiles:
        raise ValueError(
            "Perfil no válido. Debe ser: "
            + ", ".join(perfiles.keys())
        )

    perdida_max_anual = perfiles[perfil]["perdida_max_anual"]
    lam = perfiles[perfil]["lambda"]

    print("\nPerfil seleccionado:", perfil)
    print("Pérdida máxima anual aceptada:", perdida_max_anual)
    print("Lambda asignado:", lam)

    return perfil, perdida_max_anual, lam