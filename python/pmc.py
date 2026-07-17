import ctypes
import os
import numpy as np

from functions import melanger, convertir_tab_c

# =============================================================
# CHARGEMENT DE LA LIB C
# =============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(SCRIPT_DIR, "..", "C", "pmc.dll"))

lib.py_init.argtypes            = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.py_init_regression.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.py_train.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # X
    ctypes.POINTER(ctypes.c_double),  # Y
    ctypes.c_int,                     # nb_exemples
    ctypes.c_int,                     # nb_entrees
    ctypes.c_int,                     # nb_sorties
    ctypes.c_int,                     # epochs
    ctypes.c_double,                  # alpha
    ctypes.POINTER(ctypes.c_double),  # pertes
]

lib.py_train_one.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # entree
    ctypes.POINTER(ctypes.c_double),  # cible
    ctypes.c_double,                  # alpha
]

lib.py_predict.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # entrees
    ctypes.POINTER(ctypes.c_double),  # sorties
    ctypes.c_int,                     # nb_sorties
]

lib.py_sauvegarder.argtypes = [ctypes.c_char_p]
lib.py_charger.argtypes     = [ctypes.c_char_p]


# =============================================================
# FONCTIONS PYTHON QUI APPELLENT LA LIB C
# =============================================================

def init(nb_entrees, nb_cachees, nb_sorties):
    """Initialise le reseau (classification, sortie tanh)."""
    lib.py_init(nb_entrees, nb_cachees, nb_sorties)

def init_regression(nb_entrees, nb_cachees, nb_sorties):
    """Initialise le reseau (regression, sortie lineaire)."""
    lib.py_init_regression(nb_entrees, nb_cachees, nb_sorties)

def entrainer(X, Y, epochs=2000, alpha=0.1, shuffle=True, seed=42, X_test=None, Y_test=None):
    """
    Entraine le reseau.
    - Si X_test est fourni : entraine epoch par epoch via py_train_one et retourne
      (hist_train, hist_test, err_train, err_test) comme model_linear.py
    - Sinon : entraine tout d'un coup via py_train et retourne les pertes
    """
    if X_test is not None:
        # Entrainement epoch par epoch avec suivi historique
        if shuffle:
            X, Y = melanger(X, Y, seed)

        nb_sorties = len(Y[0])
        hist_train, hist_test = [], []
        err_train,  err_test  = [], []

        for _ in range(epochs):
            for x, y in zip(X, Y):
                c_x = convertir_tab_c(x)
                c_y = (ctypes.c_double * nb_sorties)(*y)
                lib.py_train_one(c_x, c_y, alpha)

            hist_train.append(precision_pmc(X, Y, nb_sorties))
            hist_test.append(precision_pmc(X_test, Y_test, nb_sorties))
            err_train.append(100 - hist_train[-1])
            err_test.append(100 - hist_test[-1])

        return hist_train, hist_test, err_train, err_test

    else:
        # Entrainement tout d'un coup (rapide, pour les runs sans historique)
        X_np = np.array(X, dtype=np.float64)
        Y_np = np.array(Y, dtype=np.float64)
        nb_ex  = X_np.shape[0]
        nb_in  = X_np.shape[1]
        nb_out = Y_np.shape[1]
        pertes = (ctypes.c_double * epochs)()
        lib.py_train(
            X_np.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Y_np.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nb_ex, nb_in, nb_out, epochs, alpha, pertes
        )
        return list(pertes)


def predire(x, nb_sorties=1):
    """Retourne le vecteur de sortie du reseau pour un exemple x."""
    x_np = np.array(x, dtype=np.float64)
    out  = (ctypes.c_double * nb_sorties)()
    lib.py_predict(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out, nb_sorties
    )
    return list(out)


def precision_pmc(X, Y, nb_sorties):
    """Pourcentage de bonnes predictions (classification, argmax)."""
    ok = sum(
        int(np.argmax(predire(x, nb_sorties))) == int(np.argmax(y))
        for x, y in zip(X, Y)
    )
    return ok / len(X) * 100
