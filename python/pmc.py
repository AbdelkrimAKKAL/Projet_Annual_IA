import ctypes
import os
import numpy as np
import matplotlib.pyplot as plt

from functions import load_dataset

# =============================================================
# CHARGEMENT DE LA LIB C
# =============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(SCRIPT_DIR, "..", "C", "pmc.dll"))


lib.py_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.py_init_regression.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.py_train.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # X : les entrees
    ctypes.POINTER(ctypes.c_double),  # Y : les cibles
    ctypes.c_int,                     # nb_exemples
    ctypes.c_int,                     # nb_entrees par exemple
    ctypes.c_int,                     # nb_sorties par exemple
    ctypes.c_int,                     # nombre d'epochs
    ctypes.c_double,                  # alpha (taux d'apprentissage)
    ctypes.POINTER(ctypes.c_double),  # pertes (tableau resultat)
]

lib.py_predict.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # entrees
    ctypes.POINTER(ctypes.c_double),  # sorties (resultat)
    ctypes.c_int,                     # nb_sorties
]

lib.py_sauvegarder.argtypes = [ctypes.c_char_p]
lib.py_charger.argtypes     = [ctypes.c_char_p]


# =============================================================
# FONCTIONS PYTHON QUI APPELLENT LA LIB C
# =============================================================

def init(nb_entrees, nb_cachees, nb_sorties):
    # Cree le reseau avec les tailles de couches donnees (sortie tanh, classification)
    lib.py_init(nb_entrees, nb_cachees, nb_sorties)

def init_regression(nb_entrees, nb_cachees, nb_sorties):
    # Meme reseau, mais sortie lineaire (pas de tanh) pour predire des valeurs reelles
    lib.py_init_regression(nb_entrees, nb_cachees, nb_sorties)

def entrainer(X, Y, epochs=2000, alpha=0.1):
    # Convertit les listes Python en tableaux numpy de float64
    X_np = np.array(X, dtype=np.float64)
    Y_np = np.array(Y, dtype=np.float64)

    nb_exemples = X_np.shape[0]  # nombre de lignes = nombre d'exemples
    nb_entrees  = X_np.shape[1]  # nombre de colonnes = nombre d'entrees
    nb_sorties  = Y_np.shape[1]  # nombre de colonnes = nombre de sorties

    # Tableau qui recevra l'erreur moyenne a chaque epoch
    pertes = (ctypes.c_double * epochs)()

    lib.py_train(
        X_np.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y_np.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nb_exemples, nb_entrees, nb_sorties, epochs, alpha, pertes
    )
    return list(pertes)

def predire(x, nb_sorties=1):
    # Fait une prediction pour un exemple x ; renvoie la liste des sorties
    x_np = np.array(x, dtype=np.float64)
    out  = (ctypes.c_double * nb_sorties)()
    lib.py_predict(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out, nb_sorties
    )
    return list(out)

