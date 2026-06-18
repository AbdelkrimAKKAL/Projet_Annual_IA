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
    # Cree le reseau avec les tailles de couches donnees
    lib.py_init(nb_entrees, nb_cachees, nb_sorties)

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


# ============================================================
# CHIENS / CHATS / AUTRES
# 0 = chat, 1 = chien, 2 = autre
# ============================================================

if __name__ == "__main__":
    TAILLE = 32
    NB_CLASSES = 3

    # Chargement des images d'entrainement
    X_train, Y_train = load_dataset(os.path.join(SCRIPT_DIR, "..", "dataset", "train_dataset"), target_size=(TAILLE, TAILLE), color=True, one_hot=True)

    perm = np.random.permutation(len(X_train))
    X_train = [X_train[i] for i in perm]
    Y_train = [Y_train[i] for i in perm]

    print(f"\nDataset : {len(X_train)} images")

    # Entrainement : RGB 32x32x3 = 3072 entrees, 16 neurones caches, 3 sorties
    init(TAILLE * TAILLE * 3, 16, NB_CLASSES)
    pertes = entrainer(X_train, Y_train, epochs=800, alpha=0.01)

    plt.figure()
    plt.plot(pertes)
    plt.title("Courbe d'apprentissage - Chiens/Chats/Autres")
    plt.xlabel("Epoch")
    plt.ylabel("Erreur")
    plt.grid(True)
    plt.savefig(os.path.join(SCRIPT_DIR, "..", "results", "courbe_chiens_chats.png"))
    plt.show()

    # Precision sur l'ENTRAINEMENT
    nb_correct_train = 0
    for x, y in zip(X_train, Y_train):
        sorties = predire(x, NB_CLASSES)
        if int(np.argmax(sorties)) == int(np.argmax(y)):
            nb_correct_train += 1
    print(f"\nPrecision entrainement : {nb_correct_train}/{len(X_train)} ({100*nb_correct_train/len(X_train):.1f}%)")

    # Test
    X_test, Y_test = load_dataset(os.path.join(SCRIPT_DIR, "..", "dataset", "test_dataset"), target_size=(TAILLE, TAILLE), color=True, one_hot=True)

    noms = ["chat", "chien", "autre"]

    print("\nResultats sur les images de test :")
    nb_correct = 0
    for x, y in zip(X_test, Y_test):
        sorties  = predire(x, NB_CLASSES)        # 3 sorties
        predit   = noms[int(np.argmax(sorties))]  # classe = sortie la plus forte
        attendu  = noms[int(np.argmax(y))]        # classe attendue (one-hot)
        resultat = "OK" if predit == attendu else "ERREUR"
        print(f"  attendu : {attendu} => predit : {predit}  {resultat}")
        if predit == attendu:
            nb_correct += 1

    print(f"\nPrecision : {nb_correct}/{len(X_test)} ({100*nb_correct/len(X_test):.1f}%)")
