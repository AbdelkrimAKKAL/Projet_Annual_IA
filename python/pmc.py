import ctypes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
# =============================================================
# CHARGEMENT DE LA LIB C
# =============================================================
lib = ctypes.CDLL("../C/pmc.dll")



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

def predire(x):
    # Fait une prediction pour un exemple x
    x_np = np.array(x, dtype=np.float64)
    out  = (ctypes.c_double * 1)()  
    lib.py_predict(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out, 1
    )
    return out[0]


# =============================================================
# CAS DE TEST : XOR
# =============================================================

X = [[0,0], [0,1], [1,0], [1,1]]  # les entrees
Y = [[0],   [1],   [1],   [0]]    # les sorties attendues

# On cree un reseau : 2 entrees, 4 neurones caches, 1 sortie
init(2, 4, 1)

# On entraine le reseau sur 3000 epochs avec un taux d'apprentissage de 0.1
pertes = entrainer(X, Y, epochs=10000, alpha=0.1)

# On affiche les predictions apres entrainement
print("Resultats XOR :")
for x, y in zip(X, Y):
    print(f"  {x} => predit : {predire(x):.3f}  attendu : {y[0]}")

# On trace la courbe d'apprentissage
# Si la courbe descend bien vers 0, le reseau a bien appris
plt.plot(pertes)
plt.title("Courbe d'apprentissage - XOR")
plt.xlabel("Epoch")       # axe horizontal : le numero de l'epoch
plt.ylabel("Erreur")      # axe vertical   : l'erreur moyenne
plt.grid(True)
plt.savefig("../results/courbe_xor.png")
plt.show()


# ============================================================
# CHIENS / CHATS / AUTRES
# 0 = chat, 1 = chien, 2 = autre
# ============================================================
 
TAILLE = 32
 
def charger_images(dossier, label):
    X, Y = [], []
    for fichier in os.listdir(dossier):
        if fichier.endswith(".jpg") or fichier.endswith(".png"):
            img = Image.open(os.path.join(dossier, fichier)).convert("L").resize((TAILLE, TAILLE))
            X.append(np.array(img, dtype=np.float64).flatten() / 255.0)
            Y.append([label])
    return X, Y
 
# Chargement des images d'entrainement
X_chats,  Y_chats  = charger_images("../dataset/train_dataset/chats",  0)
X_chiens, Y_chiens = charger_images("../dataset/train_dataset/chiens", 1)
X_autres, Y_autres = charger_images("../dataset/train_dataset/autres", 2)
 
X_train = X_chats  + X_chiens  + X_autres
Y_train = Y_chats  + Y_chiens  + Y_autres
 
print(f"\nDataset : {len(X_train)} images ({len(X_chats)} chats, {len(X_chiens)} chiens, {len(X_autres)} autres)")
 
# Entrainement : 1024 entrees, 16 neurones caches, 1 sortie
init(TAILLE * TAILLE, 16, 1)
pertes = entrainer(X_train, Y_train, epochs=2000, alpha=0.01)
 
plt.figure()
plt.plot(pertes)
plt.title("Courbe d'apprentissage - Chiens/Chats/Autres")
plt.xlabel("Epoch")
plt.ylabel("Erreur")
plt.grid(True)
plt.savefig("../results/courbe_chiens_chats.png")
plt.show()
 
# Test sur les images que le réseau n'a jamais vues
X_chats_test,  Y_chats_test  = charger_images("../dataset/test_dataset/chats",  0)
X_chiens_test, Y_chiens_test = charger_images("../dataset/test_dataset/chiens", 1)
X_autres_test, Y_autres_test = charger_images("../dataset/test_dataset/autres", 2)
 
X_test = X_chats_test  + X_chiens_test  + X_autres_test
Y_test = Y_chats_test  + Y_chiens_test  + Y_autres_test
 
# Correspondance entre la valeur et le nom de la classe
noms = ["chat", "chien", "autre"]
 
print("\nResultats sur les images de test :")
nb_correct = 0
for x, y in zip(X_test, Y_test):
    pred     = predire(x)
    predit   = noms[min(round(pred), 2)]  # on arrondit et on limite entre 0 et 2
    attendu  = noms[int(y[0])]
    resultat = "OK" if predit == attendu else "ERREUR"
    print(f"  attendu : {attendu} => predit : {predit}  {resultat}")
    if predit == attendu:
        nb_correct += 1
 
print(f"\nPrecision : {nb_correct}/{len(X_test)} ({100*nb_correct/len(X_test):.1f}%)")