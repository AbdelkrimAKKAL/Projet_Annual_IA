import os
import matplotlib.pyplot as plt

from pmc import init, entrainer, predire, SCRIPT_DIR

# =============================================================
# CAS DE TEST : XOR
# =============================================================

X = [[0,0], [0,1], [1,0], [1,1]]   # les entrees
Y = [[-1],  [1],   [1],   [-1]]    # cibles -1 / +1 (Tanh)

# On cree un reseau : 2 entrees, 4 neurones caches, 1 sortie
init(2, 4, 1)

# On entraine le reseau sur 10000 epochs avec un taux d'apprentissage de 0.1
pertes = entrainer(X, Y, epochs=10000, alpha=0.1)

# On affiche les predictions apres entrainement (decision par le signe)
print("Resultats XOR :")
for x, y in zip(X, Y):
    s = predire(x, 1)[0]
    classe = 1 if s >= 0 else -1
    print(f"  {x} => predit : {s:+.3f} (classe {classe})  attendu : {y[0]}")

# On trace la courbe d'apprentissage
# Si la courbe descend bien vers 0, le reseau a bien appris
plt.plot(pertes)
plt.title("Courbe d'apprentissage - XOR")
plt.xlabel("Epoch")       # axe horizontal : le numero de l'epoch
plt.ylabel("Erreur")      # axe vertical   : l'erreur moyenne
plt.grid(True)
plt.savefig(os.path.join(SCRIPT_DIR, "..", "results", "courbe_xor.png"))
plt.show()
