# Projet Annuel – Machine Learning 3ème année Big Data
> Classification d'images : Chat / Chien / Autre  
> Année: 2025-2026 | Encadrant : M.Nicolas VIDAL

---

## Équipe
YEH Shun
MARTINEZ Maxime
AKKAL Abdelkrim
---


---

##  Modèles implémentés

| Modèle | Description |
|--------|-------------|
| Modèle Linéaire  | Régression/classification linéaire |
| Perceptron Multi-Couches (MLP) | Réseau de neurones fully connected |
| Radial Basis Function (RBF)  | Réseau à fonctions à base radiale |
| SVM | Machine à vecteurs de support |

---

##  Dataset

- **Problématique** : Classification d'images en 3 classes — Chat  / Chien  / Autre
- **Source** : [Kaggle – Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)
- **Taille** : ~25 000 images (train) + ~12 500 images (test)
- **Prétraitement** : redimensionnement 64×64, normalisation, augmentation de données

---

## Résultats

> *(Section à compléter au fur et à mesure du projet)*

- Notebook interactif : [`python/notebooks/experiments.ipynb`](python/notebooks/experiments.ipynb)
- Phénomènes mis en évidence : sous-apprentissage, sur-apprentissage, impact des hyperparamètres

---

## Conventions Git

### Branches
```
main          → version stable
dev           → branche de développement principale
feat/XXX      → nouvelle fonctionnalité (ex: feat/mlp)
fix/XXX       → correction de bug
```

### En-tête obligatoire dans chaque fichier source
```cpp
/**
 * @file    linear_model.cpp
 * @author  Prénom NOM <email@myges.fr>
 * @date    2026-01-12
 * @brief   Implémentation du modèle linéaire
 */
```

---

## 📅 Étapes du projet


✅ Étape 1 | 12/01/2026 | Problématique choisie, repo Git créé, dataset identifié |
⬜ Étape 2 | 06/04/2026 | Modèle linéaire + MLP + début application |
⬜ Étape 3 | (provisoire) | RBF + SVM + système client/serveur |
⬜ Rendu final | 17/06/2026 | Rapport complet + notebook + sources |
⬜ Soutenance | 20/07/2026 | Présentation publique (30 min) |

---

## 📚 Références

- [Learning From Data – Caltech MOOC](https://work.caltech.edu/telecourse.html)
- [Kaggle Datasets](https://www.kaggle.com/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets.php)

---
