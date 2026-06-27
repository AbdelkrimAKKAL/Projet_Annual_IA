# Projet Annuel IA

Abdelkrim AKKAL | Maxime MARTINEZ | Shun YEH 


# Sujet

Classification d'images chat / chien / autre

## Structure du projet

- `C/` — implémentations des modèles (PMC, modèle linéaire) compilées en DLL
- `python/` — scripts d'entraînement et utilitaires
- `jupyter/` — notebooks d'expérimentation par modèle
- `server/` — API Flask 
- `front/` — interface web simple 
- `dataset/` — images d'entraînement et de test (chats, chiens, autres)
-

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Lancer l'API

```bash
python server/app.py
```

L'API démarre sur `http://localhost:5000`.

