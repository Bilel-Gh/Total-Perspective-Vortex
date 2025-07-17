# 🧠 Total-Perspective-Vortex

Interface Cerveau-Ordinateur (BCI) utilisant l'électroencéphalographie (EEG) et l'apprentissage automatique pour décoder l'activité cérébrale liée à l'imagerie motrice.

## 🎯 Objectif

Analyser les signaux EEG pour détecter et classifier les intentions de mouvement moteur :
- **Imagerie motrice** du poing gauche/droit
- **Imagerie motrice** des deux poings/deux pieds

Le système vise une précision >60% avec traitement temps réel (<2 secondes).

## 🛠️ Installation Recommandée (avec environnement virtuel)

### Option 1 : Script automatique
```bash
# Rendre le script exécutable
chmod +x setup_env.sh

# Lancer le setup
./setup_env.sh
```

### Option 2 : Installation manuelle
```bash
# 1. Créer l'environnement virtuel
python -m venv venv_tpv

# 2. Activer l'environnement virtuel
# Sur macOS/Linux :
source venv_tpv/bin/activate
# Sur Windows :
source venv_tpv/Scripts/activate

# 3. Installer les dépendances
pip install -r requirements-minimal.txt
```

## 🚀 Utilisation

### Pipeline complet
```bash
# Activer l'environnement virtuel
source venv_tpv/bin/activate  # macOS/Linux
# source venv_tpv/Scripts/activate  # Windows

cd code

# 1. Télécharger les données EEG (~2-3 GB)
python 1-download_data.py

# 2. Préprocesser les données
python 2-preprocess_data.py

# 3. Entraîner le modèle
python 3-train.py

# 4. Tester en temps réel simulé
python 4-predict.py
```

### Scripts d'exploration (optionnels)
```bash
python explore_raw_data.py        # Visualiser données brutes
python explore_processed_data.py  # Analyser données préprocessées
python train_skl_pca.py          # Comparer avec PCA sklearn
python show_sensor_location.py   # Visualiser positions électrodes
```

## 📊 Architecture

### Pipeline de traitement
1. **Téléchargement** : Dataset EEG BCI (19 sujets)
2. **Préprocessing** : Filtrage 8-30 Hz, extraction époques
3. **Entraînement** : StandardScaler → PCA → LDA
4. **Prédiction** : Simulation temps réel

### Caractéristiques extraites
- **Rythme mu (8-12 Hz)** : mouvements moteurs
- **Rythme bêta (13-30 Hz)** : préparation motrice
- Focus canaux moteurs (C3, C4, Cz)

## 📋 Dépendances

**Dépendances principales** (dans `requirements-minimal.txt`) :
- `mne` : Traitement signaux EEG
- `numpy` : Calculs numériques
- `matplotlib` : Visualisations
- `scikit-learn` : Machine learning

> **Note :** L'ancien `requirements.txt` contenait beaucoup de dépendances inutiles. Utilisez `requirements-minimal.txt` pour une installation plus propre.

## 📈 Résultats

- **Accuracy cible** : >60%
- **Temps traitement** : <2 secondes
- **Validation** : Train/Val/Test + cross-validation 5-fold
- **Généralisation** : Test inter-sujets

## 📁 Structure du projet

```
Total-Perspective-Vortex/
├── code/                    # Scripts Python
│   ├── 1-download_data.py   # Téléchargement données
│   ├── 2-preprocess_data.py # Préprocessing EEG
│   ├── 3-train.py          # Entraînement modèle
│   ├── 4-predict.py        # Prédiction temps réel
│   ├── my_pca.py           # PCA implémentation personnelle
│   └── explore_*.py        # Scripts d'exploration
├── data/                   # Données EEG et résultats
├── models/                 # Modèles entraînés
├── requirements-minimal.txt # Dépendances optimisées
└── setup_env.sh           # Script setup environnement
```

## 🔬 Aspects scientifiques

- **Neurosciences** : Rythmes mu/bêta du cortex moteur
- **Signal processing** : Filtrage, PSD, extraction caractéristiques
- **Machine Learning** : Classification discriminante, validation croisée
- **BCI** : Interface cerveau-machine temps réel

## 💡 Améliorations possibles

- Implémenter CSP (Common Spatial Patterns)
- Deep learning avec CNN
- Interface graphique temps réel
- Optimisation hyperparamètres

---

**🧠 Explorez les mystères de la conscience avec Total-Perspective-Vortex !**
