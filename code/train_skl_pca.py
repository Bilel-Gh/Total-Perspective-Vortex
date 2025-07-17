import numpy as np
import mne
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import time


def load_subject_data(subject_id, data_path):
    """
    Charge les données prétraitées pour un
     sujet et extrait les caractéristiques.
    """
    file_path = data_path / f'S{subject_id:03d}-epo.fif'
    epochs = mne.read_epochs(file_path)

    X = []
    y = []

    # Pour chaque type de mouvement
    for movement_type, event_code in epochs.event_id.items():
        # Sélectionner les époques pour ce mouvement
        movement_epochs = epochs[movement_type]

        # Pour chaque époque de ce mouvement
        for i in range(len(movement_epochs)):
            # Prendre une époque individuelle
            single_epoch = movement_epochs[i]

            # Calculer le spectre de puissance avec la méthode moderne
            spectrum = single_epoch.compute_psd(method='welch',
                                                fmin=8,
                                                fmax=30,
                                                picks='eeg',
                                                n_fft=256)

            # Obtenir les données et fréquences
            # Ajouter squeeze() pour enlever les dimensions inutiles
            psds = spectrum.get_data().squeeze()
            freqs = spectrum.freqs

            # Trouver les indices des bandes mu (8-12 Hz) et bêta (13-30 Hz)
            mu_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
            beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]

            # Calculer la puissance moyenne dans chaque bande
            mu_power = np.mean(psds[:, mu_idx],
                               axis=1)  # Moyenne par électrode
            beta_power = np.mean(psds[:, beta_idx],
                                 axis=1)  # Moyenne par électrode

            # Concaténer les caractéristiques et s'assurer qu'elles sont 1D
            features = np.concatenate([mu_power, beta_power])

            # Vérifier que features est bien un vecteur 1D
            if features.ndim > 1:
                features = features.flatten()

            # Ajouter aux données
            X.append(features)
            y.append(event_code)

    return np.array(X), np.array(y)


def create_pipeline():
    """
    Crée un pipeline de traitement avec
     réduction de dimensionnalité et classification.
    """
    # Créer un pipeline simple avec PCA et LDA
    # Note: PCA sera remplacé par CSP dans une version future
    pipeline = Pipeline([
        ('dimensionality_reduction', PCA(n_components=10)),
        ('classifier', LinearDiscriminantAnalysis())
    ])

    return pipeline

# Fonction pour simuler un traitement en temps réel


def simulate_realtime_processing(pipeline, X_test, y_test):
    """Simule un traitement en temps réel avec contrainte de délai (<2s)."""
    predictions = []
    processing_times = []

    for i, features in enumerate(X_test):
        start_time = time.time()

        # Prédire la classe (type de mouvement)
        prediction = pipeline.predict([features])[0]

        # Mesurer le temps de traitement
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        # Vérifier si le traitement est assez rapide
        if processing_time > 2.0:
            print(f"Attention: Le traitement de l'époque {i} a pris "
                  f"{processing_time:.9f}s (>2s)")

        predictions.append(prediction)

    # Calculer l'exactitude
    accuracy = np.mean(predictions == y_test)
    avg_time = np.mean(processing_times)

    print(f"Exactitude: {accuracy:.6f}")
    print(f"Temps de traitement moyen: {avg_time:.9f}s")

    return accuracy, avg_time


# Programme principal
if __name__ == "__main__":
    data_path = Path('../data/processed/')

    # 2. Diviser les sujets en ensembles d'entraînement, validation et test
    all_subjects = list(range(1, 20))
    np.random.seed(42)
    np.random.shuffle(all_subjects)

    train_subjects = all_subjects[:13]  # ~70% pour l'entraînement
    val_subjects = all_subjects[13:16]  # ~15% pour la validation
    test_subjects = all_subjects[16:]   # ~15% pour le test

    print(f"Sujets d'entraînement: {sorted(train_subjects)}")
    print(f"Sujets de validation: {sorted(val_subjects)}")
    print(f"Sujets de test: {sorted(test_subjects)}")

    # 3. Charger les données d'entraînement
    X_train = []
    y_train = []

    print("Chargement des données d'entraînement...")
    for subject in train_subjects:
        try:
            X_subj, y_subj = load_subject_data(subject, data_path)
            X_train.extend(X_subj)
            y_train.extend(y_subj)
            print(f"  Sujet {subject}: {len(X_subj)} exemples chargés")
        except FileNotFoundError:
            print(f"  Données pour le sujet {subject} non trouvées, ignorées")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Total: {len(X_train)} exemples d'entraînement")

    # 4. Créer et entraîner le pipeline
    print("Création et entraînement du pipeline...")
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. Validation croisée pour évaluer les performances
    print("Évaluation par validation croisée...")
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"Scores de validation croisée: {scores}")
    print(f"Score moyen: {np.mean(scores):.4f}")

    # 6. Charger les données de validation pour ajuster les hyperparamètres
    # (si nécessaire)
    print("Chargement des données de validation...")
    X_val = []
    y_val = []
    for subject in val_subjects:
        try:
            X_subj, y_subj = load_subject_data(subject, data_path)
            X_val.extend(X_subj)
            y_val.extend(y_subj)
            print(f"  Sujet {subject}: {len(X_subj)} exemples chargés")
        except FileNotFoundError:
            print(f"  Données pour le sujet {subject} non trouvées, ignorées")

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    val_score = pipeline.score(X_val, y_val)
    print(f"Score sur les données de validation: {val_score:.4f}")

    # 7. Charger les données de test pour l'évaluation finale
    print("Chargement des données de test...")
    X_test = []
    y_test = []
    for subject in test_subjects:
        try:
            X_subj, y_subj = load_subject_data(subject, data_path)
            X_test.extend(X_subj)
            y_test.extend(y_subj)
            print(f"  Sujet {subject}: {len(X_subj)} exemples chargés")
        except FileNotFoundError:
            print(f"  Données pour le sujet {subject} non trouvées, ignorées")

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 8. Simuler un traitement en temps réel sur les données de test
    print("Simulation du traitement en temps réel...")
    accuracy, avg_time = simulate_realtime_processing(pipeline, X_test, y_test)

    # 9. Vérifier si le score dépasse 60% comme requis
    if accuracy >= 0.6:
        print("✅ Objectif atteint: Exactitude > 60%")
    else:
        print("❌ Objectif non atteint: Exactitude < 60%")
