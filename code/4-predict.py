import numpy as np
import mne
from pathlib import Path
import pickle
import time

# Fonction pour extraire les caractéristiques (reprise de train.py)


def extract_features(epoch):
    """Extrait les caractéristiques pertinentes d'une époque EEG."""
    # Calculer le spectre de puissance
    spectrum = epoch.compute_psd(method='welch',
                                 fmin=8,
                                 fmax=30,
                                 picks='eeg',
                                 n_fft=256)

    # Obtenir les données et fréquences
    psds = spectrum.get_data().squeeze()
    freqs = spectrum.freqs

    # Trouver les indices des bandes mu (8-12 Hz) et bêta (13-30 Hz)
    mu_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]

    # Calculer la puissance moyenne dans chaque bande
    mu_power = np.mean(psds[:, mu_idx], axis=1)
    beta_power = np.mean(psds[:, beta_idx], axis=1)

    # Concaténer et aplatir si nécessaire
    features = np.concatenate([mu_power, beta_power])
    if features.ndim > 1:
        features = features.flatten()

    return features

# Fonction pour simuler un flux de données en temps réel


def simulate_realtime_processing(pipeline, data_path, subject_id):
    """Simule un traitement en temps réel à partir d'un fichier EEG."""
    # Charger les données pour la simulation
    file_path = data_path / f'S{subject_id:03d}-epo.fif'
    epochs = mne.read_epochs(file_path)

    predictions = []
    processing_times = []
    true_labels = []

    print(
        f"Simulation du traitement en temps réel sur le sujet {subject_id}...")
    print(f"Types de mouvements disponibles: {epochs.event_id}")

    # Pour chaque type de mouvement
    for movement_type, event_code in epochs.event_id.items():
        # Sélectionner les époques pour ce mouvement
        movement_epochs = epochs[movement_type]

        # Pour chaque époque de ce mouvement
        for i in range(len(movement_epochs)):
            # Prendre une époque individuelle
            single_epoch = movement_epochs[i]

            # Simuler l'arrivée des données
            print(
                f"Traitement de l'époque {i} (type: {movement_type})...",
                end="")

            start_time = time.time()

            # Extraire les caractéristiques
            features = extract_features(single_epoch)

            # Faire la prédiction
            prediction = pipeline.predict([features])[0]

            # Mesurer le temps de traitement
            processing_time = time.time() - start_time

            # Vérifier le délai
            if processing_time > 2.0:
                print(
                    f" WARNING: Traitement trop lent ({processing_time:.6f}s)")
            else:
                print(f" Terminé en {processing_time:.6f}s")

            # Enregistrer les résultats
            predictions.append(prediction)
            processing_times.append(processing_time)
            true_labels.append(event_code)

    # Calculer les statistiques
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    avg_time = np.mean(processing_times)

    print(f"\nRésultats de la simulation pour le sujet {subject_id}:")
    print(f"Nombre d'époques traitées: {len(predictions)}")
    print(f"Exactitude: {accuracy:.6f}")
    print(f"Temps de traitement moyen: {avg_time:.9f}s")

    return accuracy, avg_time


if __name__ == "__main__":
    # Vérifier que le modèle existe
    model_path = Path('../models/trained_model.pkl')
    if not model_path.exists():
        print("Erreur: Le modèle entraîné n'a pas été trouvé.")
        print("Exécutez d'abord train.py pour créer le modèle.")
        exit(1)

    # Charger le modèle entraîné
    print("Chargement du modèle entraîné...")
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    # Simuler un traitement en temps réel
    data_path = Path('../data/processed/')

    # Liste des sujets de test
    test_subjects = [7, 11, 15]  # Tous les sujets de test
    subject_accuracies = []
    subject_times = []

    # Tester chaque sujet
    for subject_id in test_subjects:
        print(f"\n\n===== ÉVALUATION DU SUJET {subject_id} =====")
        try:
            accuracy, avg_time = simulate_realtime_processing(
                pipeline, data_path, subject_id)
            subject_accuracies.append(accuracy)
            subject_times.append(avg_time)
        except FileNotFoundError:
            print(
                f"Warning: Données pour le sujet {subject_id} non trouvées,"
                f" ignorées.")

    # Calculer les moyennes globales
    if subject_accuracies:
        mean_accuracy = np.mean(subject_accuracies)
        mean_time = np.mean(subject_times)

        print("\n\n===== RÉSULTATS GLOBAUX =====")
        print("Exactitude par sujet:")
        for i, subject_id in enumerate(test_subjects):
            if i < len(subject_accuracies):
                print(f"- Sujet {subject_id}: {subject_accuracies[i]:.6f}")

        print(f"\nExactitude moyenne sur tous les sujets: {mean_accuracy:.6f}")
        print(f"Temps de traitement moyen: {mean_time:.9f}s")

        # Vérifier si l'objectif de 60% d'exactitude moyenne est atteint
        if mean_accuracy >= 0.6:
            print("\n✅ Objectif atteint: Exactitude moyenne > 60%")
        else:
            print("\n❌ Objectif non atteint: Exactitude moyenne < 60%")
    else:
        print("Aucun sujet de test n'a pu être traité.")
