# explore_processed_data.py
import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def explore_eeg_data(file_path='../data/processed/S001-epo.fif'):
    """
    Charge et explore les données EEG prétraitées avec une analyse approfondie
    des caractéristiques spectrales pertinentes pour l'imagerie motrice.
    """
    # S'assurer que le dossier de sortie existe
    output_dir = Path('../data/processed/')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger les données prétraitées
    print(f"Chargement des données depuis {file_path}...")
    epochs = mne.read_epochs(file_path)

    # Afficher les informations de base
    print("\n===== INFORMATIONS GÉNÉRALES =====")
    print(f"Nombre total d'époques: {len(epochs)}")
    print(
        f"Durée de chaque époque:"
        f" {epochs.times[0]:.1f}s à {epochs.times[-1]:.1f}s")
    print(f"Fréquence d'échantillonnage: {epochs.info['sfreq']} Hz")
    print(f"Nombre d'électrodes: {len(epochs.ch_names)}")

    # Afficher les types de mouvements disponibles
    print("\n===== TYPES DE MOUVEMENTS =====")
    for movement_type, code in epochs.event_id.items():
        count = len(epochs[movement_type])
        print(f"- {movement_type}: {count} époques (code: {code})")

    # Identifier les électrodes du cortex moteur
    motor_channels = []
    for motor_ch in ['C3', 'C4', 'Cz']:
        for ch_name in epochs.ch_names:
            if motor_ch in ch_name:
                motor_channels.append((motor_ch, ch_name))
                break

    if not motor_channels:
        print(
            "Aucune électrode du cortex moteur trouvée."
            " Voici les électrodes disponibles:")
        print(epochs.ch_names)
        return epochs

    print(
        f"\nÉlectrodes du cortex moteur"
        f" trouvées: {[ch[0] for ch in motor_channels]}")

    # 1. VISUALISATION DU SIGNAL TEMPOREL
    plt.figure(figsize=(12, 6))
    for motor_name, ch_name in motor_channels:
        ch_idx = epochs.ch_names.index(ch_name)

        # Créer un sous-graphique pour chaque électrode motrice
        plt.figure(figsize=(12, 6))
        for movement_type in epochs.event_id.keys():
            # Calculer la moyenne pour ce type de mouvement
            avg_data = epochs[movement_type].average()
            plt.plot(epochs.times, avg_data.data[ch_idx], label=movement_type)

        plt.axvline(x=0, color='k', linestyle='--', label='Début du mouvement')
        plt.title(
            f'Activité temporelle moyenne pour l\'électrode {motor_name}')
        plt.xlabel('Temps (secondes)')
        plt.ylabel('Amplitude (µV)')
        plt.legend()
        plt.grid(True)

        # Sauvegarder la figure
        output_file = output_dir / f'signal_temporal_{motor_name}.png'
        plt.savefig(output_file)
        print(f"Graphique temporel sauvegardé: {output_file}")
        plt.close()

    # 2. ANALYSE SPECTRALE (INFORMATION CLÉ POUR L'IMAGERIE MOTRICE)
    plt.figure(figsize=(15, 10))

    # Paramètres pour l'analyse spectrale
    fmin, fmax = 4, 40  # Plage de fréquences pertinente

    # Calculer le spectre de puissance pour chaque condition et chaque
    # électrode
    for movement_type in epochs.event_id.keys():
        # Créer une figure pour ce type de mouvement
        plt.figure(figsize=(12, 8))

        # Pour chaque électrode motrice
        for i, (motor_name, ch_name) in enumerate(motor_channels):
            plt.subplot(len(motor_channels), 1, i + 1)

            # Calculer le spectre de puissance (méthode multitaper pour une
            # estimation robuste)
            spectrum = epochs[movement_type].compute_psd(
                method='multitaper',
                fmin=fmin,
                fmax=fmax,
                picks=[ch_name]
            )

            # Extraire les fréquences et les données spectrales
            freqs = spectrum.freqs
            psd_data = spectrum.get_data().mean(axis=0).squeeze()

            # Tracer le spectre
            plt.plot(freqs, psd_data)

            # Marquer les bandes de fréquence pertinentes
            plt.axvspan(8, 12, color='lightblue', alpha=0.3,
                        label='Bande mu (8-12 Hz)')
            plt.axvspan(13, 30, color='lightgreen', alpha=0.3,
                        label='Bande bêta (13-30 Hz)')

            plt.title(f'Spectre de puissance - {motor_name} - {movement_type}')
            plt.xlabel('Fréquence (Hz)')
            plt.ylabel('Puissance spectrale (µV²/Hz)')
            plt.grid(True)

            # N'ajouter la légende qu'une seule fois
            if i == 0:
                plt.legend()

        # Sauvegarder le spectre pour ce type de mouvement
        output_file = output_dir / f'spectrum_{movement_type}.png'
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Spectre de puissance sauvegardé: {output_file}")
        plt.close()

    # 3. COMPARAISON DIRECTE DES BANDES DE FRÉQUENCE SPÉCIFIQUES
    # Ceci est crucial pour l'imagerie motrice

    # Extraire la puissance dans les bandes mu et bêta pour chaque condition
    mu_band = (8, 12)
    beta_band = (13, 30)

    # Créer un graphique pour chaque bande
    for band_name, (fmin, fmax) in [('mu', mu_band), ('beta', beta_band)]:
        plt.figure(figsize=(10, 8))

        # Préparer les données
        movement_types = list(epochs.event_id.keys())
        band_powers = []

        for motor_name, ch_name in motor_channels:
            powers_for_channel = []

            for movement_type in movement_types:
                # Calculer le spectre de puissance
                spectrum = epochs[movement_type].compute_psd(
                    method='multitaper',
                    fmin=fmin,
                    fmax=fmax,
                    picks=[ch_name]
                )

                # Extraire la puissance moyenne dans cette bande
                power = spectrum.get_data().mean()
                powers_for_channel.append(power)

            band_powers.append(powers_for_channel)

        # Créer un graphique à barres
        x = np.arange(len(movement_types))
        width = 0.2

        for i, (motor_name, _) in enumerate(motor_channels):
            plt.bar(x + i * width, band_powers[i], width, label=motor_name)

        plt.xlabel('Type de mouvement')
        plt.ylabel(f'Puissance dans la bande {band_name} (µV²/Hz)')
        plt.title(
            f'Comparaison de la puissance '
            f'{band_name} ({"8-12" if band_name == "mu" else "13-30"}'
            f' Hz) par condition')
        plt.xticks(x + width, movement_types)
        plt.legend()
        plt.grid(True, axis='y')

        # Sauvegarder la figure
        output_file = output_dir / f'band_power_comparison_{band_name}.png'
        plt.savefig(output_file)
        print(
            f"Comparaison de puissance {band_name} sauvegardée: {output_file}")
        plt.close()

    return epochs


if __name__ == "__main__":
    try:
        epochs = explore_eeg_data()
        print("\nExploration terminée avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'exploration des données: {str(e)}")
