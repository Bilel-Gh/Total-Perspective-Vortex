import mne
import matplotlib.pyplot as plt

eeg_file = '../data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R05.edf'

print("Chargement des données EEG...")
raw_data = mne.io.read_raw_edf(eeg_file, preload=True)

mne.datasets.eegbci.standardize(raw_data)
print("Noms des canaux standardisés")

print("\nInformations sur les données:")
print(raw_data.info)

print("\nAffichage des données brutes (5 secondes)...")
raw_data.plot(duration=5, scalings='auto')

print("\nExtraction des evenements...")
events, events_id = mne.events_from_annotations(raw_data)
print(f"Types d'événements: {events_id}")
print(f"Nombre d'événements: {len(events)}")

mne.viz.plot_events(events, raw_data.info['sfreq'])

plt.show()
