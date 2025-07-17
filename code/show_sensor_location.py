import matplotlib.pyplot as plt
import mne

eeg_file = '../data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf'
raw_data = mne.io.read_raw_edf(eeg_file, preload=True)
mne.datasets.eegbci.standardize(raw_data)

# montage = position des electrodes.
# standard_1020 c'est le plus courant
montage = mne.channels.make_standard_montage("standard_1020")
raw_data.set_montage(montage)

raw_data.plot_sensors(
    kind="topomap",
    show_names=True,
    sphere=(0, 0, 0, 0.095)  # Ajuste la valeur du rayon
)

raw_data.plot_sensors(kind="3d", show_names=True)

plt.show()
