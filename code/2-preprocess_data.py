from pathlib import Path
import mne
import numpy as np
from matplotlib import pyplot as plt


def create_output_dir(output_dir):
    """
    Creates output directory for processed data if it doesn't exist.

    Parameters:
    - output_dir: Path where processed data will be saved

    Returns:
    - Path object of the created directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def extract_events(raw):
    """
    Extracts events (markers) from the continuous EEG data.

    Parameters:
    - raw: MNE Raw object with EEG data

    Returns:
    - events: Array of event timings and types
    - event_id: Dictionary mapping event names to numerical codes
    """
    events, event_id = mne.events_from_annotations(raw)

    print("Available event types:", event_id)
    print(f"Number of events found: {len(events)}")

    return events, event_id


def preprocess_data(subject, run, data_path):
    """
    Loads and filters EEG data between 8 and 30 Hz to focus on
    mu and beta rhythms relevant for motor imagery tasks.

    Parameters:
    - subject: Subject number
    - run: Experimental run number
    - data_path: Base directory for the dataset

    Returns:
    - raw: Filtered MNE Raw object
    """
    print(f"Preprocessing data for subject {subject}, run {run}...")
    data_path = Path(data_path)
    file_path = data_path / f'S{subject:03d}' / f'S{subject:03d}R{run:02d}.edf'
    print(f"file_path = {file_path}")
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Visualize raw data before filtering
    visualize_raw_data(raw)

    filtered_raw = raw.copy()
    filtered_raw.filter(8., 30.)

    # Visualize data after filtering
    visualize_raw_data(raw, filtered_raw)

    return filtered_raw


def create_epochs(raw, events, run):
    """
    Creates epochs (time segments) around events of interest based on the
     specific run type.

    Parameters:
    - raw: MNE Raw object with EEG data
    - events: Array of event timings and types
    - run: Experimental run number to determine task type

    Returns:
    - epochs: MNE Epochs object with segmented data
    """
    # Define task IDs based on run number
    task_id = {}

    # Run 3, 7, 11: Motor execution (real movement) of left/right fist
    # Run 4, 8, 12: Motor imagery of left/right fist
    if run in [3, 4, 7, 8, 11, 12]:
        task_id = {
            'left_fist': 1,  # T1 corresponds to left fist
            'right_fist': 2,  # T2 corresponds to right fist
        }
    # Run 5, 9, 13: Motor execution (real movement) of both fists/both feet
    # Run 6, 10, 14: Motor imagery of both fists/both feet
    elif run in [5, 6, 9, 10, 13, 14]:
        task_id = {
            'both_fists': 1,  # T1 corresponds to both fists
            'both_feet': 2,  # T2 corresponds to both feet
        }

    # Create epochs from 1 second before to 4 seconds after the event
    tmin, tmax = -1., 4.
    epochs = mne.Epochs(raw, events, task_id, tmin, tmax, proj=True,
                        baseline=None, preload=True)

    print(f"Number of epochs created: {len(epochs)}")
    print(f"Task types in this run: {list(task_id.keys())}")

    return epochs


def extract_features(epochs):
    """
    Extracts frequency-domain features from epoched EEG data.

    Parameters:
    - epochs: MNE Epochs object with segmented data

    Returns:
    - features: Dictionary containing extracted features
    """
    features = {}

    # 1. Extract power spectral density (PSD)
    psds, freqs = epochs.compute_psd(method='multitaper', fmin=8, fmax=30,
                                     tmin=0.5, tmax=2.5, n_jobs=1,
                                     verbose=False).get_data(return_freqs=True)
    features['psd'] = psds
    features['freqs'] = freqs

    # 2. Calculate band power for mu (8-12 Hz) and beta (13-30 Hz) rhythms
    # Find indices corresponding to frequency bands
    mu_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]

    # Calculate average power in each band for each channel and epoch
    mu_power = np.mean(psds[:, :, mu_idx], axis=2)  # (n_epochs, n_channels)
    beta_power = np.mean(psds[:, :, beta_idx],
                         axis=2)  # (n_epochs, n_channels)

    features['mu_power'] = mu_power
    features['beta_power'] = beta_power

    # 3. Extract channels over sensorimotor areas (if they exist)
    motor_channels = []
    for ch in ['C3', 'C4', 'Cz']:
        for ch_name in epochs.ch_names:
            if ch in ch_name:
                motor_channels.append(epochs.ch_names.index(ch_name))
                break

    features['motor_channels'] = motor_channels

    return features


def visualize_raw_data(raw, filtered_raw=None):
    """
    Visualizes raw EEG data before and after filtering.
    """
    # Plot original raw data
    fig1 = plt.figure(figsize=(15, 6))
    raw.plot(duration=5.0, start=0, n_channels=5,
             scalings='auto', title='Raw EEG signal', show=False)
    plt.savefig('../data/processed/raw_signal.png')
    plt.close(fig1)

    # If filtered data is provided, plot it too
    if filtered_raw:
        fig2 = plt.figure(figsize=(15, 6))
        filtered_raw.plot(duration=5.0, start=0, n_channels=5,
                          scalings='auto',
                          title='Filtered EEG signal (8-30 Hz)',
                          show=False)
        plt.savefig('../data/processed/filtered_signal.png')
        plt.close(fig2)

    # Plot PSD using the modern approach
    fig3 = plt.figure(figsize=(12, 6))
    spectrum = raw.compute_psd()
    spectrum.plot(average=True, picks='eeg', show=False)
    plt.title('Frequency spectrum of raw EEG signal')
    plt.xlim(0, 50)  # Limit to 50 Hz
    plt.savefig('../data/processed/raw_spectrum.png')
    plt.close(fig3)

    if filtered_raw:
        fig4 = plt.figure(figsize=(12, 6))
        filtered_spectrum = filtered_raw.compute_psd()
        filtered_spectrum.plot(average=True, picks='eeg', show=False)
        plt.title('Frequency spectrum after filtering (8-30 Hz)')
        plt.xlim(0, 50)  # Limit to 50 Hz
        plt.savefig('../data/processed/filtered_spectrum.png')
        plt.close(fig4)

    print("Raw visualization completed. Plots saved to ../data/processed/")


def process_subject_data(subject, runs):
    """
    Processes all runs for a single subject and combines the data.

    Parameters:
    - subject: Subject number
    - runs: List of experimental run numbers

    Returns:
    - combined_epochs: MNE Epochs object with all data from the subject
    - all_features: Dictionary of extracted features
    """
    all_epochs = []
    all_features = {}
    data_path = '../data/MNE-eegbci-data/files/eegmmidb/1.0.0/'

    for run in runs:
        # Preprocess data
        raw = preprocess_data(subject, run, data_path)
        # Extract events
        events, event_id = extract_events(raw)
        # Create epochs
        epochs = create_epochs(raw, events, run)
        all_epochs.append(epochs)

        # Extract features
        run_features = extract_features(epochs)

        # Store features by run
        all_features[f'run_{run}'] = run_features

    # Combine all epochs
    combined_epochs = mne.concatenate_epochs(all_epochs)

    # Extract features from combined epochs
    combined_features = extract_features(combined_epochs)
    all_features['combined'] = combined_features

    return combined_epochs, all_features


if __name__ == "__main__":
    # Settings
    subjects = range(1, 20)
    runs = [4, 6]
    output_dir = create_output_dir('../data/processed/')

    # Process all subjects
    all_subject_epochs = {}
    all_subject_features = {}

    for subject in subjects:
        # Process all data for a subject
        subject_epochs, subject_features = process_subject_data(subject, runs)
        all_subject_epochs[subject] = subject_epochs
        all_subject_features[subject] = subject_features

        # Save preprocessed data for this subject
        subject_epochs.save(f'{output_dir}/S{subject:03d}-epo.fif',
                            overwrite=True)

        # Save extracted features (as numpy files)
        np.save(f'{output_dir}/S{subject:03d}-features.npy', subject_features)

    print("Preprocessing completed for all subjects!")
