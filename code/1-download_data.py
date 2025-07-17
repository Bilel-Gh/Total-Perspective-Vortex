# download_data.py
from mne.datasets import eegbci


def download_eeg_data(subjects=range(1, 20), runs=[4, 6], data_path='../data'):
    """
    Downloads EEG motor imagery data from the EEG BCI dataset.

    Parameters:
    - subjects: List of subject numbers to download
    - runs: List of experimental runs to download
        Run 3: Motor execution of left/right fist
        Run 4: Motor imagery of left/right fist
        Run 5: Motor execution of both fists/both feet
        Run 6: Motor imagery of both fists/both feet
        (Runs 7-14 repeat these same four paradigms three times)
    - data_path: Directory to save downloaded data

    Returns:
    - None, files are saved to disk
    """
    print("Starting data download...")
    for subject in subjects:
        for run in runs:
            print(f"Downloading data for subject {subject}, run {run}...")
            files = eegbci.load_data(subject, runs=[run], path=data_path,
                                     verbose=True)
            print(f"Downloaded files: {files}")

    print("\nDownload completed successfully")


if __name__ == "__main__":
    download_eeg_data()
