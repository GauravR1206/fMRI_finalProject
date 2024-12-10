import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from scipy.stats import spearmanr
from pathlib import Path
import pandas as pd 

data_dir = Path('/nfs/masi/saundam1/datasets/food_viewing_fmri/ds000157')
preproc_dir = data_dir / 'derivatives' / 'fsl-preproc'
TR = 1.6

random_state = 42

tmax=30
t = np.arange(0, 24 + TR, TR)
n1 = 5.0          
t1 = 1.1          
n2 = 12.0      
t2 = 0.9         
a2 = 0.4      
t = np.arange(0, tmax + TR, TR)  

h1 = (t ** n1) * np.exp(-t / t1)
h2 = (t ** n2) * np.exp(-t / t2)

h = h1 / np.max(h1) - a2 * (h2 / np.max(h2))
h = h / np.max(h)  

for sub in ['sub-01']:
    preproc_nifti = nib.load(preproc_dir / sub / f'{sub}_bold_preproc_masked.nii.gz')
    mask_nifti = nib.load(preproc_dir / sub / f'{sub}_T1w_brain_mask_to_MNI.nii.gz')

    preproc_data = preproc_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

    # Get event time course
    tsv_file = f'/nfs/masi/saundam1/datasets/food_viewing_fmri/ds000157/{sub}/func/events.tsv'
    events_df = pd.read_csv(tsv_file, sep='\t')

    # Create a time course of the stimulus
    stimulus = np.zeros(preproc_data.shape[-1])
    for index, row in events_df.iterrows():
        onset = row['onset'] 
        duration = row['duration']
        
        start_time = onset
        end_time = onset + duration
        
        start_idx = int(np.floor(start_time / TR))
        end_idx = int(np.floor(end_time / TR))
        
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, stimulus.shape[0])
        
        stimulus[start_idx:end_idx] = 1

    plt.figure(figsize=(10, 3))
    time_vector = np.arange(0, stimulus.shape[0]) * TR
    plt.step(time_vector, stimulus, where='post', color='blue', label='Stimulus')
    plt.xlabel('Time (s)')
    plt.ylabel('Stimulus')
    plt.title('Binary Stimulus Vector')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    convolved = np.convolve(stimulus, h)[:stimulus.shape[0]]
    regressor = convolved / np.max(convolved)

    plt.figure(figsize=(10, 3))
    plt.plot(time_vector, regressor, color='purple', label='Convolved Regressor')
    plt.xlabel('Time (s)')
    plt.ylabel('Regressor Value')
    plt.title('Regressor: Stimulus Convolved with HRF')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Reshape the data to 2D
    X = preproc_data.reshape(-1, preproc_data.shape[-1])  # (voxels, timepoints)
    time = np.arange(0, X.shape[1])*TR

    # Only keep the voxels within the mask
    mask = mask_data.astype(bool).ravel()
    X = X[mask]

    print(f'{X.shape} voxels x timepoints')

    # Run PCA with a high number of components for explained variance plot
    n_components = min(X.shape[1], 100)  # Limited by time points or max 100
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X.T).T

    print(f'{X_pca.shape} PCA components x timepoints')

    # Plot the elbow plot for explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Elbow Plot of Explained Variance by PCA Components')
    plt.grid(True)

    # Select a number of components for ICA based on explained variance threshold
    chosen_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1  # 95% variance
    print(f'Selected number of components for ICA: {chosen_components}')

    # Perform ICA
    ica = FastICA(n_components=8, random_state=random_state)
    X_ica = ica.fit_transform(X_pca[:chosen_components].T).T

    print(f'{X_ica.shape} ICA components x timepoints')

    # Plot the ICA components
    fig, axes = plt.subplots(4, 2, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(time, X_ica[i], color='red', label='ICA Component')
        ax.set_xlabel('Time (s)')

        # Plot the stimulus and convolved regressor
        # ax.plot(time_vector, stimulus, color='blue', label='Stimulus')
        ax.plot(time_vector, regressor, color='b', label='Regressor')
        ax.legend()

        # Calculate the correlation between the ICA component and the regressor
        corr, _ = spearmanr(X_ica[i], regressor)
        ax.set_title(f'IC {i+1} (Corr: {corr:.2f})')

    fig.tight_layout()
    plt.suptitle(f'{sub} ICA components')
    plt.show()
