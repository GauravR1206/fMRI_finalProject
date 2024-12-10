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
t = np.arange(0, tmax + TR, TR)
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
tsv_file = f'/nfs/masi/saundam1/datasets/food_viewing_fmri/ds000157/sub-01/func/sub-01_task-passiveimageviewing_events.tsv'


for sub in ['sub-07']:
    preproc_nifti = nib.load(preproc_dir / sub / f'{sub}_bold_preproc_masked.nii.gz')
    mask_nifti = nib.load(preproc_dir / sub / f'{sub}_T1w_brain_mask_to_MNI.nii.gz')

    preproc_data = preproc_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

    events_df = pd.read_csv(tsv_file, sep='\t')

    food_stimulus = np.zeros(preproc_data.shape[-1])
    nonfood_stimulus = np.zeros(preproc_data.shape[-1])

    for _, row in events_df.iterrows():
        onset, duration, trial_type = row['onset'], row['duration'], row['trial_type']
        start_idx = int(np.floor(onset / TR))
        end_idx = int(np.floor((onset + duration) / TR))
        
        if trial_type == 'food':
            food_stimulus[start_idx:end_idx] = 1
        elif trial_type == 'nonfood':
            nonfood_stimulus[start_idx:end_idx] = 1

    time_vector = np.arange(0, food_stimulus.shape[0]) * TR
    # plt.figure(figsize=(10, 3))
    # plt.step(time_vector, food_stimulus, where='post', color='green', label='Food Stimulus')
    # plt.step(time_vector, nonfood_stimulus, where='post', color='orange', label='No Food Stimulus')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Stimulus')
    # plt.title('Binary Stimulus Vectors for Food and No Food')
    # plt.ylim(-0.1, 1.1)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    food_regressor = np.convolve(food_stimulus, h)[:food_stimulus.shape[0]]
    food_regressor = food_regressor / np.max(food_regressor) 

    nonfood_regressor = np.convolve(nonfood_stimulus, h)[:nonfood_stimulus.shape[0]]
    nonfood_regressor = nonfood_regressor / np.max(nonfood_regressor)

    # plt.figure(figsize=(10, 3))
    # plt.plot(time_vector, food_regressor, color='green', label='Food')
    # plt.plot(time_vector, nonfood_regressor, color='orange', label='No Food')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Food and No Food Stimulus HRFs')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    X = preproc_data.reshape(-1, preproc_data.shape[-1]) 
    time = np.arange(0, X.shape[1]) * TR

    mask = mask_data.astype(bool).ravel()
    X = X[mask]
    print(f'{X.shape[0]} voxels x {X.shape[1]} timepoints')


    # Demean 
    X = X - X.mean(axis=1, keepdims=True)

    n_components = min(X.shape[1], 100)  
    pca = PCA(n_components=n_components,random_state=random_state)
    X_pca = pca.fit_transform(X.T).T 

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, n_components + 1), cumulative_variance_ratio, marker='o')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.title('Elbow Plot of Explained Variance by PCA Components')
    # plt.grid(True)
    # plt.show()

    chosen_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f'Selected number of components for ICA: {chosen_components}')

    # Perform ICA on chosen number of components
    ica = FastICA(n_components=8, random_state=random_state)
    X_ica = ica.fit_transform(X_pca[:chosen_components].T).T  # ICA on PCA-reduced data
    print(f'{X_ica.shape[0]} ICA components x {X_ica.shape[1]} timepoints')

    # Plot ICA components with correlation to both food and nonfood regressors
    fig, axes = plt.subplots(4, 2, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(time, X_ica[i], color='red', label='ICA Component')
        ax.set_xlabel('Time (s)')
        ax.plot(time_vector, food_regressor, color='green', label='Food Regressor')
        ax.plot(time_vector, nonfood_regressor, color='orange', label='No Food Regressor')

        # Calculate and display correlation between ICA component and each regressor
        corr_food, _ = spearmanr(X_ica[i], food_regressor)
        corr_nonfood, _ = spearmanr(X_ica[i], nonfood_regressor)
        ax.set_title(f'IC {i+1} (Food Corr: {corr_food:.2f}, No Food Corr: {corr_nonfood:.2f})')
        ax.legend()
    fig.tight_layout()
    plt.suptitle(f'{sub} ICA components')
    plt.show()
