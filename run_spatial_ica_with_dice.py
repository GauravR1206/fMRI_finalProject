import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from scipy.stats import spearmanr
from pathlib import Path
import pandas as pd 
from scipy import stats
from nilearn.plotting import plot_stat_map

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


for sub in ['sub-01']:
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

    food_regressor = np.convolve(food_stimulus, h)[:food_stimulus.shape[0]]
    food_regressor = food_regressor / np.max(food_regressor) 

    nonfood_regressor = np.convolve(nonfood_stimulus, h)[:nonfood_stimulus.shape[0]]
    nonfood_regressor = nonfood_regressor / np.max(nonfood_regressor)

    # Make design matrix with food, nonfood, constant, linear and quadratic terms
    design_matrix = np.vstack([food_regressor, nonfood_regressor, np.ones_like(food_regressor), time_vector, time_vector**2]).T
    
    X = preproc_data.reshape(-1, preproc_data.shape[-1]) 

    # Get betas
    betas = np.linalg.pinv(design_matrix) @ X.T
    print(f'{betas.shape[0]} betas x {betas.shape[1]} voxels')

    # Hypothesis testing by forming t-stat
    # Get contrast for food vs nonfood
    contrast = np.array([1, -1, 0, 0, 0])
    err = (X.T - (design_matrix @ betas))
    n = design_matrix.shape[0]
    p = design_matrix.shape[1]
    print(f'{err.shape=}, {n=}, {p=}')
    errorVar = 1/(n-p) * np.sum(err**2, axis=0)
    t_num = contrast.T @ betas
    t_denom = np.sqrt(errorVar * (contrast.T @ np.linalg.pinv(design_matrix.T @ design_matrix) @ contrast))
    t_stat = t_num / t_denom
    t_vol = t_stat.reshape(preproc_data.shape[:-1])

    # Plot t-stat map
    crit_val = stats.t.ppf(1-0.001, n-p) # alpha = 0.001
    t_vol_thresh = t_vol.copy()
    t_vol_thresh[np.abs(t_vol) < crit_val] = 0

    t_nifti = nib.Nifti1Image(t_vol_thresh, preproc_nifti.affine, preproc_nifti.header)
    # plot_stat_map(t_nifti, title=f'{sub} Food vs No Food', display_mode='ortho', draw_cross=True, threshold=crit_val, radiological=True, bg_img=mask_nifti, resampling_interpolation='nearest')

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from pathlib import Path
from nilearn.plotting import plot_stat_map
from scipy.stats import scoreatpercentile

data_dir = Path('/nfs/masi/saundam1/datasets/food_viewing_fmri/ds000157')
preproc_dir = data_dir / 'derivatives' / 'fsl-preproc'
TR = 1.6
random_state = 42

for sub in ['sub-01']:
    preproc_nifti = nib.load(preproc_dir / sub / f'{sub}_bold_preproc_masked.nii.gz')
    mask_nifti = nib.load(preproc_dir / sub / f'{sub}_T1w_brain_mask_to_MNI.nii.gz')
    t1w_nifti = nib.load(preproc_dir / sub / f'{sub}_T1w_to_MNI_nlin.nii.gz')

    preproc_data = preproc_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

    # Reshape the data to 2D
    X = preproc_data.reshape(-1, preproc_data.shape[-1])  # (voxels, timepoints)
    time = np.arange(0, X.shape[1])*TR

    # Only keep the voxels within the mask
    mask = mask_data.astype(bool).ravel()
    X = X[mask]

    print(f'{X.shape} voxels x timepoints')

    # De-mean
    X -= X.mean(axis=1, keepdims=True)

    # Run PCA with a high number of components for explained variance plot
    n_components = min(X.shape[1], 100)  # Limited by time points or max 100
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X).T

    print(f'{X_pca.shape} PCA components x voxels')

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

    print(f'{X_ica.shape} ICA components x voxels')

    # Reshape back into shape of the brain (with mask)
    ic_nifti_data = np.zeros((preproc_data.shape[:-1] + (X_ica.shape[0],)))
    ic_nifti_data = ic_nifti_data.reshape(-1, X_ica.shape[0])
    ic_nifti_data[mask, :] = X_ica.T
    ic_nifti_data = ic_nifti_data.reshape(preproc_data.shape[:-1] + (X_ica.shape[0],))

    print(f'{ic_nifti_data.shape} IC nifti data shape')

    # # Plot the ICA components
    # fig, axes = plt.subplots(4, 2, figsize=(20, 10))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.plot(time, X_ica[i])
    #     ax.set_title(f'IC {i+1}')
    #     ax.set_xlabel('Time (s)')

    for i in range(X_ica.shape[0]):
        ic_thresh = scoreatpercentile(np.abs(ic_nifti_data[..., i]), 95)
        ic_nifti = nib.Nifti1Image(ic_nifti_data[..., i], preproc_nifti.affine, preproc_nifti.header)
        
        # Calculate Dice overlap with food vs nonfood contrast
        ic_data = ic_nifti.get_fdata()
        ic_data[np.abs(ic_data) < ic_thresh] = 0
        ic_data = ic_data > 0
        food_data = t_vol_thresh > 0
        dice = 2 * np.sum(ic_data * food_data) / (np.sum(ic_data) + np.sum(food_data))

        plot_stat_map(ic_nifti, title=f'{sub} IC {i+1}, Dice score: {dice}', display_mode='ortho', draw_cross=True, threshold=ic_thresh, radiological=True, bg_img=t1w_nifti, resampling_interpolation='nearest') 

    # fig.tight_layout()
    # plt.suptitle(f'{sub} ICA components')
    plt.show()
