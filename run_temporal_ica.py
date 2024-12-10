import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from pathlib import Path

data_dir = Path('/nfs/masi/saundam1/datasets/food_viewing_fmri/ds000157')
preproc_dir = data_dir / 'derivatives' / 'fsl-preproc'
TR = 1.6

for sub in ['sub-01']:
    preproc_nifti = nib.load(preproc_dir / sub / f'{sub}_bold_preproc_masked.nii.gz')
    mask_nifti = nib.load(preproc_dir / sub / f'{sub}_T1w_brain_mask_to_MNI.nii.gz')

    preproc_data = preproc_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

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
    ica = FastICA(n_components=8)
    X_ica = ica.fit_transform(X_pca[:chosen_components].T).T

    print(f'{X_ica.shape} ICA components x timepoints')

    # Plot the ICA components
    fig, axes = plt.subplots(4, 2, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(time, X_ica[i])
        ax.set_title(f'IC {i+1}')
        ax.set_xlabel('Time (s)')

    fig.tight_layout()
    plt.suptitle(f'{sub} ICA components')
    plt.show()
