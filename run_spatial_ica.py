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
        plot_stat_map(ic_nifti, title=f'{sub} IC {i+1}', display_mode='ortho', draw_cross=True, threshold=ic_thresh, radiological=True, bg_img=t1w_nifti, resampling_interpolation='nearest') 

    # fig.tight_layout()
    # plt.suptitle(f'{sub} ICA components')
    plt.show()
