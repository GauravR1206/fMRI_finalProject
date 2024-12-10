import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy import stats

events = pd.read_csv('task-checkerboard_events.tsv', delimiter='\t')

nifti_file = 'fmri_blockDes.nii'
img = nib.load(nifti_file)
image_data = img.get_fdata()

hdr = img.header
zooms = hdr.get_zooms()

TR = zooms[3]  
tmax = 30        
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

plt.figure(figsize=(8, 4))
plt.plot(t, h, label='HRF Model')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Response')
plt.title('Hemodynamic Response Function (HRF) Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

tsv_file = 'task-checkerboard_events.tsv'
events_df = pd.read_csv(tsv_file, sep='\t')

print("\nEvent Timings:")
print(events_df.head())


nframes = image_data.shape[-1]
stimulus = np.zeros(nframes)

for index, row in events_df.iterrows():
    onset = row['onset'] 
    duration = row['duration']
    
    start_time = onset
    end_time = onset + duration
    
    start_idx = int(np.floor(start_time / TR))
    end_idx = int(np.floor(end_time / TR))
    
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, nframes)
    
    stimulus[start_idx:end_idx] = 1

plt.figure(figsize=(10, 3))
time_vector = np.arange(nframes) * TR
plt.step(time_vector, stimulus, where='post', color='blue', label='Stimulus')
plt.xlabel('Time (s)')
plt.ylabel('Stimulus')
plt.title('Binary Stimulus Vector')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

convolved = np.convolve(stimulus, h)[:nframes]

regressor = convolved / np.max(convolved)

plt.figure(figsize=(10, 3))
plt.plot(time_vector, regressor, color='purple', label='Convolved Regressor')
plt.xlabel('Time (s)')
plt.ylabel('Regressor Value')
plt.title('Regressor: Stimulus Convolved with HRF')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
intercept = np.ones(nframes)

linear_trend = np.linspace(0, 1, nframes)

quadratic_trend = linear_trend ** 2

X = np.column_stack((intercept, regressor, linear_trend, quadratic_trend))

X[:, 2] = X[:, 2] - np.mean(X[:, 2])
X[:, 3] = X[:, 3] - np.mean(X[:, 3])

print("\nDesign Matrix (X) shape:", X.shape)

plt.figure(figsize=(8, 6))
plt.imshow(X, aspect='auto', cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('Predictors')
plt.ylabel('Time Frames')
plt.title('Design Matrix (X)')
plt.xticks(ticks=np.arange(X.shape[1]), labels=['Intercept', 'Regressor', 'Linear Trend', 'Quadratic Trend'])
plt.yticks([])
plt.tight_layout()
plt.show()

fmri_data = img.get_fdata()
x_dim, y_dim, z_dim, t_dim = fmri_data.shape
print(f"Spatial Dimensions: X={x_dim}, Y={y_dim}, Z={z_dim}")

Y = fmri_data.reshape(-1, t_dim).T  
print(f"Reshaped fMRI Data Y shape: {Y.shape}")

X_pinv = pinv(X) 

Beta = X_pinv @ Y  

print(f"Beta Matrix shape: {Beta.shape}")  

alpha = 0.001

df = nframes - X.shape[1]
print(f"Degrees of Freedom (df): {df}")

t_critical = stats.t.ppf(1 - alpha/2, df)
print(f"Critical t-value for a two-sided test at alpha={alpha}: {t_critical:.4f}")


c = np.array([0, 1, 0, 0]) 

t_num = c @ Beta 


XtX_inv = np.linalg.inv(X.T @ X)  
cXtXc = c @ XtX_inv @ c 


err = Y - X @ Beta 


errorVar = (1 / df) * np.sum(err**2, axis=0) 


t_den = np.sqrt(errorVar * cXtXc) 

t_stat = t_num / t_den 

t_vol = t_stat.reshape(x_dim, y_dim, z_dim)
print(f"t_vol shape: {t_vol.shape}")

significant_mask = np.abs(t_vol) > t_critical

num_voxels_positive = np.sum(t_vol > t_critical)

num_voxels_negative = np.sum(t_vol < -t_critical)

total_significant_voxels = num_voxels_positive + num_voxels_negative

print(f"Number of voxels with t > {t_critical:.2f}: {num_voxels_positive}")
print(f"Number of voxels with t < {-t_critical:.2f}: {num_voxels_negative}")
print(f"Total number of significant voxels: {total_significant_voxels}")

voxel_x, voxel_y, voxel_z = 20, 20, 12

voxel_time_course = fmri_data[voxel_x, voxel_y, voxel_z, :]  
print(f"Voxel Time Course Shape: {voxel_time_course.shape}")

plt.figure(figsize=(14, 6))

plt.plot(time_vector, regressor, color='blue', label='Stimulus Regressor')

plt.plot(time_vector, voxel_time_course, color='red', label='Voxel Time Course')

plt.xlabel('Time (s)')
plt.ylabel('Signal Intensity')
plt.title(f'Time Course of Voxel (X={voxel_x}, Y={voxel_y}, Z={voxel_z})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(14, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Stimulus Regressor', color=color1)
ax1.plot(time_vector, regressor, color=color1, label='Stimulus Regressor')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

ax2 = ax1.twinx() 

color2 = 'tab:red'
ax2.set_ylabel('Voxel Signal Intensity', color=color2)
ax2.plot(time_vector, voxel_time_course, color=color2, label='Voxel Time Course')
ax2.tick_params(axis='y', labelcolor=color2)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.title(f'Time Course of Voxel (X={voxel_x}, Y={voxel_y}, Z={voxel_z})')
fig.tight_layout()
plt.show()
