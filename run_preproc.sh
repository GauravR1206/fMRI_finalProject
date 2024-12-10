#!/bin/bash

data_dir="/nfs/masi/saundam1/datasets/food_viewing_fmri"

for subject in $(find $data_dir -maxdepth 2 -type d -name "sub-*" | sort | xargs -n1 basename); do
      echo "Running preprocessing for $subject..."

      # Get names of files
      t1w=$data_dir/ds000157/$subject/anat/${subject}_T1w.nii.gz
      fmri=$data_dir/ds000157/$subject/func/${subject}_task-passiveimageviewing_bold.nii.gz

      # Make an output directory
      output_dir=$data_dir/ds000157/derivatives/fsl-preproc/$subject
      mkdir -p $output_dir

      # Run BET on T1w
      bet $t1w $output_dir/${subject}_T1w_brain.nii.gz -o -m

      # Motion correction
      mcflirt -in $fmri \
            -out $output_dir/${subject}_mccorr.nii.gz \
            -mats -plots

      # Get first volume for registration
      fslroi $output_dir/${subject}_mccorr.nii.gz \
            $output_dir/${subject}_mccorr_vol0.nii.gz 0 1

      # Align fMRI to T1
      epi_reg --epi=$output_dir/${subject}_mccorr_vol0.nii.gz \
            --t1=$t1w --t1brain=$output_dir/${subject}_T1w_brain.nii.gz \
            --out=$output_dir/${subject}_mccorr_vol0_to_anat

      # Affine registration (T1w to MNI)
      flirt -in $output_dir/${subject}_T1w_brain.nii.gz \
            -ref $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
            -out $output_dir/${subject}_T1w_to_MNI_aff -omat $output_dir/${subject}_T1w_to_MNI_aff.mat
      
      # Nonlinear registration (T1w to MNI)
      fnirt --ref=$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz \
            --in=$t1w \
            --iout=$output_dir/${subject}_T1w_to_MNI_nlin.nii.gz \
            --cout=$output_dir/${subject}_T1w_to_MNI_nlin.mat \
            --aff=$output_dir/${subject}_T1w_to_MNI_aff.mat \
            --config=T1_2_MNI152_2mm \
            --lambda=400,200,150,75,60,45 \
            --warpres=6,6,6 \
            --verbose

      # Apply warp to fMRI data
      applywarp --ref=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
            --in=$output_dir/${subject}_mccorr.nii.gz \
            --out=$output_dir/${subject}_mccorr_to_MNI.nii.gz \
            --warp=$output_dir/${subject}_T1w_to_MNI_nlin.mat.nii.gz \
            --premat=$output_dir/${subject}_mccorr_vol0_to_anat.mat \
            --verbose

      # Warp mask to MNI
      applywarp --ref=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
            --in=$output_dir/${subject}_T1w_brain_mask.nii.gz \
            --out=$output_dir/${subject}_T1w_brain_mask_to_MNI.nii.gz \
            --warp=$output_dir/${subject}_T1w_to_MNI_nlin.mat.nii.gz \
            --interp=nn

      # Smooth fMRI data
      fslmaths $output_dir/${subject}_mccorr_to_MNI.nii.gz \
            -kernel gauss 3 -fmean $output_dir/${subject}_bold_preproc.nii.gz

      # Mask fMRI data
      fslmaths $output_dir/${subject}_bold_preproc.nii.gz \
            -mas $output_dir/${subject}_T1w_brain_mask_to_MNI.nii.gz \
            $output_dir/${subject}_bold_preproc_masked.nii.gz
done
