#!/bin/bash

# Example input:
#   00a369b4ec1e5e0ff34e6bd838e5f2d6_L_MLO.nii.gz

filename=$1

# Remove extension (works for .nii.gz)
filename_without_extension=$(basename "${filename}" .nii.gz)

# Extract patient ID (first field before '_')
patid=$(echo "${filename_without_extension}" | cut -d'_' -f1)

# Extract view type (second and third fields joined by '_')
type=$(echo "${filename_without_extension}" | cut -d'_' -f2-3)

echo "Start processing: ${filename}"

# Flip view side (L <-> R) while keeping view type (MLO/CC)
if [[ "${type}" == "L_MLO" ]]; then
  type="R_MLO"
elif [[ "${type}" == "R_MLO" ]]; then
  type="L_MLO"
elif [[ "${type}" == "L_CC" ]]; then
  type="R_CC"
elif [[ "${type}" == "R_CC" ]]; then
  type="L_CC"
fi

# Path settings
fixed_image_base="/workspace/data/VinDr-Mammo/Normal/image"
moving_image_base="/workspace/data/VinDr-Mammo/Normal/image"
output_base="/workspace/data/VinDr-Mammo/Normal/registered_oppositeSide_image"
tmp_base="/workspace/data/VinDr-Mammo/tmp-normal"
error_log="/workspace/data/VinDr-Mammo/Error_Registration_Normal.txt"

# Create error log if missing
touch "${error_log}"

fixed_image="${fixed_image_base}/${filename}"
moving_image="${moving_image_base}/${patid}_${type}.nii.gz"
fixed_image_2d="${tmp_base}/${patid}_${filename_without_extension}_2d.nii.gz"
moving_image_2d="${tmp_base}/${patid}_${filename_without_extension}_${type}_2d.nii.gz"
flipped_moving_image="${tmp_base}/${patid}_${filename_without_extension}_${type}_flipped.nii.gz"
output_prefix="${tmp_base}/${patid}_${filename_without_extension}_"
registered_image="${output_base}/${patid}_${filename}"

echo "Moving image: ${moving_image}"

# Check if input image files exist
if [ ! -f "${fixed_image}" ] || [ ! -f "${moving_image}" ]; then
  echo "Missing image file: ${patid}_${filename}" >> "${error_log}"
  exit
fi

# Extract 2D slice from fixed image
fslroi "$fixed_image" "$fixed_image_2d" 0 -1 0 -1 0 1
# Extract 2D slice from moving image
fslroi "$moving_image" "$moving_image_2d" 0 -1 0 -1 0 1

# Flip moving image horizontally
fslswapdim "$moving_image_2d" -x y z "$flipped_moving_image"

# Perform registration
antsRegistrationSyN.sh -d 2 -f "$fixed_image_2d" -m "$flipped_moving_image" -o "${output_prefix}" -t r

# Apply transformation to moving image
antsApplyTransforms -d 2 -r "$fixed_image_2d" -i "$flipped_moving_image" -o "$registered_image" -n LanczosWindowedSinc -t "${output_prefix}Warped.nii.gz" -t "${output_prefix}0GenericAffine.mat"

# Clean up temporary files
rm "${output_prefix}"*

