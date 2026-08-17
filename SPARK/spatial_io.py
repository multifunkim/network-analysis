#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spatial_io.py
=============

Format-aware spatial I/O utilities for SPARK.

Currently supported input formats
---------------------------------
1. NIfTI
   - .nii
   - .nii.gz

2. GIFTI functional surface
   - .func.gii

The purpose of this module is to separate neuroimaging file-format
handling from the core SPARK algorithm.

All input formats are converted internally to:

    tseries : (T, V)

where:
    T = number of time points
    V = number of spatial elements

For NIfTI:
    V = retained voxels

For GIFTI:
    V = retained surface vertices

Future formats such as CIFTI can be added here without changing
SPARK Steps 2–5.
"""

import os
import copy
import numpy as np
import nibabel as nib


# ============================================================
# FORMAT DETECTION
# ============================================================

def detect_format(path):
    """
    Detect neuroimaging input format from filename.

    Returns
    -------
    str
        "nifti" or "gifti"
    """

    path_lower = str(path).lower()

    if path_lower.endswith(".nii") or path_lower.endswith(".nii.gz"):
        return "nifti"

    if path_lower.endswith(".func.gii"):
        return "gifti"

    raise ValueError(
        f"Unsupported input format: {path}\n"
        "Currently supported formats are:\n"
        "  .nii\n"
        "  .nii.gz\n"
        "  .func.gii"
    )


# ============================================================
# NIFTI LOADER
# ============================================================

def load_nifti_timeseries(fmri_path, mask_path):
    """
    Load a 4D NIfTI fMRI dataset and apply a 3D NIfTI mask.

    Parameters
    ----------
    fmri_path : str
        Path to 4D NIfTI fMRI image.

    mask_path : str
        Path to 3D NIfTI mask.

    Returns
    -------
    result : dict

        result["tseries"]
            (T, V) time-series matrix.

        result["spatial_index"]
            (V, 3) voxel coordinates.

        result["n_spatial_original"]
            Number of voxels before zero-variance removal.

        result["image_shape"]
            Original spatial NIfTI shape.

        result["affine"]
            NIfTI affine.

        result["format"]
            "nifti"
    """

    if mask_path is None:
        raise ValueError(
            "A NIfTI input requires --mask_path."
        )

    img = nib.load(fmri_path)
    data = img.get_fdata()

    if data.ndim != 4:
        raise ValueError(
            f"NIfTI input must be 4D (X,Y,Z,T). "
            f"Received shape: {data.shape}"
        )

    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata() > 0

    if mask.ndim != 3:
        raise ValueError(
            f"NIfTI mask must be 3D. Received shape: {mask.shape}"
        )

    if tuple(data.shape[:3]) != tuple(mask.shape):
        raise ValueError(
            "Spatial dimensions of fMRI and mask do not match:\n"
            f"  fMRI: {data.shape[:3]}\n"
            f"  mask: {mask.shape}"
        )

    # Voxel coordinates inside mask
    coords = np.column_stack(np.where(mask))

    # Convert X×Y×Z×T -> T×V
    flat_mask = mask.reshape(-1)

    tseries = data.reshape(
        -1,
        data.shape[-1]
    )[flat_mask].T

    return {
        "tseries": np.asarray(tseries, dtype=np.float64),
        "spatial_index": np.asarray(coords, dtype=np.int64),
        "n_spatial_original": int(coords.shape[0]),
        "image_shape": np.asarray(img.shape[:3], dtype=np.int64),
        "affine": np.asarray(img.affine, dtype=np.float64),
        "format": "nifti",
    }


# ============================================================
# GIFTI LOADER
# ============================================================

def load_gifti_timeseries(gifti_path):
    """
    Load a functional GIFTI time-series.

    SPARK internal convention is:

        T × V

    where:
        T = time points
        V = vertices

    GIFTI functional files commonly contain one DataArray per
    time point. This loader stacks them explicitly to guarantee
    the correct orientation.

    Parameters
    ----------
    gifti_path : str
        Path to .func.gii file.

    Returns
    -------
    result : dict

        result["tseries"]
            (T, V) time-series matrix.

        result["spatial_index"]
            (V,) original vertex indices.

        result["n_spatial_original"]
            Original number of surface vertices.

        result["format"]
            "gifti"

        result["gifti_path"]
            Original GIFTI path.
    """

    img = nib.load(gifti_path)

    if not isinstance(img, nib.gifti.GiftiImage):
        raise TypeError(
            f"Expected GIFTI image but received: {type(img)}"
        )

    if len(img.darrays) == 0:
        raise ValueError(
            f"No DataArrays found in GIFTI file: {gifti_path}"
        )

    arrays = [
        np.asarray(darray.data)
        for darray in img.darrays
    ]

    # ---------------------------------------------------------
    # Most functional GIFTIs:
    #
    # one DataArray = one time point
    #
    # therefore stacking gives:
    #
    #     T × V
    # ---------------------------------------------------------

    if len(arrays) > 1:

        lengths = [a.size for a in arrays]

        if len(set(lengths)) != 1:
            raise ValueError(
                "GIFTI DataArrays have inconsistent sizes. "
                "Cannot construct a T×V matrix."
            )

        tseries = np.vstack([
            a.reshape(1, -1)
            for a in arrays
        ])

    else:
        # Some files may contain the full matrix in one DataArray.
        arr = np.asarray(arrays[0])

        if arr.ndim == 1:
            # Single surface map -> one time point
            tseries = arr.reshape(1, -1)

        elif arr.ndim == 2:
            # We need to determine orientation.
            #
            # Functional imaging usually has:
            # vertices >> time points.
            #
            # We therefore standardize to T×V.

            if arr.shape[0] <= arr.shape[1]:
                tseries = arr
            else:
                tseries = arr.T

        else:
            raise ValueError(
                "Unsupported GIFTI DataArray dimensionality: "
                f"{arr.shape}"
            )

    tseries = np.asarray(tseries, dtype=np.float64)

    if tseries.ndim != 2:
        raise ValueError(
            f"GIFTI time-series must become a 2D T×V matrix. "
            f"Received shape: {tseries.shape}"
        )

    n_vertices = int(tseries.shape[1])

    vertex_indices = np.arange(
        n_vertices,
        dtype=np.int64
    )

    return {
        "tseries": tseries,
        "spatial_index": vertex_indices,
        "n_spatial_original": n_vertices,
        "format": "gifti",
        "gifti_path": os.path.abspath(gifti_path),
    }


# ============================================================
# GENERIC LOADER
# ============================================================

def load_timeseries(input_path, mask_path=None):
    """
    Automatically load supported SPARK input.

    Returns a common internal representation regardless of
    input file format.
    """

    input_format = detect_format(input_path)

    if input_format == "nifti":

        result = load_nifti_timeseries(
            input_path,
            mask_path
        )

    elif input_format == "gifti":

        result = load_gifti_timeseries(
            input_path
        )

    else:
        # Defensive branch for future formats.
        raise ValueError(
            f"No loader implemented for format: {input_format}"
        )

    # Common validation
    X = result["tseries"]

    if X.ndim != 2:
        raise ValueError(
            f"Internal SPARK representation must be T×V. "
            f"Received: {X.shape}"
        )

    if X.shape[0] < 2:
        raise ValueError(
            f"Input contains only {X.shape[0]} time point(s)."
        )

    if X.shape[1] < 1:
        raise ValueError(
            "Input contains no spatial elements."
        )

    return result


# ============================================================
# GIFTI OUTPUT
# ============================================================

def save_gifti_map(
    values,
    output_path,
    template_path=None,
):
    """
    Save one SPARK surface map as a functional GIFTI.

    Parameters
    ----------
    values : array-like, shape (V,)
        Surface values.

    output_path : str
        Output .func.gii path.

    template_path : str, optional
        Original GIFTI file. Metadata from its first DataArray
        is reused where possible.
    """

    values = np.asarray(
        values,
        dtype=np.float32
    ).reshape(-1)

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Use original GIFTI metadata where possible.
    # ---------------------------------------------------------

    if template_path is not None:

        template = nib.load(template_path)

        if not isinstance(
            template,
            nib.gifti.GiftiImage
        ):
            raise TypeError(
                f"Template is not GIFTI: {template_path}"
            )

        new_img = nib.gifti.GiftiImage()

        # Preserve global metadata
        try:
            new_img.meta = copy.deepcopy(template.meta)
        except Exception:
            pass

        # Preserve label table if present
        try:
            new_img.labeltable = copy.deepcopy(
                template.labeltable
            )
        except Exception:
            pass

        # Use metadata from first DataArray if possible
        if len(template.darrays) > 0:

            source_da = template.darrays[0]

            new_da = nib.gifti.GiftiDataArray(
                data=values,
                intent="NIFTI_INTENT_NONE",
                datatype="NIFTI_TYPE_FLOAT32",
            )

            try:
                new_da.meta = copy.deepcopy(source_da.meta)
            except Exception:
                pass

            try:
                new_da.coordsys = copy.deepcopy(
                    source_da.coordsys
                )
            except Exception:
                pass

        else:

            new_da = nib.gifti.GiftiDataArray(
                data=values,
                intent="NIFTI_INTENT_NONE",
                datatype="NIFTI_TYPE_FLOAT32",
            )

    else:

        new_img = nib.gifti.GiftiImage()

        new_da = nib.gifti.GiftiDataArray(
            data=values,
            intent="NIFTI_INTENT_NONE",
            datatype="NIFTI_TYPE_FLOAT32",
        )

    new_img.add_gifti_data_array(new_da)

    nib.save(
        new_img,
        output_path
    )


# ============================================================
# INFORMATION HELPER
# ============================================================

def describe_input(input_path, mask_path=None):
    """
    Load an input and return basic dimensionality information.

    Useful for debugging and pipeline logging.
    """

    result = load_timeseries(
        input_path,
        mask_path
    )

    X = result["tseries"]

    return {
        "format": result["format"],
        "time_points": int(X.shape[0]),
        "spatial_elements": int(X.shape[1]),
    }
