import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def get_data(datapath):
    img = nib.load(datapath + 'raw_data_masked.nii.gz')
    data = img.get_data()
    mask = nib.load(datapath + 'raw_data_binary_mask.nii.gz').get_data()
    bvals, bvecs = read_bvals_bvecs(datapath + 'bvals', datapath + 'bvecs')
    gtab = gradient_table(bvals, bvecs, b0_threshold=30)
    return img, data, mask, gtab
