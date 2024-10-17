# the code for generating masks for ABD-CT  

import os
import cv2
import numpy as np
import SimpleITK as itk
import matplotlib.pyplot as plt
import matplotlib
import cv2
import SimpleITK as itk

# Abd_CT dataset

abd_ct_pth = '../results//Abd_CT/'
abd_mri_pth = '../results/SABS/'
cmr_pth = './data/CMR/'

abd_mri_gt_pth = os.path.join(abd_mri_pth, 'label_1.nii.gz')                 # the ground truth mask, x is case number
abd_mri_img_pth = os.path.join(abd_mri_pth, 'image_1.nii.gz')                   # the image

abd_mri_liver_pth = os.path.join(abd_mri_pth, 'prediction_1_LIVER.nii.gz')        # the liver prediction of case x
abd_mri_spleen_pth = os.path.join(abd_mri_pth, 'prediction_1_SPLEEN.nii.gz')
# the spleen prediction of case x
abd_mri_rk_pth = os.path.join(abd_mri_pth, 'prediction_1_RK.nii.gz')              # the right kidney prediction of case x
abd_mri_lk_pth = os.path.join(abd_mri_pth, 'prediction_1_LK.nii.gz')              # the left kidney prediction of case x
output_path = "../results/SABSPNG/"
abd_mri_gt = itk.GetArrayFromImage(itk.ReadImage(abd_mri_gt_pth))
abd_mri_img = itk.GetArrayFromImage(itk.ReadImage(abd_mri_img_pth))


abd_mri_gt[abd_mri_gt == 200] = 1
abd_mri_gt[abd_mri_gt == 500] = 2
abd_mri_gt[abd_mri_gt == 600] = 3

# ********************************liver**********************************

abd_mri_liver = itk.GetArrayFromImage(itk.ReadImage(abd_mri_liver_pth))
abd_ct_liver = itk.GetArrayFromImage(itk.ReadImage(abd_mri_liver_pth))
abd_ct_liver_gt = 1 * (abd_mri_gt == 6)
idx = abd_ct_liver_gt.sum(axis=(1, 2)) > 0
abd_ct_liver_gt = abd_ct_liver_gt[idx]
abd_ct_img_liver = abd_mri_img[idx]
for i in range(abd_ct_img_liver.shape[0]):

    gt_slice = abd_ct_liver_gt[i] * 200  # choose the 16th slice of case x to illustrate, you also can choose other slices
    img_slice = abd_ct_img_liver[i]
    pred_slice = abd_ct_liver[i] * 200

    cv2.imwrite(os.path.join(output_path, f'abd_mri_liver_gt_{i}.png'), gt_slice)  # the ground truth
    cv2.imwrite(os.path.join(output_path, f'abd_mri_liver_img_{i}.png'), img_slice)
    cv2.imwrite(os.path.join(output_path, f'abd_mri_liver_pred_{i}.png'), pred_slice)  # the liver prediction

'''
'''
# **********************************spleen*********************************
abd_ct_spleen = itk.GetArrayFromImage(itk.ReadImage(abd_mri_spleen_pth))
abd_ct_spleen_gt = 1 * (abd_mri_gt == 1)
idx = abd_ct_spleen_gt.sum(axis=(1, 2)) > 0
abd_ct_spleen_gt = abd_ct_spleen_gt[idx]
abd_ct_img_spleen = abd_mri_img[idx]

for i in range(abd_ct_img_spleen.shape[0]):
    gt_slice = abd_ct_spleen_gt[i]*200
    img_slice = abd_ct_img_spleen[i]
    pred_slice = abd_ct_spleen[i]*200

    cv2.imwrite(os.path.join(output_path, f'abd_mri_spleen_gt_{i}.png'), gt_slice)  # the ground truth
    cv2.imwrite(os.path.join(output_path, f'abd_mri_spleen_img_{i}.png'), img_slice)
    cv2.imwrite(os.path.join(output_path, f'abd_mri_spleen_pred_{i}.png'), pred_slice)  # the liver prediction



# **********************************RK************************************
abd_ct_rk = itk.GetArrayFromImage(itk.ReadImage(abd_mri_rk_pth))
abd_ct_rk_gt = 1 * (abd_mri_gt == 2)
idx = abd_ct_rk_gt.sum(axis=(1, 2)) > 0
abd_ct_rk_gt = abd_ct_rk_gt[idx]
abd_ct_img_rk = abd_mri_img[idx]
for i in range(abd_ct_img_rk.shape[0]):
    gt_slice = abd_ct_rk_gt[i]*200
    img_slice = abd_ct_img_rk[i]
    pred_slice = abd_ct_rk[i]*200

    cv2.imwrite(os.path.join(output_path, f'abd_mri_rk_gt_{i}.png'), gt_slice)  # the ground truth
    cv2.imwrite(os.path.join(output_path, f'abd_mri_rk_img_{i}.png'), img_slice)
    cv2.imwrite(os.path.join(output_path, f'abd_mri_rk_pred_{i}.png'), pred_slice)  # the liver prediction


# *********************************LK**************************************
abd_mri_lk = itk.GetArrayFromImage(itk.ReadImage(abd_mri_lk_pth))
abd_mri_lk_gt = 1 * (abd_mri_gt == 3)
idx = abd_mri_lk_gt.sum(axis=(1, 2)) > 0
abd_mri_lk_gt = abd_mri_lk_gt[idx]
abd_mri_img_lk = abd_mri_img[idx]
abd_mri_lk = abd_mri_lk

for i in range(abd_mri_lk.shape[0]):
    gt_slice = abd_mri_lk_gt[i] * 200
    img_slice = abd_mri_img_lk[i]
    pred_slice = abd_mri_lk[i] * 200

    #abd_mri_liver_spt = abd_mri_liver_gt[5] * 200    # choose the 5th slice of case x as support image.
    #abd_mri_img_spt = abd_mri_img_liver[5] / 4.5
    #gt_slice = np.flip(gt_slice, axis=0)
   # img_slice = np.flip(img_slice, axis=0)
   # pred_slice = np.flip(pred_slice, axis=0)

    cv2.imwrite(os.path.join(output_path, f'abd_mri_lk_gt_{i}.png'), gt_slice)   # the ground truth
    cv2.imwrite(os.path.join(output_path, f'abd_mri_lk_img_{i}.png'), img_slice)
    cv2.imwrite(os.path.join(output_path, f'abd_mri_lk_pred_{i}.png'), pred_slice)


