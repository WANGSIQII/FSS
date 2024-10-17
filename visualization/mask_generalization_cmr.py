# the code for generating masks for CMR

import os
import cv2
import numpy as np
import SimpleITK as itk
import matplotlib.pyplot as plt
import matplotlib
import cv2
import SimpleITK as itk

# CMR dataset

abd_ct_pth = 'data/Abd_CT/'
abd_mri_pth = './data/Abd_MRI/'
cmr_pth = '../results/CMR/'

cmr_gt_pth = os.path.join(cmr_pth, 'label_4.nii.gz')                     # the ground truth mask, x is case number
cmr_img_pth = os.path.join(cmr_pth, 'image_4.nii.gz')                       # the image
cmr_lvbp_pth = os.path.join(cmr_pth, 'prediction_4_LV-BP.nii.gz')         # the LV-BP prediction of case x
cmr_lvmyo_pth = os.path.join(cmr_pth, 'prediction_4_LV-MYO.nii.gz')       # the LV-MYO prediction of case x
cmr_rv_pth = os.path.join(cmr_pth, 'prediction_4_RV.nii.gz')              # the RV prediction of case x

cmr_gt = itk.GetArrayFromImage(itk.ReadImage(cmr_gt_pth))  
cmr_img = itk.GetArrayFromImage(itk.ReadImage(cmr_img_pth))  

cmr_gt[cmr_gt == 200] = 1
cmr_gt[cmr_gt == 500] = 2
cmr_gt[cmr_gt == 600] = 3
output_path = "../results/CMRPNG/"
# ********************************lvbp**********************************
cmr_lvbp = itk.GetArrayFromImage(itk.ReadImage(cmr_lvbp_pth))
cmr_lvbp_gt = 1 * (cmr_gt == 2)
idx = cmr_lvbp_gt.sum(axis=(1, 2)) > 0
cmr_lvbp_gt = cmr_lvbp_gt[idx]
cmr_img_lvbp = cmr_img[idx]

for i in range(cmr_img_lvbp.shape[0]):

    gt_slice = cmr_lvbp_gt[i]*200                                   # choose the 4th slice of case x to illustrate, you also can choose other slices
    img_slice = cmr_img_lvbp[i] / 4.5
    pred_slice = cmr_lvbp[i]*200

    cv2.imwrite(os.path.join(output_path, f'cmr_lvbp_gt_{i}.png'), gt_slice)  # the ground truth
    cv2.imwrite(os.path.join(output_path, f'cmr_lvbp_img_{i}.png'), img_slice)
    cv2.imwrite(os.path.join(output_path, f'cmr_lvbp_pred_{i}.png'), pred_slice)
           # the support image

# **********************************lvmyo*********************************
cmr_lvmyo = itk.GetArrayFromImage(itk.ReadImage(cmr_lvmyo_pth))
cmr_lvmyo_gt = 1 * (cmr_gt == 1)
idx = cmr_lvmyo_gt.sum(axis=(1, 2)) > 0
cmr_lvmyo_gt = cmr_lvmyo_gt[idx]
cmr_img_lvmyo = cmr_img[idx]

for i in range(cmr_img_lvmyo.shape[0]):

    gt_slice = cmr_lvmyo_gt[i] * 200
    img_slice = cmr_img_lvmyo[i] / 4.5
    pred_slice =  cmr_lvmyo[i] * 200


    cv2.imwrite(os.path.join(output_path, f'cmr_lvmyo_gt_{i}.png'), gt_slice)  # the ground truth
    cv2.imwrite(os.path.join(output_path, f'cmr_lvmyo_img_{i}.png'), img_slice)
    cv2.imwrite(os.path.join(output_path, f'cmr_lvmyo_pred_{i}.png'), pred_slice)

# **********************************rv************************************
cmr_rv = itk.GetArrayFromImage(itk.ReadImage(cmr_rv_pth))
cmr_rv_gt = 1 * (cmr_gt == 3)
idx = cmr_rv_gt.sum(axis=(1, 2)) > 0
cmr_rv_gt = cmr_rv_gt[idx]
cmr_img_rv = cmr_img[idx]

for i in range(cmr_img_rv.shape[0]):
    gt_slice = cmr_rv_gt[i] * 200
    img_slice = cmr_img_rv[i] / 4.5
    pred_slice =  cmr_rv[i] * 200
    cv2.imwrite(os.path.join(output_path, f'cmr_rv_gt_{i}.png'), gt_slice)  # the ground truth
    cv2.imwrite(os.path.join(output_path, f'cmr_rv_img_{i}.png'), img_slice)
    cv2.imwrite(os.path.join(output_path, f'cmr_rv_pred_{i}.png'), pred_slice)




