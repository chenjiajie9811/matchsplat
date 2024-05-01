from packaging import version
import torch
import numpy as np
import matplotlib.colors as colors
import torch.nn.functional as F
import torch.nn as nn
import os, struct, math
import numpy as np
import functools
import cv2
from typing import Any, List, Union, Tuple
from glob import glob
import collections
from copy import deepcopy

def flow2kps(trg_kps, flow, n_pts, upsample_size=(256, 256)):
    _, _, h, w = flow.size()

    flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
    
    src_kps = []
    mask_list = []
    for trg_kps, flow in zip(trg_kps.long(), flow):
        size = trg_kps.size(0)
        mask_list.append(((0<=trg_kps) & (trg_kps<256))[:,0] & ((0<=trg_kps) & (trg_kps<256))[:,1])
        kp = torch.clamp(trg_kps.transpose(0,1).narrow_copy(1, 0, n_pts), 0, upsample_size[0]-1)
        estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
        
      
     
        src_kps.append(estimated_kps)

    return torch.stack(src_kps),  torch.stack(mask_list)


def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        # torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(1,2,0).float()
        return mapping.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                mapping[i, :, :, 0] = flow[i, :, :, 0] + X
                mapping[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.transpose(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            mapping[:, :, 0] = flow[:, :, 0] + X
            mapping[:, :, 1] = flow[:, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(2, 0, 1)
        return mapping.astype(np.float32)


def get_gt_correspondence_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image). """

    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    if isinstance(mapping, np.ndarray):
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[:, 0] >= 0, mapping[:, 0] <= w-1)
            mask_y = np.logical_and(mapping[:, 1] >= 0, mapping[:, 1] <= h-1)
            mask = np.logical_and(mask_x, mask_y)
        else:
            _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[0] >= 0, mapping[0] <= w - 1)
            mask_y = np.logical_and(mapping[1] >= 0, mapping[1] <= h - 1)
            mask = np.logical_and(mask_x, mask_y)
        mask = mask.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") else mask.astype(np.uint8)
    else:
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask = mapping[:, 0].ge(0) & mapping[:, 0].le(w-1) & mapping[:, 1].ge(0) & mapping[:, 1].le(h-1)
        else:
            _, h, w = mapping.shape
            mask = mapping[0].ge(0) & mapping[0].le(w-1) & mapping[1].ge(0) & mapping[1].le(h-1)
        mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
    return mask


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    Args:
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy),1).float()

    if x.is_cuda:
        grid = grid.to(flo.device)
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    output = nn.functional.grid_sample(x, vgrid)
    return output


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def two_view_geometry(intrinsics1, intrinsics2, rel_pose_est, rel_pose_gt):
    relative_pose = rel_pose_est
    R = relative_pose[:,:3, :3].detach()
    T = relative_pose[:, :3, 3].detach()
   
    tx = []
    for i in range(len(T)):
        tx.append(skew(T[i].cpu().numpy()))
    tx = np.stack(tx)
 
    E = np.matmul(tx, R.cpu().numpy())
    F = []
    for i in range(len(T)):
        F.append(np.linalg.inv(intrinsics2[ i, :3, :3].cpu().numpy()).T.dot(E[i]).dot(np.linalg.inv(intrinsics1[i, :3, :3].cpu().numpy())))

    relative_pose_gt = rel_pose_gt
    R_gt = relative_pose_gt[:,:3, :3].detach()
    T_gt = relative_pose_gt[:, :3, 3].detach()
    tx_gt = []
    for i in range(len(T)):
        tx_gt.append( skew(T_gt[i].cpu().numpy()))
    tx_gt = np.stack(tx_gt)
    E_gt = np.matmul(tx_gt, R_gt.cpu().numpy())
    F_gt = []
    for i in range(len(T)):
        F_gt.append(np.linalg.inv(intrinsics2[ i, :3, :3].cpu().numpy()).T.dot(E_gt[i]).dot(np.linalg.inv(intrinsics1[i, :3, :3].cpu().numpy())))
   



    return E, F, relative_pose,  E_gt, F_gt, relative_pose_gt


def drawpointslines(img1, pts1, img2, lines2, colors):
    
    h, w = img2.shape[:2]

    for p, l, c in zip(pts1, lines2, colors):
        c = tuple(c.tolist())
        img1 = cv2.circle(img1, tuple(p), 4, c, -1)

        x0, y0 = map(int, [0, -l[2]/l[1]])
        x1, y1 = map(int, [w, -(l[2]+l[0]*w)/l[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), c,10)
    return img1, img2

def drawpoint(img1, pts1, img2, pts2, colors):
        # 
    for p2, color in zip(pts2, colors):
        
        color = tuple(color.tolist())
        img1 = cv2.circle((img1)  , tuple(p2), 5, color, -1)
        
        img2 = cv2.circle((img2  ) , (int(pts1[p2[0] + (p2[1] * 256)].round()[0]),int(pts1[p2[0] + (p2[1] * 256)].round()[1]) ), 5, color, -1)


    return img1, img2

def inspect(img1, K1, img2, K2, rel_pose_est, rel_pose_gt):
    E, F, relative_pose, E_gt, F_gt, relative_pose_gt = two_view_geometry(K1, K2, rel_pose_est, rel_pose_gt)
    
    # try:
    # orb = cv2.ORB_create()
    img = []
    img_gt = []
    
    pts1 = np.array([[64,64], [64,128], [64,192], [128,64], [128,128], [128,192], [192,64], [192,128], [192,192]])
    colors = np.array([[63,228,92], [222, 155, 167], [ 56, 220, 130],  [216,  43, 206], [ 47, 172,  72],  [198, 181,   0], [137,  99, 246],  [ 22, 160,  10], [ 23, 240, 252]  ]   )

    for i in range(len(E)):
        
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F[i])
        lines2 = lines2.reshape(-1, 3)

        lines2_gt = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F_gt[i])
        lines2_gt = lines2_gt.reshape(-1, 3)


        im1, im2 = drawpointslines(
            (img1[i].cpu().numpy().copy() + 1) * 127.5, pts1, 
            (img2[i].cpu().numpy().copy() + 1) * 127.5, lines2, colors)

        im1_gt, im2_gt = drawpointslines(
            (img1[i].cpu().numpy().copy() + 1) * 127.5, pts1, 
            (img2[i].cpu().numpy().copy() + 1) * 127.5, lines2_gt, colors)
        
        im2_copy = ((img2[i].cpu().numpy().copy() + 1) * 127.5).copy()
        im2_gt_copy =((img2[i].cpu().numpy().copy() + 1) * 127.5).copy()

        # Define the alpha value to control line intensity reduction (0.0 to 1.0)
        alpha = 0.5  # You can adjust this value to control the intensity

        # Blend the lines with im2 and im2_gt using addWeighted
        im2 = cv2.addWeighted(im2,  0.4, im2_copy, 0.6, 0)
        im2_gt = cv2.addWeighted(im2_gt, 0.4, im2_gt_copy, 0.6, 0)

        im_to_show = np.concatenate((im1, im2), axis=1)

        im_to_show_gt = np.concatenate((im1_gt, im2_gt), axis=1)
        img.append(im_to_show)
        img_gt.append(im_to_show_gt)
    
    img= np.stack(img)
    img_gt = np.stack(img_gt)


    return img, img_gt
    # except:
    #     return None, None
        

