import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from .copo_geometry import (
    warp, convert_flow_to_mapping,
    inspect, drawpoint, drawpointslines, two_view_geometry
)
import torch
from packaging import version
import cv2
from torch import nn
import torchvision


def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def overlay_semantic_mask(im, ann, alpha=0.5, mask=None, colors=None, color=[255, 218, 185], contour_thickness=1):
    """
    example usage:
    image_overlaid = overlay_semantic_mask(im.astype(np.uint8), 255 - mask.astype(np.uint8) * 255, color=[255, 102, 51])
    """
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.uint8)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)
    colors[-1, :] = color

    if mask is None:
        mask = colors[ann]

    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]  # where the mask is zero (where object is), shoudlnt be any color

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, color,
                             contour_thickness)
    return img

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

def copo_summary(
        batch, 
        img_shape=(256, 256)):

     #(2, 256, 512, 3),  (2, 256, 512, 3)
    epipolar_pred, epipolar_gt = inspect(
        batch['context']['image'][:, 1].permute(0, 2, 3, 1), 
        batch['context']['intrinsics'][:, 1],
        batch['context']['image'][:, 0].permute(0, 2, 3, 1), 
        batch['context']['intrinsics'][:, 0],
        batch['rel_pose'],
        batch['gt_rel_pose'],
    )


    b, _, h, w = batch['flow'][0].size()
    # batch['flow']  -> list (4) -> each with size (b, 2, 64, 64)

    flow = F.interpolate(batch['flow'][0], 256, mode='bilinear') * (256 / h) #torch.Size([b, 2, 256, 256])
    flow2 = F.interpolate(batch['flow'][1], 256, mode='bilinear') * (256 / h) #torch.Size([b, 2, 256, 256])
    
    cyclic_consistency_error = torch.norm(flow + warp(flow2, flow), dim=1).le(10) # torch.Size([b, 256, 256]), .le(10) -> less than 10
    cyclic_consistency_error2 = torch.norm(flow2 + warp(flow, flow2), dim=1).le(10) # torch.Size([b, 256, 256])
    mask_padded = cyclic_consistency_error * get_gt_correspondence_mask(flow)
    mask_padded2 = cyclic_consistency_error2 * get_gt_correspondence_mask(flow2)

    context_imgs = batch['context']['image']
    masked_imgs = []

    for i in range(b):
        context1 = context_imgs[i, 0].clone().detach() # (c h, w)
        context2 = context_imgs[i, 1].clone().detach()

        context1[:, ~mask_padded2[i]] = 0.
        context2[:, ~mask_padded[i]] = 0.

        masked_imgs.append(context1)
        masked_imgs.append(context2)

    masked_imgs = torch.stack(masked_imgs)

    #TODO extract keypoints from flow


    warped_img = []
    warped_img_mask = []
    for i in range(len(flow)):
        temp = warp((batch['context']['image'][i, 1].unsqueeze(0) + 1) * 127.5, flow[i]).squeeze(0)
        warped_img.append(temp)
        warped_img_mask.append(
            overlay_semantic_mask(
                temp.permute(1, 2, 0).detach().cpu().numpy(), 
                255 - mask_padded[i].detach().cpu().numpy() * 255, 
                color=[255, 102, 51])
        )
     
    warped_img = torch.stack(warped_img).to(flow.device) #torch.Size([2, 3, 256, 256])
    warped_img_mask = np.stack(warped_img_mask) #(2, 256, 256, 3)

    ret = {}
    ret["epipolar_pred"] = epipolar_pred        # np(2, 256, 512, 3)
    ret["epipolar_gt"] = epipolar_gt            # np(2, 256, 512, 3)
    ret["warped_img"] = warped_img              # torch.Size([2, 3, 256, 256])
    ret["warped_img_mask"] = warped_img_mask    # np(2, 256, 256, 3)

    ret["masked_img"] = masked_imgs

    return ret