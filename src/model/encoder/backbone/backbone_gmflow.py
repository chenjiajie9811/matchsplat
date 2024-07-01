import torch
import torch.nn as nn
from einops import rearrange

from .unimatch.backbone import CNNEncoder
from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import feature_add_position, split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine
from .unimatch.matching import (
    global_correlation_softmax, 
    local_correlation_softmax, 
    correlation, interpolate4d, 
    soft_argmax, 
    unnormalise_and_convert_mapping_to_flow)
from .unimatch.geometry import flow_warp

from ..costvolume.conversions import depth_to_relative_disparity
from ....geometry.epipolar_lines import get_depth

import torch.nn.functional as F


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class BackboneMultiviewGmCoPo(torch.nn.Module):
    """docstring for BackboneMultiview."""

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_self_attn=False,
        no_cross_attn=False,
        num_head=1,
        no_split_still_shift=False,
        no_ffn=False,
        global_attn_fast=True,
        downscale_factor=8,
        use_epipolar_trans=False,
    ):
        super(BackboneMultiviewGmCoPo, self).__init__()
        self.feature_channels = feature_channels
        # Table 3: w/o cross-view attention
        self.no_cross_attn = no_cross_attn
        # Table B: w/ Epipolar Transformer
        self.use_epipolar_trans = use_epipolar_trans

        self.downscale_factor = downscale_factor
        self.num_scales = 2

        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            # num_output_scales=1 if downscale_factor == 8 else 0,
            num_output_scales=2,
        )

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, downscale_factor ** 2 * 9, 1, 1, 0))


    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1] # [1, 1, 3, 1, 1]  B, N_Views 都变成为1

        mean = torch.tensor([0.485, 0.456, 0.406]).reshape( # torch.Size([1, 1, 3, 1, 1])
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        """
        # return a 2d list of extracted features
        # first dimension => views
        # second dimension => different scales
        # element => [B, C, h, w]
        """
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(v)]
        for feature in features:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            for idx in range(v):
                features_list[idx].append(feature[:, idx])

        return features_list

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.downscale_factor, self.downscale_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.downscale_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.downscale_factor * h,
                                      self.downscale_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
        epipolar_kwargs=None,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''

        B, N, C, H, W = images.shape

        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of features
        
        cur_features_list = [x[1] for x in features_list]
        if return_cnn_features:
            cnn_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        attn_splits_list = [2, 8]

        #TODO add coponerf attention layer for correlating corr, src, trg

        feat0_c, feat1_c = features_list[0][0], features_list[1][0] # B, C, h, w
        # add position to features
        feat0_c, feat1_c = feature_add_position(feat0_c, feat1_c, attn_splits_list[0], self.feature_channels)
        # transformer
        trans_feats_c = self.transformer([feat0_c, feat1_c], attn_num_splits=attn_splits_list[0])
        feat0_c, feat1_c = trans_feats_c[0], trans_feats_c[1]
        # correlation
        correlation_c = correlation(feat0_c, feat1_c)
        correlation_c = interpolate4d(correlation_c[:, None], (64, 64, 64, 64)).squeeze(1)

        feat0_f, feat1_f = features_list[0][1], features_list[1][1] # B, C, h, w
        # add position to features
        feat0_f, feat1_f = feature_add_position(feat0_f, feat1_f, attn_splits_list[1], self.feature_channels)
        # transformer
        trans_feats_f = self.transformer([feat0_f, feat1_f], attn_num_splits=attn_splits_list[1])
        feat0_f, feat1_f = trans_feats_f[0], trans_feats_f[1]
        # correlation
        correlation_f = correlation(feat0_f, feat1_f)

        c = (correlation_c + correlation_f) / 2. 
        c = c[:, None]

        #(b, 1, 64, 64, 64, 64) flatten-> (b, 4096, 64, 64) -> softargmax -> (b, 1, 64, 64)/(b, 1, 64, 64)
        grid_x_t_to_s, grid_y_t_to_s = soft_argmax(c.permute(0,1,4,5,2,3).flatten(1, 3)) 
        flow_t_to_s = torch.cat((grid_x_t_to_s, grid_y_t_to_s), dim=1) # (b, 2, 64, 64)
        flow = unnormalise_and_convert_mapping_to_flow(flow_t_to_s) # 2 -> 1 
  
        grid_x_s_to_t, grid_y_s_to_t = soft_argmax(c.flatten(1, 3))
        flow_s_to_t = torch.cat((grid_x_s_to_t, grid_y_s_to_t), dim=1)
        flow_flip = unnormalise_and_convert_mapping_to_flow(flow_s_to_t) # 1 -> 2 

        trans_features_coarse = torch.cat([feat0_c[:, None], feat1_c[:, None]], dim=1)
        trans_features_coarse = F.interpolate(trans_features_coarse.flatten(0, 1), size=(64, 64), mode='bilinear', align_corners=True)
        trans_features_coarse = trans_features_coarse.reshape(B, 2, 128, 64, 64)
        
        return {
            "trans_features" : torch.cat([feat0_f[:, None], feat1_f[:, None]], dim=1),
            "trans_features_coarse" : trans_features_coarse,
            "cnn_features" : cnn_features,
            "correlation" : c,
            "correlation_flip" : c.permute(0,1,4,5,2,3),
            "flow" : flow, # 2 -> 1
            "flow_flip" : flow_flip # 1 -> 2
        }






        
    
    def forward_(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
        epipolar_kwargs=None,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''

        B, N, C, H, W = images.shape
        
        # fixed for now (gmflow train with refinement)
        attn_splits_list = [2, 8]
        corr_radius_list = [-1, 4]
        prop_radius_list = [-1, 1]
        
        
        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of features
        
        # print ("backbone 0 Feature list shape:") 
        # for x in features_list:
        #     for y in x:
        #         print (y.shape)
        
        cur_features_list = [x[1] for x in features_list]

        # print ("backbone 0 Feature list shape:") 
        # for x in cur_features_list:
        #     # [B, 128, 64, 64]
        #     print (x.shape)

        if return_cnn_features:
            cnn_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]
            # print ("cnn features shape: ", cnn_features.shape)


        pred_bidir_flow = True
        
        flow = None
        correlations = []
        trans_features = None
        # resolution from low to high
        for scale_idx in range(self.num_scales):
            feat0, feat1 = features_list[0][scale_idx], features_list[1][scale_idx] # B, C, h, w
            
            if pred_bidir_flow:
                # 2*B, C, h, w
                feat0, feat1 = torch.cat([feat0, feat1], dim=0), torch.cat([feat1, feat0], dim=0)

            upsample_factor = self.downscale_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
            
            if flow is not None:
                flow = flow.detach()
                feat1 = flow_warp(feat1, flow)

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx] # [-1, 4]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feat0, feat1 = feature_add_position(feat0, feat1, attn_splits, self.feature_channels)

            # transformer
            trans_feats = self.transformer([feat0, feat1], attn_num_splits=attn_splits)
            feat0, feat1 = trans_feats[0], trans_feats[1]

            # trans feature from differernt scales are concatenated together for the next stage
            if scale_idx == 0:
                trans_features_coarse = torch.cat([feat0[:B, None], feat1[:B, None]], dim=1)
                trans_features_coarse = F.interpolate(trans_features_coarse.flatten(0, 1), size=(64, 64), mode='bilinear', align_corners=True)
                trans_features_coarse = trans_features_coarse.reshape(B, 2, 128, 64, 64)
            if scale_idx == self.num_scales - 1:
                trans_features = torch.cat([feat0[:B, None], feat1[:B, None]], dim=1) # B, V, C, H, W (B, 2, 128, 64, 64)

            # correlation and softmax
            if corr_radius == -1: # global matching
                flow_pred, _, costvolume = global_correlation_softmax(feat0, feat1, pred_bidir_flow=False, return_correlation=True)
                costvolume = interpolate4d(costvolume[:, None], (64, 64, 64, 64)).squeeze(1)
                # print (f"costvolume shape at scale {scale_idx}: {costvolume.shape}")
                correlations.append(costvolume) # (B, h, w, h, w)
            else: # local matching
                flow_pred = local_correlation_softmax(feat0, feat1, corr_radius)[0]
                # local matching at the last scale but we still need the global correlation
                correlations.append(correlation(feat0, feat1))
                # print (f"costvolumen shape at scale {scale_idx}: {correlations[-1].shape}")


            # 2 * B, C, h, w
            flow = flow + flow_pred if flow is not None else flow_pred

            # flow propagation with self-attn
            # TODO
            # if pred_bidir_flow and scale_idx == 0:
            #     feature0 = torch.cat((feat0, feat1), dim=0)  # [2*B, C, H, W] for propagation
            # flow = self.feature_flow_attn(feature0, flow.detach(),
            #                               local_window_attn=prop_radius > 0,
            #                               local_window_radius=prop_radius)

            # print (f"predicted flow at scale{scale_idx}: {flow.shape}")

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feat0, bilinear=False, upsample_factor=upsample_factor)
                # print (f"upsampled predicted flow at scale{scale_idx}: {flow_up.shape}")

                flow_out, flow_out_flip = torch.split(flow_up, B, dim=0)

            # flow_preds stores flow prediction from each scale and 
            # are already upsampled to the original resolution
            # we only care about the last 

        c = sum(correlations) / len(correlations) 
        c_out, c_out_flip = torch.split(c, B, dim=0)


        return {
            "trans_features" : trans_features,
            "trans_features_coarse" : trans_features_coarse,
            "cnn_features" : cnn_features,
            "correlation" : c_out,
            "correlation_flip" : c_out_flip,
            "flow" : flow_out, # 1 -> 2
            "flow_flip" : flow_out_flip # 2 -> 1
        }
        
        # add position to features
        # cur_features_list = feature_add_position_list(
        #     cur_features_list, attn_splits, self.feature_channels)

        # # Transformer
        # cur_features_list = self.transformer(
        #     cur_features_list, attn_num_splits=attn_splits)

        # features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        # if return_cnn_features:
        #     out_lists = [features, cnn_features] #
        # else:
        #     out_lists = [features, None]

        # return out_lists

