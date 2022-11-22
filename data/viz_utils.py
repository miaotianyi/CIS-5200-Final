# import argparse
# import logging
# from pickletools import uint8
import sys
from pathlib import Path

import torch
import torchvision

import os
import numpy as np

import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

class VizUtils():
    def __init__(self, max_depth_clip, viz_max_depth_clip, pred_cmap, target_cmap, image_cmap, im_resize = (180, 320), real_dset=False, near_zero_tol=1e-2, circle_dict=None):
        self.max_depth_clip = max_depth_clip
        self.viz_max_depth_clip = viz_max_depth_clip
        self.tensor_to_float32 = torchvision.transforms.ConvertImageDtype(torch.float32)
        self.tensor_to_pil = torchvision.transforms.functional.to_pil_image
        self.pred_cmap = pred_cmap
        self.target_cmap = target_cmap
        self.image_cmap = image_cmap
        self.real_dset = real_dset
        self.near_zero_tol = near_zero_tol
        self.im_resize = im_resize

        if circle_dict is None:
            self.circle_dict = {'enable': False}
        else:
            self.circle_dict = circle_dict

    def tensor_to_depth_im(self, im_tensor, colormap, return_BGR = False):
            image_np = np.array(self.tensor_to_float32(im_tensor))
            if self.viz_max_depth_clip is not None:
                unnorm_image_np = image_np*self.max_depth_clip # back to meters
                image_np_clipped = np.clip(unnorm_image_np, 0, self.viz_max_depth_clip)
                image_np = ((255. / self.viz_max_depth_clip) * image_np_clipped)
            else: 
                image_np = (255. * image_np) 

            image_np_uint8 = (255.-image_np).astype(np.uint8) # invert the depth map so that colormap is also inverted to be more white than black
            image_np_color = cv2.applyColorMap(image_np_uint8, colormap)
            if return_BGR:
                return image_np_color
            else:
                return cv2.cvtColor(image_np_color, cv2.COLOR_BGR2RGB)

    def path_to_im(self, im_path):
        color_im = cv2.imread(im_path)
        color_im_resize = cv2.resize(color_im, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC)
        # DONT NEED GRAY2RGB CONVERSION AS IMREAD AUTOMATICALLY CONVERTS, UNLESS FLAG IS SET!
        # if is_gray: 
        #     color_im_resize = cv2.cvtColor(color_im_resize, cv2.COLOR_GRAY2BGR) # H x W x 3
        image_np_color = np.array(color_im_resize)
        return image_np_color

    def tensor_to_colormap_im(self, im_tensor, colormap):
        NotImplementedError
        # expects im_tensor of HxW dimension
        # returns np_array of HxWx3 RGB
        # tensor_to_pil = torchvision.transforms.functional.to_pil_image
        im_tensor_np = np.array(im_tensor)
        return cv2.cvtColor(cv2.applyColorMap(im_tensor_np, colormap), cv2.COLOR_BGR2RGB)

    def draw_heatmaps(self, im_tensor, pred_tensor=None, target_tensor=None, color_path=None, contact_pxls_flt=None):
        # accepts tensors of HxW dim which should be put on cpu
        heatmaps_dict = {}
        im_size = im_tensor.shape # H x W 

        if not pred_tensor is None:
            pred_loc_map_tensor = torch.sigmoid(pred_tensor) # index batch and channel
            pred_mask = np.array(pred_loc_map_tensor > self.near_zero_tol)

            pred_loc_map_np = (np.array(pred_loc_map_tensor)*255.).astype(np.uint8)
            pred_loc_map_np_color = cv2.applyColorMap(pred_loc_map_np, self.pred_cmap) # colormap expects 0-255 uint8
            
            pred_im_masked = pred_loc_map_np_color*pred_mask[..., None]
            pred_im = cv2.cvtColor(pred_im_masked.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['pred_im'] = pred_im

        if not self.real_dset:
            target_mask = np.array(target_tensor > self.near_zero_tol)

            target_loc_map_np = (np.array(target_tensor)*255.).astype(np.uint8)
            target_loc_map_np_color = cv2.applyColorMap(target_loc_map_np, self.target_cmap)
            # reverse the JET colormap so that highlighted pixel is blue!
            # target_loc_map_np_color = cv2.cvtColor(target_loc_map_np_color, cv2.COLOR_BGR2RGB)
            target_loc_map_masked = target_loc_map_np_color*target_mask[..., None]

            if self.circle_dict['enable']:
                circled_contacts_im = np.full((im_size[0], im_size[1], 3), fill_value=0).astype(np.uint8)
                for pxls in contact_pxls_flt:
                    if not torch.isnan(pxls).any():
                        pxls = pxls.int().tolist()
                        circled_contacts_im = cv2.circle(circled_contacts_im, tuple(pxls), radius=self.circle_dict['radius'], color=self.circle_dict['color'], thickness=self.circle_dict['thickness'])
                # circled_contacts_im = cv2.cvtColor(circled_contacts_im, cv2.COLOR_RGB2BGR) 

        if not color_path is None:
            color_im = cv2.imread(color_path)
            color_im_resize = cv2.resize(color_im, (im_size[1], im_size[0]), interpolation=cv2.INTER_CUBIC)
            # DONT NEED GRAY2RGB CONVERSION AS IMREAD AUTOMATICALLY CONVERTS, UNLESS FLAG IS SET!
            # if is_gray: 
            #     color_im_resize = cv2.cvtColor(color_im_resize, cv2.COLOR_GRAY2BGR) # H x W x 3
            image_np_color = np.array(color_im_resize)

        # else:
        image_np_depth = self.tensor_to_depth_im(im_tensor, self.image_cmap, return_BGR=True)

        # pred_overlay_im = (0.5*pred_loc_map_np_color*pred_mask[..., None]) + (0.5*image_np_color)
        # pred_overlay_im = np.clip(pred_overlay_im, a_min=0., a_max=255.)
        # pred_overlay_im = cv2.cvtColor(pred_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
        # heatmaps_dict['pred_color_overlay_im'] = pred_overlay_im

        # pred_overlay_im = (0.5*pred_loc_map_np_color*pred_mask[..., None]) + (0.5*image_np_depth)
        # pred_overlay_im = np.clip(pred_overlay_im, a_min=0., a_max=255.)
        # pred_overlay_im = cv2.cvtColor(pred_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
        # heatmaps_dict['pred_depth_overlay_im'] = pred_overlay_im

        # H x W x 3
        if not pred_tensor is None:

            target_pred_color_overlay_im = (0.33*pred_im_masked) + (0.8*target_loc_map_masked) + (0.33*image_np_color)
            target_pred_color_overlay_im = np.clip(target_pred_color_overlay_im, a_min=0., a_max=255.)
            target_pred_color_overlay_im = cv2.cvtColor(target_pred_color_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['target_pred_color_overlay_im'] = target_pred_color_overlay_im

            target_pred_depth_overlay_im = (0.33*pred_im_masked) + (0.8*target_loc_map_masked) + (0.33*image_np_depth)
            target_pred_depth_overlay_im = np.clip(target_pred_depth_overlay_im, a_min=0., a_max=255.)
            target_pred_depth_overlay_im = cv2.cvtColor(target_pred_depth_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['target_pred_depth_overlay_im'] = target_pred_depth_overlay_im

            pred_depth_overlay_im = (1.0*pred_im_masked) + (0.33*image_np_depth)
            pred_depth_overlay_im = np.clip(pred_depth_overlay_im, a_min=0., a_max=255.)
            pred_depth_overlay_im = cv2.cvtColor(pred_depth_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['pred_depth_overlay_im'] = pred_depth_overlay_im
        
            target_pred_im = ((0.8*target_loc_map_masked) + (0.4*pred_im_masked))
            target_pred_im = np.clip(target_pred_im, a_min=0., a_max=255.)
            target_pred_im = cv2.cvtColor(target_pred_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['target_pred_im'] = target_pred_im

        if self.circle_dict['enable']:
            target_color_overlay_im = (1.0*target_loc_map_masked) + (0.5*image_np_color) + (0.2*circled_contacts_im)
            target_color_overlay_im = np.clip(target_color_overlay_im, a_min=0., a_max=255.)
            target_color_overlay_im = cv2.cvtColor(target_color_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['circled_target_color_overlay_im'] = target_color_overlay_im

            target_depth_overlay_im = (1.0*target_loc_map_masked) + (0.5*image_np_depth) + (0.2*circled_contacts_im)
            target_depth_overlay_im = np.clip(target_depth_overlay_im, a_min=0., a_max=255.)
            target_depth_overlay_im = cv2.cvtColor(target_depth_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['circled_target_depth_overlay_im'] = target_depth_overlay_im

            if not pred_tensor is None:
                circled_target_pred_color_overlay_im = (0.8*pred_im_masked) + (0.2*target_loc_map_masked) + (0.33*image_np_color) + (0.2*circled_contacts_im)
                circled_target_pred_color_overlay_im = np.clip(circled_target_pred_color_overlay_im, a_min=0., a_max=255.)
                circled_target_pred_color_overlay_im = cv2.cvtColor(circled_target_pred_color_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
                heatmaps_dict['circled_target_pred_color_overlay_im'] = circled_target_pred_color_overlay_im

                circled_target_pred_depth_overlay_im = (0.8*pred_im_masked) + (0.2*target_loc_map_masked) + (0.33*image_np_depth) + (0.2*circled_contacts_im)
                circled_target_pred_depth_overlay_im = np.clip(circled_target_pred_depth_overlay_im, a_min=0., a_max=255.)
                circled_target_pred_depth_overlay_im = cv2.cvtColor(circled_target_pred_depth_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
                heatmaps_dict['circled_target_pred_depth_overlay_im'] = circled_target_pred_depth_overlay_im

                circled_target_pred_im = ((0.33*target_loc_map_masked) + (0.8*pred_im_masked)) + (0.25*circled_contacts_im)
                circled_target_pred_im = np.clip(circled_target_pred_im, a_min=0., a_max=255.)
                circled_target_pred_im = cv2.cvtColor(circled_target_pred_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
                heatmaps_dict['circled_target_pred_im'] = circled_target_pred_im
        else:
            target_color_overlay_im = (1.0*target_loc_map_masked) + (0.5*image_np_color)
            target_color_overlay_im = np.clip(target_color_overlay_im, a_min=0., a_max=255.)
            target_color_overlay_im = cv2.cvtColor(target_color_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['target_color_overlay_im'] = target_color_overlay_im

            target_depth_overlay_im = (1.0*target_loc_map_masked) + (0.33*image_np_depth)
            target_depth_overlay_im = np.clip(target_depth_overlay_im, a_min=0., a_max=255.)
            target_depth_overlay_im = cv2.cvtColor(target_depth_overlay_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
            heatmaps_dict['target_depth_overlay_im'] = target_depth_overlay_im

            # target_im = target_loc_map_np_color
            # heatmaps_dict['target_im'] = target_im
            
        return heatmaps_dict
    def plot_wrench_history(self, wrench_history, wrench_times, figsize_pxls): #in width x height 
        force_mags = np.linalg.norm(wrench_history[:, :3], ord=2, axis=1)
        torque_mags = np.linalg.norm(wrench_history[:, 3:], ord=2, axis=1)

        fig = Figure(figsize=(figsize_pxls[0]/100,figsize_pxls[1]/100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.plot(wrench_times, force_mags, 'k', wrench_times, force_mags, 'w|', ms=2, lw=2) # plot the line
        force_range = 10
        force_mag_mid = (np.min(force_mags) + np.max(force_mags))/2
        ax.set_ylim([force_mag_mid - (force_range/2),force_mag_mid + (force_range/2)])
        ax.grid(visible=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        canvas.draw()
        force_mag_plot = np.array(canvas.buffer_rgba())[..., :3]
        # force_mag_plot = Image.fromarray(rgba).convert('RGB')
        ax.cla()
        ax.plot(wrench_times, torque_mags, 'k', wrench_times, torque_mags, 'w|', ms=2, lw=2) # plot the line
        torque_range = 5
        torque_mag_mid = (np.min(torque_mags) + np.max(torque_mags))/2
        ax.set_ylim([torque_mag_mid - (torque_range/2), torque_mag_mid + (torque_range/2)])
        ax.grid(visible=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        canvas.draw()
        torque_mag_plot = np.array(canvas.buffer_rgba())[..., :3]
        # torque_mag_plot = Image.fromarray(rgba).convert('RGB')
        ax.cla()
        ax.plot(wrench_times, wrench_history[:, 0], 'r', wrench_times, wrench_history[:, 0], 'w|', ms=2, lw=2) # plot the line
        ax.plot(wrench_times, wrench_history[:, 1], 'g', wrench_times, wrench_history[:, 1], 'w|', ms=2, lw=2) # plot the line
        ax.plot(wrench_times, wrench_history[:, 2], 'b', wrench_times, wrench_history[:, 2], 'w|', ms=2, lw=2) # plot the line
        ax.grid(visible=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        canvas.draw()
        forces_plot = np.array(canvas.buffer_rgba())[..., :3]
        # forces_plot = Image.fromarray(rgba).convert('RGB')
        ax.cla()
        ax.plot(wrench_times, wrench_history[:, 3], 'r', wrench_times, wrench_history[:, 3], 'w|', ms=2, lw=2) # plot the line
        ax.plot(wrench_times, wrench_history[:, 4], 'g', wrench_times, wrench_history[:, 4], 'w|', ms=2, lw=2) # plot the line
        ax.plot(wrench_times, wrench_history[:, 5], 'b', wrench_times, wrench_history[:, 5], 'w|', ms=2, lw=2) # plot the line
        ax.grid(visible=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        canvas.draw()
        torques_plot = np.array(canvas.buffer_rgba())[..., :3]
        # torques_plot = Image.fromarray(rgba).convert('RGB')
        wrench_plots_dict = {
            'force_mag_plot': force_mag_plot,
            'torque_mag_plot': torque_mag_plot,
            'forces_plot': forces_plot,
            'torques_plot': torques_plot,
        }
        return wrench_plots_dict

    def viz_stn(self, model, im_tensor):
        with torch.no_grad():
            tfed_im_tensor = model.stn(im_tensor[None, None, ...]).cpu() # fake the batch and channel dims
        return self.tensor_to_depth_im(tfed_im_tensor[0, 0, ...], self.image_cmap, return_BGR=False) # remove the fake