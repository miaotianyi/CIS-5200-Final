import os
import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

import yaml

import skimage
import torchvision
import cv2

import glob
import natsort

import math

import pickle
class ContactDataset(Dataset):
    # if window_size = -1, produce the full episode !
    def __init__(self, bag_path, window_size=1, obj_name='EE_object', standardize_ft_pose=False, 
    calib_FT=False, im_time_offset=0.1, max_depth_clip=2.0, max_num_contact=20, 
    contact_persist_time=0.002, in_cam_frame=True, 
    im_resize=None, centered=False, proprio_history_dict=None, blur_contact_prob_dict=None):
        self.proprio_history_dict = proprio_history_dict
        self.standardize_ft_pose = standardize_ft_pose
       
        self.calib_FT = calib_FT
        self.max_depth_clip = max_depth_clip
        self.im_time_offset = im_time_offset
        self.max_num_contact = max_num_contact
        self.contact_persist_time = contact_persist_time

        if blur_contact_prob_dict is None:
            self.blur_contact_prob_dict = {'enable': False} 
        else:
            self.blur_contact_prob_dict = blur_contact_prob_dict 
        ## for now hardcode the main topic and topics of interest...
        
        # bag_path = '~/datasets/rosbags/input.bag'
        self.bag_path = os.path.expanduser(bag_path)
        if not os.path.exists(self.bag_path):
            raise AssertionError('bag path does not exist')
        
        assert os.path.exists(os.path.join(self.bag_path.strip('.bag'), 'tot_bag_dict.pickle')), 'tot_bag_dict.pickle does not exist' 
        with open(os.path.join(self.bag_path.strip('.bag'), 'tot_bag_dict.pickle'), 'rb') as handle:
            print('loaded info dict from pickle file!!')
            self.info_dict =  pickle.load(handle)

        for topic_dict in self.info_dict['topics']:
            # print(topic_dict)
            if 'contact_data' in topic_dict['topic']:
                self.contact_freq = topic_dict['frequency']
                self.contact_num_msgs = topic_dict['messages']
            elif 'franka_states' in topic_dict['topic']:
                self.robot_state_freq = topic_dict.get('frequency', -1.0) #default -1 when key not avail
                self.proprio_num_msgs = topic_dict['messages']
            elif 'depth' in topic_dict['topic']:
                self.image_freq = topic_dict['frequency']
                self.depth_num_msgs = topic_dict['messages']
            else:
                pass

        self.main_topic = '/panda/franka_state_controller_custom/franka_states'

        self.total_T = self.info_dict['duration']

        # if self.is_real_dataset:
        if os.path.exists(os.path.join(self.bag_path.strip('.bag'), 'aligned_depth_to_color')):
            self.im_type = 'aligned_depth_to_color'
        # else: 
        elif os.path.exists(os.path.join(bag_path.strip('.bag'), 'depth')):
            self.im_type = 'depth'
        else:
            raise AssertionError('no depth directory!')

        self.im_path = os.path.join(self.bag_path.strip('.bag'), self.im_type)
        assert os.path.exists(self.im_path), 'im_path does not exist!!'
        self.im_times = np.load(os.path.join(self.im_path, 'timestamps.npy'), allow_pickle=True) - self.im_time_offset
        self.im_path_list = natsort.natsorted(glob.glob(os.path.join(self.im_path, '*.png')))
        self.im_shape = cv2.imread(self.im_path_list[0]).shape[:2]
        self.in_cam_frame = in_cam_frame

        if self.im_type == 'depth':
            self.depth_tf_world = np.load(os.path.join(self.im_path, 'D_tf_W.npy'))
        elif self.im_type == 'aligned_depth_to_color':
            self.depth_tf_world = np.load(os.path.join(self.im_path, 'C_tf_W.npy'))
        
        self.K_cam = np.load(os.path.join(self.im_path, 'depth_K.npy'))
    
        assert self.depth_num_msgs == len(self.im_times), 'bag depth num msgs does not match depth timestamp length!'
        self.main_num_msgs = self.depth_num_msgs

        self.color_type = 'gray'
        self.color_path = os.path.join(self.bag_path.strip('.bag'), self.color_type)
        assert os.path.exists(self.color_path), 'color_path does not exist!!'
        self.color_times = np.load(os.path.join(self.color_path, 'timestamps.npy'), allow_pickle=True) - self.im_time_offset
        self.color_path_list = natsort.natsorted(glob.glob(os.path.join(self.color_path, '*.png')))
        self.color_shape = cv2.imread(self.color_path_list[0]).shape[:2]

        self.color_tf_world = np.load(os.path.join(self.color_path, 'C_tf_W.npy'))
        self.K_color = np.load(os.path.join(self.color_path, 'color_K.npy'))
    
        assert self.depth_num_msgs == len(self.im_times), 'bag depth num msgs does not match depth timestamp length!'
        self.main_num_msgs = self.depth_num_msgs

        ## WINDOW SIZE IS THE SAME FOR ALL STREAMS FOR NOW
        # max_window = max(list(time_window_dict.values()))
        self.window_size = window_size

        self.obj_name = obj_name

        assert self.main_num_msgs > 1, "must be at least one number of msgs!" 
        assert self.main_num_msgs >= self.window_size, "number of msgs must be geq window size!" 

        self.centered = centered 

        if self.window_size == -1:
            self._len = 1
        else:
            self._len = self.main_num_msgs - window_size + 1

        self.base_proprio_topic = '/panda/franka_state_controller_custom/franka_states/'

        self.pose_topics = []
        for i in range(16):
            topic = self.base_proprio_topic + 'O_T_EE/' + str(i)
            self.pose_topics.append(topic)
        
        # to add EE vel??
        self.EE_vel_topics = []
        for i in range(6):
            topic = self.base_proprio_topic + 'O_dP_EE/' + str(i)
            self.EE_vel_topics.append(topic)
        # O_dP_EE # EE vel computed as J*dq

        self.wrench_topics = []
        for i in range(6):
            topic = self.base_proprio_topic + 'O_F_ext_hat_K/' + str(i) #change from K_F_ext_hat_K to match other modality frames
            self.wrench_topics.append(topic)

        self.ep_info_base_topic = '/episode_info'
        
        if self.calib_FT:
            self.objCoM_pos_topic = self.ep_info_base_topic + '/EE_T_objCoM/position'
            self.objCoM_ori_topic = self.ep_info_base_topic + '/EE_T_objCoM/orientation'
            self.obj_mass_topic = self.ep_info_base_topic + '/EE_obj_mass'
          
        self.im_resize = im_resize #HxW
        self.im_to_tensor = torchvision.transforms.ToTensor() 
        self.tensor_to_float32 = torchvision.transforms.ConvertImageDtype(torch.float32)

        ## read the label
        self.contact_dt = 1./self.contact_freq

        self.contact_df = pd.read_pickle(os.path.join(bag_path.strip('.bag'), 'contact_df.pkl'))
    
        ## need to filter out all rows where none of the contact state collision names have the object name
        ## this is in order for the no contact estimation heuristic to work 
        collision_df = self.contact_df.loc[:, self.contact_df.columns.str.contains('collision')]
        self.contact_filtered_df = self.contact_df[collision_df.astype(str).sum(axis=1).str.contains(obj_name)]

        ## get features
        ## images
        self.depth_times = np.load(os.path.join(self.im_path, 'timestamps.npy'), allow_pickle=True)

        ## proprio
        self.proprio_df = pd.read_pickle(os.path.join(bag_path.strip('.bag'), 'proprio_df.pkl'))

        if self.calib_FT:
            self.ep_info_df = pd.read_pickle(os.path.join(bag_path.strip('.bag'), 'ep_info_df.pkl'))

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        # self.main_num_msgs
        if idx >= self._len:
            raise AssertionError('index out of range')

        ## indexing for non-overlapping indexing
        if self.window_size == -1:
            start_idx = 0
            end_idx = self.main_num_msgs
        else:
            start_idx = idx
            end_idx = idx + self.window_size
        
        # get images
        images = skimage.io.imread_collection(self.im_path_list[start_idx:end_idx]) # H x W x T
        #resize 
        if not self.im_resize: # if im resize is none
            images_np = np.array([img for img in images]) # now become T x H x W
        else:
            images_np = np.array([cv2.resize(img, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC) for img in images]) #preserves uint16 
        assert images_np.dtype == np.uint16, 'depth images not np uint16!'

        images_np = (images_np*1.0e-3).astype(np.float32)
        images_clipped = np.clip(images_np, 0, self.max_depth_clip)
        images_normalized = images_clipped / self.max_depth_clip
        # print(images_normalized.shape)

        # expects np array of dim N x H x W x C
        im_times = self.im_times[start_idx:end_idx]
        # im_max_vals = np.amax(images_np, axis=(-2, -1))

        # get color
        ## need to sort by nearest index here because color and depth timestamps arent aligned!
        nrst_color_idxs = self.get_nearest_idxs(im_times, self.color_times)
        color_im_paths = [self.color_path_list[idx] for idx in nrst_color_idxs]

        # get nearest extrinsics
        depth_tf_world = self.depth_tf_world

        nrst_proprio_idxs = self.get_nearest_idxs(im_times, self.proprio_df.index)
        nrst_tfs_np = np.array(self.proprio_df.iloc[nrst_proprio_idxs][self.pose_topics].values)
        nrst_poses_np = []
        pose_pxls = []
        for i in range(nrst_tfs_np.shape[0]):
            # print(tfs_np[i])
            pose = self.affine_tf_to_pose(nrst_tfs_np[i])
            pose_prj = self.point_proj(self.K_cam, depth_tf_world, pose[:3])
            if self.in_cam_frame:
                pose = self.transform_pose(pose, depth_tf_world)
            
            nrst_poses_np.append(pose)
            
            if not self.im_resize: # if im_resize is None
                pose_pxls.append(pose_prj)
            else:
                ## THIS ONLY WORKS BECAUSE THE RESIZE IS BY THE SAME FACTOR ON BOTH H AND W! AND ALSO THE RESIZE IS MADE TO BE A NICE INT FACTOR (4 in this case...)
                ## TODO fix this resize reprojection by scaling the projection matrix...
                pose_prj_resized = ((self.im_resize[0]/self.im_shape[0])*pose_prj).astype(int)
                pose_pxls.append(pose_prj_resized)

        nrst_poses_np = np.array(nrst_poses_np)
        pose_pxls = np.array(pose_pxls)

        ## get wrenches
        if self.proprio_history_dict is not None:
            nrst_wrench_np, _ = self.get_wrenches_history(im_times, self.proprio_history_dict, in_cam_frame=self.in_cam_frame)
        else:
            nrst_wrench_np = np.array(self.proprio_df.iloc[nrst_proprio_idxs][self.wrench_topics].values)
            if self.calib_FT:
                O_T_EE = np.reshape(nrst_tfs_np[0], (4,4), order='F') # indexing the time dimension here assuming T=1 
                EE_T_CoM, obj_mass = self.get_EE_T_CoM_and_mass(im_times)
                EEO_grav_wrench = self.get_EEO_grav_wrench(O_T_EE, EE_T_CoM, obj_mass)
                nrst_wrench_np = (nrst_wrench_np - EEO_grav_wrench).astype(float) #broadcasting grav_wrench from (6,) to (T,6) 

            if self.in_cam_frame:
                tfed_nrst_wrench_np = []
                for i in range(nrst_wrench_np.shape[0]):
                    wrench = nrst_wrench_np[i]
                    R = depth_tf_world[:3, :3]
                    wrench = np.concatenate((R@wrench[:3], R@wrench[3:]))
                    tfed_nrst_wrench_np.append(wrench)
                nrst_wrench_np = np.array(tfed_nrst_wrench_np)
        
        # get target contact label
        contact_time_label = self.get_contact_time_label(im_times, centered=self.centered)
        
        # no longer accepts a list of image times
        contact_dict = self.get_contact_data(contact_time_label, self.contact_dt, self.contact_filtered_df)
        
        # output the contact location prob map
        # output the contact forces map

        if not self.im_resize:
            contact_prob_map = np.zeros(self.im_shape)
            contact_force_map = np.zeros((3,) + self.im_shape) #3 x H x W
            contact_normal_map = np.zeros((3,) + self.im_shape)
        else: 
            contact_prob_map = np.zeros(self.im_resize)
            contact_force_map = np.zeros((3,) + self.im_resize) #fx,fy,fz
            contact_normal_map = np.zeros((3,) + self.im_resize) #fx,fy,fz
        
        contact_pxls = []

        if contact_dict['num_contacts'] != 0:
            for idx in range(contact_dict['num_contacts']): 
                contact_pos = contact_dict['positions'][idx]
                contact_force = contact_dict['wrenches'][idx][:3]
                contact_torque = contact_dict['wrenches'][idx][3:]
                contact_normal = contact_dict['normals'][idx]

                contact_pos_prj = self.point_proj(self.K_cam, depth_tf_world, contact_pos)

                if self.in_cam_frame:
                    R = depth_tf_world[:3, :3]
                    contact_pos = (depth_tf_world @ np.concatenate((contact_pos, np.array([1]))))[:-1]
                    contact_force = R@contact_force
                    contact_torque = R@contact_torque
                    contact_normal = R@contact_normal

                if not self.im_resize: # if im_resize is None
                    contact_pxls.append(contact_pos_prj)
                    contact_prob_map[contact_pos_prj[1], contact_pos_prj[0]] = 1.0
                    contact_force_map[:, contact_pos_prj[1], contact_pos_prj[0]] = contact_force
                    contact_normal_map[:, contact_pos_prj[1], contact_pos_prj[0]] = contact_normal
                else:
                    ## THIS ONLY WORKS BECAUSE THE RESIZE IS BY THE SAME FACTOR ON BOTH H AND W! AND ALSO THE RESIZE IS MADE TO BE A NICE INT FACTOR (4 in this case...)
                    ## TODO fix this resize reprojection by scaling the projection matrix...
                    contact_pos_prj_resized = ((self.im_resize[0]/self.im_shape[0])*contact_pos_prj).astype(int)
                    contact_pxls.append(contact_pos_prj_resized)
                    contact_prob_map[contact_pos_prj_resized[1], contact_pos_prj_resized[0]] = 1.0

                    contact_force_map[:, contact_pos_prj_resized[1], contact_pos_prj_resized[0]] = contact_force
                    contact_normal_map[:, contact_pos_prj_resized[1], contact_pos_prj_resized[0]] = contact_normal
        
            # now pad each array to fit into max num contacts

            # 2D np array of num contact x feature dim
            # for some really odd reason these arrays were being converted to type object... had to explicitly convert them to float dtype...
            num_pad_contacts = self.max_num_contact - contact_dict['num_contacts']
            assert num_pad_contacts >= 0, 'number of contacts is bigger than max padding!!!'

            padded_contact_positions = np.pad(contact_dict['positions'], ((0, num_pad_contacts), (0,0)), mode='constant', constant_values=(np.nan))
            padded_contact_wrenches = np.pad(contact_dict['wrenches'], ((0, num_pad_contacts), (0,0)), mode='constant', constant_values=(np.nan))
            padded_contact_normals = np.pad(contact_dict['normals'], ((0, num_pad_contacts), (0,0)), mode='constant', constant_values=(np.nan))
            # padded_contact_pxls = np.pad(np.array(contact_pxls, dtype=int), ((0, num_pad_contacts), (0,0)), mode='constant', constant_values=(np.nan))
            ## need to convert to float bc np.nan is a float and need to pad with it 
            padded_contact_pxls_flt = np.pad(np.array(contact_pxls, dtype=float), ((0, num_pad_contacts), (0,0)), mode='constant', constant_values=(np.nan))

            
        else:
            padded_contact_positions = np.full((self.max_num_contact, 3), np.nan)
            padded_contact_wrenches = np.full((self.max_num_contact, 6), np.nan)
            padded_contact_normals = np.full((self.max_num_contact, 3), np.nan)
            padded_contact_pxls_flt = np.full((self.max_num_contact, 2), np.nan)

        if self.blur_contact_prob_dict['enable']:
            contact_prob_map_blurred = cv2.GaussianBlur(contact_prob_map, (self.blur_contact_prob_dict['kernel_size'], self.blur_contact_prob_dict['kernel_size']), self.blur_contact_prob_dict['sigma'])


        return_dict = {
        'poses_np': nrst_poses_np, # B x T x 7
        'poses_pxls_np': pose_pxls, # B x T x 2
        'wrenches_np': nrst_wrench_np, # B x T x 6
        'images_tensor': images_normalized, # need to convert to dim T x H x W (im using Color channel as time...)
        'color_paths': color_im_paths, # T dim list of B dim tuples
        'im_times': im_times, # B x T
        'cam_tf_world': depth_tf_world, 
        'prob_map_np': contact_prob_map, # B x H x W
        'force_map_np': contact_force_map,
        'normal_map_np': contact_normal_map,
        'contact_positions': padded_contact_positions, # B x max_num_contact x 3
        'contact_wrenches': padded_contact_wrenches, # B x max_num_contact x 6
        'contact_normals': padded_contact_normals, # B x max_num_contact x 3
        'num_contacts': contact_dict['num_contacts'], # B
        'contact_pxls_flt': padded_contact_pxls_flt, # need to apply astype(int) after loading
        'contact_time': contact_dict['time'], # B
        'contact_time_diff': contact_dict['time_diff'],
        'len_samples': self._len,
        'idx_accessed': idx
        }
        if self.blur_contact_prob_dict['enable']:
            return_dict['prob_map_blurred_np'] = contact_prob_map_blurred

        # return poses_wrenches_actions_tensor, self.target, normalized_times_np,  self.total_T
        return return_dict

    def affine_tf_to_pose(self, tf): 
        ## returns 7d vector of trans, quat (x,y,z,w) format
        tf_np = np.reshape(tf, (4,4), order='F')
        # pose_np[0:4, 3]
        rot = tf_np[0:3, 0:3]
        # R @ R.T
        rot = R.from_matrix(rot)
        quat = rot.as_quat()
        quat = np.divide(quat, np.linalg.norm(quat))

        trans = tf_np[0:3, -1]
        pose = np.concatenate((trans, quat))
        return pose

    def invert_transform(self, tf):
        R = tf[0:3, 0:3]
        T = tf[:3, -1]
        tf_inv = np.diag([1.,1.,1.,1.])
        tf_inv[:3, :3] = R.T
        tf_inv[:3, -1] = -R.T @ T
        return tf_inv
    
    def transform_pose(self, pose_W, C_tf_W):
        C_rot_W = R.from_matrix(C_tf_W[:3, :3])

        pos = pose_W[:3]
        W_rot_EE = R.from_quat(pose_W[3:])
        pos_tfed = (C_tf_W @ np.concatenate((pos, np.array([1.,]))))[:-1]
        
        ori_tfed = C_rot_W * W_rot_EE

        return np.concatenate((pos_tfed, ori_tfed.as_quat())) 

    #TODO make this return a list of tfs!
    def get_inv_cam_extrin(self, times, depth=True): # make the assumption we are only dealing with one time...
        # TODO fix the assumption for the time window case
        # use only previous to make sure the extrinsics are not of the next trial where the cam pose has been newly sampled
        nrst_idx = self.get_nearest_idxs(times, self.ep_info_df.index, only_prev=True)[0] #indexing into time dim, assuming len(T)=1
        row = self.ep_info_df.iloc[nrst_idx]
        if depth:
            base_topic = '/episode_info/cam_depth_extrin'
        else:
            base_topic = '/episode_info/cam_color_extrin'

        cam_pos_topic = base_topic + '/position'
        cam_pos_cols = [col for col in row.keys() if cam_pos_topic in col]
        cam_pos = np.array(row[cam_pos_cols].values)

        cam_ori_topic = base_topic + '/orientation'
        cam_ori_cols = [col for col in row.keys() if cam_ori_topic in col]
        cam_ori = np.roll(np.array(row[cam_ori_cols].values), -1) # w,x,y,z to x,y,z,w

        # W_tf_cam = np.diag([0.,0.,0.,1.])
        # W_tf_cam[:3, :3] = (R.from_quat(cam_ori)).as_matrix()
        # W_tf_cam[:3, -1] = cam_pos

        cam_tf_W = np.diag([1.,1.,1.,1.])
        cam_tf_W[:3, :3] = (R.from_quat(cam_ori)).as_matrix().T
        cam_tf_W[:3, -1] = -cam_tf_W[:3, :3] @ cam_pos 
        
        return cam_tf_W
    
    def get_EE_T_CoM_and_mass(self, times):
        nrst_idx = self.get_nearest_idxs(times, self.ep_info_df.index, only_prev=True)[0] #indexing into time dim, assuming len(T)=1
        row = self.ep_info_df.iloc[nrst_idx]

        objCoM_pos_cols = [col for col in row.keys() if self.objCoM_pos_topic in col]
        objCoM_pos = np.array(row[objCoM_pos_cols].values)
        
        objCoM_ori_cols = [col for col in row.keys() if self.objCoM_ori_topic in col]
        objCoM_ori = np.roll(np.array(row[objCoM_ori_cols].values), -1) # w,x,y,z to x,y,z,w

        EE_T_CoM= np.eye(4)
        EE_T_CoM[:3, :3] = (R.from_quat(objCoM_ori)).as_matrix()
        EE_T_CoM[:3, -1] = objCoM_pos 

        obj_mass_cols = [col for col in row.keys() if self.obj_mass_topic in col]
        obj_mass = np.array(row[obj_mass_cols].values)

        return EE_T_CoM, obj_mass

    def get_EEO_grav_wrench(self, O_T_EE, EE_T_CoM, obj_mass):
        # EEO is the K/EE frame but rotated to align with the O frame
        CoM_pos_EE = -EE_T_CoM[:3, :3].T @ EE_T_CoM[:3, -1] 

        O_R_CoM = O_T_EE[:3, :3] @ EE_T_CoM[:3, :3]
        CoMO_pos_EEO = O_R_CoM @ CoM_pos_EE 

        CoMO_pos_EEO_skew = np.array([
            [0, -CoMO_pos_EEO[2], CoMO_pos_EEO[1]],
            [CoMO_pos_EEO[2], 0, -CoMO_pos_EEO[0]],
            [-CoMO_pos_EEO[1], CoMO_pos_EEO[0], 0],
        ])

        CoMO_adj_EEO = np.eye(6)
        # CoMO_adj_EEO[:3, :3] = EEO_R_CoMO 
        # CoMO_adj_EEO[3:, 3:] = EEO_R_CoMO
        CoMO_adj_EEO[3:, :3] = - CoMO_pos_EEO_skew # because the rotation matrix here is identity just need the negative of the skew sym matrix

        # this is the accel applied by gravity at the CoM frame but oriented with world/origin
        CoMO_grav_wrench  =  np.array([0, 0, -9.81, 0, 0, 0]) * obj_mass
        EEO_grav_wrench = CoMO_adj_EEO @ CoMO_grav_wrench

        return EEO_grav_wrench

    # TODO return times list
    def get_nearest_idxs(self, times, df_index, only_prev=False): #df_index can equivalently be an np array
        # from https://stackoverflow.com/a/26026189
        ## if both elements exactly match, left will give index at the matching value itself and if right, will give index ahead of match
        ## search sorted is like if at that returned idx location, you split the list (inclusive of the given idx) to the right and inserted in the query value in between the split
        ## here times is the query value and we want to insert somwhere into the df_index timestamp list
        ## so left will always return the idx of the time in df_index that is "forward/ahead" of the queried time 
        idxs = df_index.searchsorted(times, side="left") 
        idxs_list = []
        for i in range(len(times)):
            if idxs[i] > 0 and only_prev: # always return the prev index given index is not first 
                idxs_list.append(idxs[i]-1)
            elif idxs[i] > 0 and (idxs[i] == len(df_index) or math.fabs(times[i] - df_index[idxs[i]-1]) < math.fabs(times[i] - df_index[idxs[i]])): # FIXED BUG HERE WHERE I COMPARED TO SAME IDX - 1 ON THE RIGHT SIDE OF INEQ
                idxs_list.append(idxs[i]-1)
            else:
                idxs_list.append(idxs[i])
        return idxs_list
    
    def get_contact_time_label(self, im_times, centered=False):
        if not centered:
            return im_times[-1]
        else:
            return im_times[len(im_times)//2]

    ## TODO incorporate this into the below function since its essentially duplicate code...
    def get_contact_data_from_row(self, row): # for clump contact logic
        # These are all 2d num_contacts x number of feature dims
        contact_dict = {}
        contact_positions = []
        contact_wrenches = []
        contact_normals = []

        total_num_contacts = 0
        contact_time = row.name
        contact_dict['time'] = contact_time

        ## again need to filter per timestamp, only the contact states that have object name in collisions
        collision_row = row[row.keys().str.contains('collision')]
        ## filter out to only collisions which contain object name
        ## need to handle when some collision fields are None by setting na flag!
        filtered_collision_row = collision_row[collision_row.str.contains(self.obj_name, na=False)]
        ## get the state index after the state substring
        filtered_state_idxs = filtered_collision_row.keys().str.split('/').str.get(3)
        for state_idx_str in filtered_state_idxs:
            state_num_contacts = len(row[row.keys().str.contains(state_idx_str + '/depths', na=False)])
            total_num_contacts += state_num_contacts

            base_contact_name = '/contact_data/states/' + state_idx_str
            for contact_idx in range(state_num_contacts):

                contact_pos_idx = base_contact_name + '/contact_positions/' + str(contact_idx) 
                contact_pos_cols = [col for col in row.keys() if contact_pos_idx in col]
                contact_pos = row[contact_pos_cols].values.astype(np.float64)
                contact_positions.append(contact_pos)

                contact_nrml_idx = base_contact_name + '/contact_normals/' + str(contact_idx) 
                contact_nrml_cols = [col for col in row.keys() if contact_nrml_idx in col]
                contact_nrml = row[contact_nrml_cols].values.astype(np.float64)
                contact_normals.append(contact_nrml)

                contact_force_idx = base_contact_name + '/wrenches/' + str(contact_idx) + '/force/'
                contact_force_cols = [col for col in row.keys() if contact_force_idx in col]
                contact_force = row[contact_force_cols].values.astype(np.float64)


                contact_torque_idx = base_contact_name + '/wrenches/' + str(contact_idx) + '/torque/'
                contact_torque_cols = [col for col in row.keys() if contact_torque_idx in col]
                contact_torque = row[contact_torque_cols].values.astype(np.float64)

                contact_wrenches.append(np.concatenate((contact_force, contact_torque)))
        
        contact_dict['positions'] = np.array(contact_positions, dtype=float)
        contact_dict['wrenches'] = np.array(contact_wrenches, dtype=float)
        contact_dict['normals'] = np.array(contact_normals, dtype=float)

        contact_dict['num_contacts'] = total_num_contacts
        return contact_dict

    def get_poses_history(self, times_list, proprio_history_dict, in_cam_frame=False, return_pxls=False):
        depth_tf_world = self.depth_tf_world

        num_samples = int(proprio_history_dict['time_window'] * proprio_history_dict['sample_freq'])
        proprio_hist_times = np.linspace(min(times_list) - self.proprio_history_dict['time_window'], min(times_list), num_samples, endpoint=True).tolist()
        nrst_proprio_hist_idxs = self.get_nearest_idxs(proprio_hist_times, self.proprio_df.index)
        nrst_tfs_np = np.array(self.proprio_df.iloc[nrst_proprio_hist_idxs][self.pose_topics].values)
        nrst_poses_np = []
        pose_pxls = []
        for i in range(nrst_tfs_np.shape[0]):
            pose = self.affine_tf_to_pose(nrst_tfs_np[i])
            if in_cam_frame:
                pose = self.transform_pose(pose, depth_tf_world)
            nrst_poses_np.append(pose)
            if return_pxls:            
                pose_prj = self.point_proj(self.K_cam, depth_tf_world, pose[:3])
                if not self.im_resize: # if im_resize is None
                    pose_pxls.append(pose_prj)
                else:
                    ## THIS ONLY WORKS BECAUSE THE RESIZE IS BY THE SAME FACTOR ON BOTH H AND W! AND ALSO THE RESIZE IS MADE TO BE A NICE INT FACTOR (4 in this case...)
                    ## TODO fix this resize reprojection by scaling the projection matrix...
                    pose_prj_resized = ((self.im_resize[0]/self.im_shape[0])*pose_prj).astype(int)
                    pose_pxls.append(pose_prj_resized)

        nrst_poses_np = np.array(nrst_poses_np)
        if return_pxls:
            pose_pxls = np.array(pose_pxls) 
            return nrst_poses_np, pose_pxls
        else:
            return nrst_poses_np

    def get_wrenches_history(self, times_list, proprio_history_dict, in_cam_frame=False, calib_grav=False):
        depth_tf_world = self.depth_tf_world

        num_samples = int(proprio_history_dict['time_window'] * proprio_history_dict['sample_freq'])
        proprio_hist_times = np.linspace(min(times_list) - proprio_history_dict['time_window'], min(times_list), num_samples, endpoint=True).tolist()
        nrst_proprio_hist_idxs = self.get_nearest_idxs(proprio_hist_times, self.proprio_df.index)
        nrst_proprio_hist_times = self.proprio_df.index[nrst_proprio_hist_idxs]
        nrst_wrench_np = np.array(self.proprio_df.iloc[nrst_proprio_hist_idxs][self.wrench_topics].values)
        
        if calib_grav:
            # O_T_EE = np.reshape(nrst_tfs_np[0], (4,4), order='F') # indexing the time dimension here assuming T=1 
            # EE_T_CoM, obj_mass = self.get_EE_T_CoM_and_mass(times_list)
            # EEO_grav_wrench = self.get_EEO_grav_wrench(O_T_EE, EE_T_CoM, obj_mass)
            # nrst_wrench_np = (nrst_wrench_np - EEO_grav_wrench).astype(float) #broadcasting grav_wrench from (6,) to (T,6) 
            NotImplementedError

        if in_cam_frame:    
            tfed_nrst_wrench_np = []
            for i in range(nrst_wrench_np.shape[0]):
                # print(tfs_np[i])
                wrench = nrst_wrench_np[i]
                R = depth_tf_world[:3, :3]
                wrench = np.concatenate((R@wrench[:3], R@wrench[3:]))
                tfed_nrst_wrench_np.append(wrench)
            nrst_wrench_np = np.array(tfed_nrst_wrench_np)

        return nrst_wrench_np, nrst_proprio_hist_times

    # https://stackoverflow.com/a/61704344
    def lambda_max(self, arr, axis=None, key=None, keepdims=False):
        if callable(key):
            idxs = np.argmax(key(arr), axis)
            if axis is not None:
                idxs = np.expand_dims(idxs, axis)
                result = np.take_along_axis(arr, idxs, axis)
                if not keepdims:
                    result = np.squeeze(result, axis=axis)
                return result
            else:
                return arr.flatten()[idxs]
        else:
            return np.amax(arr, axis)
    
    def get_contact_data(self, im_time, contact_dt, contact_df, clump=False):
        # These are all 2d num_contacts x number of feature dims
        contact_dict = {}
        contact_positions = []
        contact_wrenches = []
        contact_normals = []

        total_num_contacts = 0

        nrst_contact_idx = self.get_nearest_idxs([im_time], contact_df.index)[0]
        contact_dict['contact_idx'] = nrst_contact_idx

        row = contact_df.iloc[nrst_contact_idx].loc[contact_df.iloc[nrst_contact_idx].notnull()]
        contact_time = row.name
        contact_dict['time'] = contact_time

        ## again need to filter per timestamp, only the contact states that have object name in collisions
        collision_row = row[row.keys().str.contains('collision')]
        ## filter out to only collisions which contain object name
        ## need to handle when some collision fields are None by setting na flag!
        filtered_collision_row = collision_row[collision_row.str.contains(self.obj_name, na=False)]
        ## get the state index after the state substring
        filtered_state_idxs = filtered_collision_row.keys().str.split('/').str.get(3)


        contact_time_diff = abs(contact_time - im_time)
        contact_dict['time_diff'] = contact_time_diff

        ## if there is no contact ...
        ## inflating by dt factor greater than 1 allows checking multiple local timestamps for contact
        ## in the case where the contact dissapears intermittently!
        # if contact_time_diff > (contact_dt*1.5): #1.5
        if (contact_time_diff > (self.contact_persist_time)): #1.5
            contact_dict['positions'] = np.nan
            contact_dict['wrenches'] = np.nan
            contact_dict['normals'] = np.nan
        ## if there is contact temporally locally
        else:
            for state_idx_str in filtered_state_idxs:
                state_num_contacts = len(row[row.keys().str.contains(state_idx_str + '/depths', na=False)])
                total_num_contacts += state_num_contacts

                base_contact_name = '/contact_data/states/' + state_idx_str
                for contact_idx in range(state_num_contacts):

                    contact_pos_idx = base_contact_name + '/contact_positions/' + str(contact_idx) 
                    contact_pos_cols = [col for col in row.keys() if contact_pos_idx in col]
                    contact_pos = row[contact_pos_cols].values.astype(np.float64)
                    contact_positions.append(contact_pos)

                    contact_nrml_idx = base_contact_name + '/contact_normals/' + str(contact_idx) 
                    contact_nrml_cols = [col for col in row.keys() if contact_nrml_idx in col]
                    contact_nrml = row[contact_nrml_cols].values.astype(np.float64)
                    contact_normals.append(contact_nrml)

                    contact_force_idx = base_contact_name + '/wrenches/' + str(contact_idx) + '/force/'
                    contact_force_cols = [col for col in row.keys() if contact_force_idx in col]
                    contact_force = row[contact_force_cols].values.astype(np.float64)


                    contact_torque_idx = base_contact_name + '/wrenches/' + str(contact_idx) + '/torque/'
                    contact_torque_cols = [col for col in row.keys() if contact_torque_idx in col]
                    contact_torque = row[contact_torque_cols].values.astype(np.float64)

                    contact_wrenches.append(np.concatenate((contact_force, contact_torque)))
            
            contact_dict['positions'] = np.array(contact_positions, dtype=float)
            contact_dict['wrenches'] = np.array(contact_wrenches, dtype=float)
            contact_dict['normals'] = np.array(contact_normals, dtype=float)

        contact_dict['num_contacts'] = total_num_contacts
        return contact_dict

    def point_proj(self, K, C_tf_W, pos):
        contact_pos_in_depth = (C_tf_W @ np.append(pos, 1))[:-1]
        # print(contact_pos_in_depth)
        project_coords = K @ (contact_pos_in_depth)
        return (project_coords[:2]/project_coords[-1]).astype(int)  

    #### VISUALIZATION UTILS
    # pink color by default
    def viz_contact_pos(self, image_np, contact_pxls_flt, num_contacts, radius=1, color=(255, 133, 233)): # expects contact positions in world frame
        contact_pxls = contact_pxls_flt[:num_contacts, ...].astype(int)
        for idxs in contact_pxls:
            image_np = cv2.circle(image_np, tuple(idxs), radius=radius, color=color, thickness=-1)
        return image_np
    
    # def viz_EE_pos(self, image_np, EE_pxls, radius=1, color=(255, 133, 233))
    #     image_np = cv2.circle(image_np, tuple(idxs), radius=radius, color=color, thickness=-1)
    #     return image_np

    def tensor_to_depth_im(self, im_tensor, colormap, is_np = False, return_BGR=False):
        if not np:
            image_np = np.array(self.tensor_to_float32(im_tensor))
        else: 
            image_np = im_tensor 
        image_np_uint8 = (255 * image_np).astype(np.uint8)
        image_np_color = cv2.applyColorMap(image_np_uint8, colormap)
        if return_BGR:
            return image_np_color
        else:
            return cv2.cvtColor(image_np_color, cv2.COLOR_BGR2RGB)

    def color_path_to_im(self, color_path):
        color_im = cv2.imread(color_path)
        color_im_resize = cv2.resize(color_im, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC)
        return np.array(color_im_resize)
    # def get_FT_history(self, idx, window_size):
    #     im_time = self.im_times[idx]






