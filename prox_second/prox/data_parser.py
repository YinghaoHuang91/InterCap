# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from misc_utils import smpl_to_openpose
from projection_utils import Projection


Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)

def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 calib_dir='',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 depth_folder='Depth',
                 mask_folder='BodyIndex',
                 mask_color_folder='BodyIndexColor',
                 read_depth=False,
                 read_mask=False,
                 mask_on_color=False,
                 depth_scale=1e-3,
                 flip=False,
                 start=0,
                 step=1,
                 scale_factor=1,
                 frame_ids=None,
                 init_mode='sk',
                 batch_id=0,
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)
        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(keyp_folder)
        self.depth_folder = os.path.join(data_folder, depth_folder)
        self.mask_folder = os.path.join(data_folder, mask_folder)
        self.mask_color_folder = os.path.join(data_folder, mask_color_folder)

        self.data_folder = data_folder

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder.replace('Color', 'Depth'))
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') and
                          not img_fn.startswith('.')]
        self.img_paths = sorted(self.img_paths)

        self.img_paths = [p.replace('.png', '.jpg') for p in self.img_paths]



        if frame_ids is None:
            self.img_paths = self.img_paths[start::step]
        else:
            self.img_paths = [self.img_paths[id -1] for id in frame_ids]

        self.cnt = 0
        self.depth_scale = depth_scale
        self.flip = flip
        self.read_depth = read_depth
        self.read_mask = read_mask
        self.scale_factor = scale_factor
        self.init_mode = init_mode
        self.mask_on_color = mask_on_color


        self.projection = [Projection(calib_dir, 1), Projection(calib_dir, 2), Projection(calib_dir, 3), Projection(calib_dir, 4), Projection(calib_dir, 5), Projection(calib_dir, 6)]

        self.batch_id = batch_id

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        global load_one

        img = None
        img_2 = None
        img_3 = None
        img_4 = None
        img_5 = None
        img_6 = None


        pkl_path = os.path.join('/is/cluster/work/yhuang2/Azure/First_PC_Directly_4_Crop', img_path.split('/')[-3].replace('capture_', 'Capture_'), 'Res/res.pkl')

        with open(pkl_path, 'rb') as fin:
            import pickle as pkl; import trimesh
            pcs = pkl.load(fin)
            import open3d as o3d
            tmp = o3d.io.read_triangle_mesh( pcs[0]['mesh_path'] )


            pcs[0]['ob_v'] = np.asarray( tmp.vertices )
            pcs[0]['ob_f'] = np.asarray( tmp.triangles )
            pcs[0]['ob_normal'] = np.asarray( tmp.vertex_normals )

            # Load annotation points for each object
            contact_path = pcs[0]['mesh_path'].replace('.obj', '.pkl')#.replace('OBJs_2', 'OBJs_2_annotate')
            with open(contact_path, 'rb') as fin2:
                contact_points = pkl.load(fin2)

            pcs[0]['points'] = contact_points

            import sys; sys.path.append('../../obj_track')
            from utils import create_renderers

            pcs[0]['renders'] = create_renderers('Capture_06081_M1_Seg0')[0]
            func_renders_dyna = create_renderers('Capture_06081_M1_Seg0')[1]

            for p in pcs:
                p['renders_dyna'] = func_renders_dyna(p['bbs'])
                p['segs'] = p['segs'][:6]

                p['segs'] = [torch.from_numpy(s) if np.all(s != None) else s for s in p['segs']]


        total = len(self.img_paths)

        # Read all keypoints and depths
        keypoints_whole = []
        keypoints_2_whole = []
        keypoints_3_whole = []
        keypoints_4_whole = []
        keypoints_5_whole = []
        keypoints_6_whole = []

        img_path_whole = []

        mask_whole = []
        mask_2_whole = []
        mask_3_whole = []
        mask_4_whole = []
        mask_5_whole = []
        mask_6_whole = []

        depth_im_whole = []
        depth_im_2_whole = []
        depth_im_3_whole = []
        depth_im_4_whole = []
        depth_im_5_whole = []
        depth_im_6_whole = []

        scan_dict_whole = []
        scan_dict_2_whole = []
        scan_dict_3_whole = []
        scan_dict_4_whole = []
        scan_dict_5_whole = []
        scan_dict_6_whole = []

        def load_one(i):
            print('Load %d' % i)

            img_fn = '%05d' % i

            img_path_whole.append( img_path.replace('/00000', '/%05d' % i) ) 

            keypoint_fn = osp.join(self.keyp_folder, 'kp',
                                   img_fn + '_keypoints.json')

            keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                        use_face=self.use_face,
                                        use_face_contour=self.use_face_contour)
            if len(keyp_tuple.keypoints) < 1:
                return {}
            keypoints = np.stack(keyp_tuple.keypoints)

            bn = os.path.join(self.keyp_folder, 'kp')
            self.keyp_folder = bn


            keypoint_fn_2 = osp.join(self.keyp_folder.replace(bn, bn+'_2'),
                                   img_fn + '_keypoints.json')

            keyp_tuple_2 = read_keypoints(keypoint_fn_2, use_hands=self.use_hands,
                                        use_face=self.use_face,
                                        use_face_contour=self.use_face_contour)
            if len(keyp_tuple_2.keypoints) < 1:
                return {}
            keypoints_2 = np.stack(keyp_tuple_2.keypoints)


            # For 3-ed view
            keypoint_fn_3 = osp.join(self.keyp_folder.replace(bn, bn+'_3'),
                                   img_fn + '_keypoints.json')
            keyp_tuple_3 = read_keypoints(keypoint_fn_3, use_hands=self.use_hands,
                                        use_face=self.use_face,
                                        use_face_contour=self.use_face_contour)
            if len(keyp_tuple_3.keypoints) < 1:
                return {}
            keypoints_3 = np.stack(keyp_tuple_3.keypoints)

            # FOr 4-th view
            keypoint_fn_4 = osp.join(self.keyp_folder.replace(bn, bn+'_4'),
                                   img_fn + '_keypoints.json')
            keyp_tuple_4 = read_keypoints(keypoint_fn_4, use_hands=self.use_hands,
                                        use_face=self.use_face,
                                        use_face_contour=self.use_face_contour)
            if len(keyp_tuple_4.keypoints) < 1:
                return {}
            keypoints_4 = np.stack(keyp_tuple_4.keypoints)

            # For 5-th view
            keypoint_fn_5 = osp.join(self.keyp_folder.replace(bn, bn+'_6'),
                                   img_fn + '_keypoints.json')
            keyp_tuple_5 = read_keypoints(keypoint_fn_5, use_hands=self.use_hands,
                                        use_face=self.use_face,
                                        use_face_contour=self.use_face_contour)
            if len(keyp_tuple_5.keypoints) < 1:
                return {}
            keypoints_5 = np.stack(keyp_tuple_5.keypoints)

            # For 6-th view
            keypoint_fn_6 = osp.join(self.keyp_folder.replace(bn, bn+'_6'),
                                   img_fn + '_keypoints.json')
            keyp_tuple_6 = read_keypoints(keypoint_fn_6, use_hands=self.use_hands,
                                        use_face=self.use_face,
                                        use_face_contour=self.use_face_contour)
            if len(keyp_tuple_6.keypoints) < 1:
                return {}
            keypoints_6 = np.stack(keyp_tuple_6.keypoints)

            # 
            self.keyp_folder = self.keyp_folder[:-3]

            depth_im = None
            depth_im_2 = None
            depth_im_3 = None
            depth_im_4 = None
            depth_im_5 = None
            depth_im_6 = None


            if self.read_depth:
                #import ipdb; ipdb.set_trace()
                depth_im = cv2.imread(os.path.join(self.depth_folder, img_fn + '.png'), flags=-1).astype(float)
                depth_im_2 = cv2.imread(os.path.join(self.depth_folder.replace('Depth', 'Depth_2'), img_fn + '.png'), flags=-1).astype(float)
                depth_im_3 = cv2.imread(os.path.join(self.depth_folder.replace('Depth', 'Depth_3'), img_fn + '.png'), flags=-1).astype(float)
                depth_im_4 = cv2.imread(os.path.join(self.depth_folder.replace('Depth', 'Depth_4'), img_fn + '.png'), flags=-1).astype(float)
                depth_im_5 = cv2.imread(os.path.join(self.depth_folder.replace('Depth', 'Depth_6'), img_fn + '.png'), flags=-1).astype(float)
                depth_im_6 = cv2.imread(os.path.join(self.depth_folder.replace('Depth', 'Depth_6'), img_fn + '.png'), flags=-1).astype(float)


                depth_im = depth_im * self.depth_scale
                depth_im_2 = depth_im_2 * self.depth_scale
                depth_im_3 = depth_im_3 * self.depth_scale                
                depth_im_4 = depth_im_4 * self.depth_scale
                depth_im_5 = depth_im_5 * self.depth_scale
                depth_im_6 = depth_im_6 * self.depth_scale


                if self.flip:
                    depth_im = cv2.flip(depth_im, 1)
                    depth_im_2 = cv2.flip(depth_im_2, 1)
                    depth_im_3 = cv2.flip(depth_im_3, 1)
                    depth_im_4 = cv2.flip(depth_im_4, 1)
                    depth_im_5 = cv2.flip(depth_im_5, 1)
                    depth_im_6 = cv2.flip(depth_im_6, 1)

            mask = None
            mask_2 = None
            mask_3 = None
            mask_4 = None
            mask_5 = None
            mask_6 = None


            if self.read_mask:
                if self.mask_on_color:
                    mask = cv2.imread(os.path.join(self.mask_color_folder, img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
                    mask_2 = cv2.imread(os.path.join(self.mask_color_folder.replace('BodyIndexColor', 'BodyIndexColor_2'), img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
                    mask_3 = cv2.imread(os.path.join(self.mask_color_folder.replace('BodyIndexColor', 'BodyIndexColor_3'), img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
                    mask_4 = cv2.imread(os.path.join(self.mask_color_folder.replace('BodyIndexColor', 'BodyIndexColor_4'), img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
                    mask_5 = cv2.imread(os.path.join(self.mask_color_folder.replace('BodyIndexColor', 'BodyIndexColor_6'), img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
                    mask_6 = cv2.imread(os.path.join(self.mask_color_folder.replace('BodyIndexColor', 'BodyIndexColor_6'), img_fn + '.png'), cv2.IMREAD_GRAYSCALE)

                else:
                    mask = cv2.imread(os.path.join(self.mask_folder, img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
                if self.flip:
                    mask = cv2.flip(mask, 1)
                    mask_2 = cv2.flip(mask_2, 1)
                    mask_3 = cv2.flip(mask_3, 1)
                    mask_4 = cv2.flip(mask_4, 1)
                    mask_5 = cv2.flip(mask_5, 1)
                    mask_6 = cv2.flip(mask_6, 1)

            # YH: direclty mask on the depth image
            if self.read_depth:
                depth_im[mask == 255] = 0
                depth_im_2[mask_2 == 255] = 0
                depth_im_3[mask_3 == 255] = 0
                depth_im_4[mask_4 == 255] = 0
                depth_im_5[mask_5 == 255] = 0
                depth_im_6[mask_6 == 255] = 0


            scan_dict = None
            scan_dict_2 = None
            scan_dict_3 = None
            scan_dict_4 = None
            scan_dict_5 = None
            scan_dict_6 = None


            init_trans = None
            if depth_im is not None and mask is not None:
                scan_dict = self.projection[0].create_scan(mask, depth_im, mask_on_color=self.mask_on_color)
                scan_dict_2 = self.projection[1].create_scan(mask_2, depth_im_2, mask_on_color=self.mask_on_color)
                scan_dict_3 = self.projection[2].create_scan(mask_3, depth_im_3, mask_on_color=self.mask_on_color)
                scan_dict_4 = self.projection[3].create_scan(mask_4, depth_im_4, mask_on_color=self.mask_on_color)
                scan_dict_5 = self.projection[4].create_scan(mask_5, depth_im_5, mask_on_color=self.mask_on_color)
                scan_dict_6 = self.projection[5].create_scan(mask_6, depth_im_6, mask_on_color=self.mask_on_color)
            else:
                scan_dict = []
                scan_dict_2 = []
                scan_dict_3 = []
                scan_dict_4 = []
                scan_dict_5 = []
                scan_dict_6 = []

                init_trans = 0.

            print('Done %d' % i)        
            return {'idx':i, 'kp': keypoints, 'kp2': keypoints_2, 'kp3': keypoints_3, 'kp4': keypoints_4, 'kp5': keypoints_5, 'kp6': keypoints_6, 'di':depth_im, 'di2': depth_im_2, 'di3': depth_im_3, 'di4': depth_im_4, 'di5': depth_im_5, 'di6': depth_im_6, 'mask': mask, 'mask2': mask_2,  'mask3': mask_3, 'mask4': mask_4, 'mask5': mask_5, 'mask6': mask_6, 'sd': scan_dict, 'sd2':scan_dict_2, 'sd3':scan_dict_3, 'sd4':scan_dict_4, 'sd5':scan_dict_5, 'sd6':scan_dict_6}

        import multiprocessing
        pool = multiprocessing.Pool(64)
        res = pool.map(load_one, range(0, total))

        res = sorted(res, key=lambda x: x['idx'])


        for r in res:
            keypoints_whole.append( r['kp'] ) 
            keypoints_2_whole.append( r['kp2'] ) 
            keypoints_3_whole.append( r['kp3'] ) 
            keypoints_4_whole.append( r['kp4'] ) 
            keypoints_5_whole.append( r['kp5'] ) 
            keypoints_6_whole.append( r['kp6'] ) 

            depth_im_whole.append( r['di'] )
            depth_im_2_whole.append( r['di2'] )
            depth_im_3_whole.append( r['di3'] )
            depth_im_4_whole.append( r['di4'] )
            depth_im_5_whole.append( r['di5'] )
            depth_im_6_whole.append( r['di6'] )

            mask_whole.append( r['mask'] ) 
            mask_2_whole.append( r['mask2'] ) 
            mask_3_whole.append( r['mask3'] ) 
            mask_4_whole.append( r['mask4'] ) 
            mask_5_whole.append( r['mask5'] ) 
            mask_6_whole.append( r['mask6'] ) 

            scan_dict_whole.append( r['sd'] )
            scan_dict_2_whole.append( r['sd2'] )
            scan_dict_3_whole.append( r['sd3'] )
            scan_dict_4_whole.append( r['sd4'] )
            scan_dict_5_whole.append( r['sd5'] )
            scan_dict_6_whole.append( r['sd6'] )


        img_fn = '%05d' % 0
        init_trans = None

        output_dict = {'fn': img_fn,
                       'img_path': img_path_whole,
                       'keypoints': [keypoints_whole, keypoints_2_whole, keypoints_3_whole, keypoints_4_whole, keypoints_5_whole, keypoints_6_whole],
                       'img': [img, img_2, img_3, img_4, img_5, img_6],
                       'init_trans': init_trans,
                       'depth_im': [depth_im_whole, depth_im_2_whole, depth_im_3_whole, depth_im_4_whole, depth_im_5_whole, depth_im_6_whole],
                       'mask': [mask_whole, mask_2_whole, mask_3_whole, mask_4_whole, mask_5_whole, mask_6_whole],
                       'scan_dict':[scan_dict_whole, scan_dict_2_whole, scan_dict_3_whole, scan_dict_4_whole, scan_dict_5_whole, scan_dict_6_whole], #}
                       'pc': pcs}


        print('Data loading over')


        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1


        return self.read_item(img_path)
