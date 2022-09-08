# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman

import torch

import numpy as np
import open3d as o3d
import cv2
import json
import torch.nn as nn
import neural_renderer as nr


TYPE_DIC = {'book': ['book', 'remote', 'tv', 'laptop'], 'cup': ['cup', 'remote'], 'bottle': ['bottle', 'remote'], 'teddy': ['teddy bear', 'dog', 'remote'], 'sports': ['sports ball'], 'skate':['skateboard'], 'tennis': ['tennis racket', 'knife', 'remote'], 'chair':['chair'], 'bowl': ['bowl'], 'suitcase':['suitcase'], 'baseball': ['baseball bat', 'knife', 'tennis racket', 'frisbee', 'cell phone'], 'umbrella': ['umbrella'], 'vase': ['vase']}

class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist



#def create_renderers():
def create_renderers(CAPTURE_NAME):
    def get_idx(name):
        subject = name.split('_')[1]
        motion = name.split('_')[2][1:]
        motion = int(motion)

        if (subject in ['01', '02', '03'] and motion in [1, 2, 3, 4, 5, 6, 7]) or (subject in ['04'] and motion in [1, 2, 3, 4]):
            return 0

        if (subject in ['08', '09', '10']) or (subject in ['01', '02', '04', '05', '06', '07'] and motion in [8, 9, 10]):
            return 1

        if (subject in ['03'] and motion in [8, 9, 10]) or (subject in ['05', '06', '07'] and motion in [1, 2, 3, 4, 5, 6, 7]):
            return 2


    # First
    MATs = []
    MATs_ori = []
    tmp = np.eye(4)
    MATs.append(tmp)
    MATs_ori.append(tmp)

    tmp = np.array(
        [[0.578526, -0.198473, 0.791148, -1.98866 ],
        [0.16954, 0.97802, 0.121377, -0.254528 ],
        [-0.797849, 0.0639117, 0.59946, 1.13892 ],
        [0, 0, 0, 1 ],
    ])
    MATs.append(tmp)
    MATs_ori.append( np.linalg.inv(tmp) )

    tmp = np.array(
        [[-0.448919, -0.156267, 0.879802, -2.2178 ],
        [0.133952, 0.961696, 0.239161, -0.597409 ],
        [-0.883475, 0.225215, -0.410791, 3.5759 ],
        [0, 0, 0, 1 ]
    ])
    MATs.append(tmp)
    MATs_ori.append( np.linalg.inv(tmp) )

    tmp = np.array(
        [[-0.997175, -0.0640807, 0.0391773, 0.0546731 ],
        [-0.0501059, 0.956143, 0.288582, -0.806246 ],
        [-0.0559517, 0.285804, -0.956653, 5.28593 ],
        [0, 0, 0, 1 ],
    ])
    MATs.append(tmp)
    MATs_ori.append( np.linalg.inv(tmp) )

    tmp = np.array(
        [[-0.410844, 0.0626555, -0.90955, 2.65605 ],
        [-0.188877, 0.970143, 0.152145, -0.45593 ],
        [0.891926, 0.234302, -0.386743, 3.65228 ],
        [0, 0, 0, 1 ]
    ])
    MATs.append(tmp)
    MATs_ori.append( np.linalg.inv(tmp) )

    tmp = np.array(
        [[0.424204, 0.0773163, -0.90226, 2.32111 ],
        [-0.0993286, 0.994309, 0.0385042, -0.106149 ],
        [0.900103, 0.0732864, 0.42947, 1.05865 ],
        [0, 0, 0, 1 ]
    ])
    MATs.append(tmp)
    MATs_ori.append( np.linalg.inv(tmp) )


    cur_idx = get_idx(CAPTURE_NAME)

    # First
    if cur_idx == 0:
        pass
    elif cur_idx == 1:
        tmp = np.array(
            [[0.586004, -0.197023, 0.78599, -2.0087 ],
            [0.167134, 0.978521, 0.120675, -0.268243 ],
            [-0.792884, 0.0606496, 0.606348, 1.14359 ],
            [0, 0, 0, 1 ]
        ])
        MATs[1] = tmp
        MATs_ori[1] = np.linalg.inv(tmp)

        tmp = np.array(
            [[-0.433389, -0.155316, 0.887722, -2.21588],
            [0.137053, 0.962221, 0.235259, -0.597695 ],
            [-0.890724, 0.223623, -0.39573, 3.59858 ],
            [0, 0, 0, 1 ]
        ])
        MATs[2] = tmp
        MATs_ori[2] = np.linalg.inv(tmp)

    elif cur_idx == 2:
        tmp = np.array(
            [[-0.433389, -0.155316, 0.887722, -2.21588],
            [0.137053, 0.962221, 0.235259, -0.597695 ],
            [-0.890724, 0.223623, -0.39573, 3.59858 ],
            [0, 0, 0, 1 ]
        ])
        MATs[2] = tmp
        MATs_ori[2] = np.linalg.inv(tmp)


    FLs = np.array([[918.457763671875, 918.4373779296875], [915.29962158203125, 915.1966552734375], [912.8626708984375, 912.67633056640625], [909.82025146484375, 909.62469482421875], [920.533447265625, 920.09722900390625], [909.17633056640625, 909.23529052734375]])
    CENTERs = np.array([[956.9661865234375, 555.944580078125], [956.664306640625, 551.6165771484375], [956.72003173828125, 554.2166748046875], [957.6181640625, 554.60296630859375], [958.4615478515625, 550.42987060546875], [956.14801025390625, 555.01593017578125]])
    Ks = np.array([[0.535593, -2.509073, 0.000718, -0.000244, 1.362741, 0.414365, -2.340596, 1.297858], [0.486854, -2.639548, 0.000848, -0.000512, 1.499817, 0.363917, -2.457485, 1.424830], 
                    [0.457903, -2.524319, 0.000733, -0.000318, 1.464439, 0.340047, -2.355746, 1.395222], [0.396468, -2.488340, 0.000909, -0.000375, 1.456987, 0.278806, -2.316228, 1.385524],
                    [0.615471, -2.643317, 0.000616, -0.000661, 1.452086, 0.492699, -2.474038, 1.386289], [0.494798, -2.563026, 0.000720, -0.000212, 1.484987, 0.376524, -2.396207, 1.416732]])

    renderers = []
    total = len(MATs[:])

    def wrapper_dyna(bbs):
        rs_dyna = []
        img_sizes = []

        for j in range( len(bbs) ):
            bb = bbs[j]
            S = np.max( [bb[3]-bb[1], bb[2]-bb[0]] )

            K = np.array([[FLs[j][0], 0, CENTERs[j][0] - bb[0]], [0, FLs[j][1], CENTERs[j][1]-bb[1]], [0., 0., 1.]])
            #K = np.array([[FLs[j][0], 0, CENTERs[j][0] - 0], [0, FLs[j][1], CENTERs[j][1]-0], [0., 0., 1.]])

            K = K.reshape([1, 3, 3])
            R = MATs[j][:3, :3].T.reshape([1, 3, 3])
            t = np.matmul(R.squeeze(), -1 * MATs[j][:3, -1]).reshape([1, 3])

            r = nr.Renderer(camera_mode='projection', image_size=S, K = K, R=R, t=t, anti_aliasing=False, orig_size=S)
        
            rs_dyna.append(r)
            img_sizes.append(S)

        return rs_dyna


    for i in range(total):
        K = np.array([[FLs[i][0], 0, CENTERs[i][0]], [0, FLs[i][1], CENTERs[i][1]], [0., 0., 1.]])        
        K = K.reshape([1, 3, 3])
        R = MATs[i][:3, :3].T.reshape([1, 3, 3])
        t = np.matmul(R.squeeze(), -1 * MATs[i][:3, -1]).reshape([1, 3])

        r = nr.Renderer(camera_mode='projection', image_size=1920, K = K, R=R, t=t, anti_aliasing=False)

        renderers.append(r)

    def func_trans(v, cid, flag):
        v = np.require(v, np.float32)

        # Major to sub
        if flag == 0:
            mat = (MATs[cid])
            res = np.dot(mat[:3, :3], v.T).T + mat[:3, -1]
        elif flag == 1:
            mat = (MATs_ori[cid])
            res = np.dot(mat[:3, :3], v.T).T + mat[:3, -1]

        return res

    def func_unproject(img, depth, cid):
        import open3d as o3d
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, fx=FLs[cid][0], fy=FLs[cid][1], cx=CENTERs[cid][0], cy=CENTERs[cid][1])
        depth = depth.detach().cpu().numpy().astype(np.float32)
        depth = cv2.resize(depth, (1920, 1080))
        depth = o3d.geometry.Image( depth )

        img = np.asarray(img).astype(np.uint8)
        img = cv2.resize(img, (1920, 1080))
        img = np.concatenate([img[:, :, None], img[:, :, None], img[:, :, None]], axis=-1)
        img = o3d.geometry.Image(img)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth, depth_scale=1., convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=True)
        pc = np.array(pcd.points).astype(np.float32)
        
        return pc


    #return renderers, func_trans, func_unproject
    return renderers, wrapper_dyna

def load_targets(idx, CAPTURE_NAME, mesh_type):
    CAPTURE_NAME = CAPTURE_NAME.replace('_Seg', '/Seg_')
    CAPTURE_NAME = CAPTURE_NAME.replace('_M', '/')

    targets = []
    bbs = []

    #for i in range(0, 5):
    for i in range(0, 6):
        import pickle as pkl
        import os

        if False:
            # Remove 5th view for missing 5-th camera
            if i != 4:
            #if True:
                sil_path = '../Data/mrcnn_res/%s/Frames_Cam%d/color_pointrend_X_101_det/%05d/detections.json' % (CAPTURE_NAME, i+1, idx)
            else:
                sil_path = '../Data/mrcnn_res/%s/Frames_Cam%d/color_pointrend_X_101_det/%05d/detections.json' % (CAPTURE_NAME, 1, idx)
        else:
            sil_path = '../Data/mrcnn_res/%s/Frames_Cam%d/color_pointrend_X_101_det/%05d/detections.json' % (CAPTURE_NAME, i+1, idx)


        try:
            with open(sil_path) as fin:
                data = json.load(fin)
        except:
            try:
                sil_path = sil_path.replace('%05d' % idx, '%05d' % (idx + 1))
                with open(sil_path) as fin:
                    data = json.load(fin)
            except:
                sil_path = sil_path.replace('%05d' % idx, '%05d' % (idx + 2))
                with open(sil_path) as fin:
                    data = json.load(fin)



        masks = [np.zeros([1080, 1920]).astype(np.uint8)]
        bb = [np.zeros(4)]

        for d in data:
            if d['class'] in TYPE_DIC[mesh_type]:
                masks.append( np.asarray(d['mask']).astype(np.uint8) )
                bb.append( np.asarray(d['bbox']).astype(int) )


        masks = np.array(masks)
        targets.append( masks )

        bbs.append( bb )


    for i in range(0, 6):
        tmp = np.asarray(tmp, dtype=np.float32) / 1e3
        tmp = torch.from_numpy(tmp)#.to(device)
        targets.append( tmp )

    return targets, bbs


def get_vis(v, f):
    from psbody.mesh.visibility import visibility_compute
    from psbody.mesh import Mesh

    m = Mesh(v=v, f=f)

    import random
    TMP_PATH = '/tmp/tmp_%d.ply' % random.randint(0, 1e10)

    import os
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)
    m.write_ply(TMP_PATH)
    #m.load_from_ply('./tmp/tmp.ply')

    m = Mesh(filename=TMP_PATH)
    import os
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)

    (vis, n_dot) = visibility_compute(v=m.v, f=m.f, cams=np.array([[0.0, 0.0, 0.0]]))

    return vis
