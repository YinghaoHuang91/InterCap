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


#import os
#os.environ['PYOPENGL_PLATFORM'] = 'egl'

import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img
import json
from optimizers import optim_factory

import fitting
#import fitting_try_paralel_2 as fitting

from human_body_prior.tools.model_loader import load_vposer
from psbody.mesh import Mesh
import scipy.sparse as sparse


def fit_single_frame(img,
                     keypoints,
                     init_trans,
                     scan,
                     pc,
                     scene_name,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     body_scene_rendering_fn='body_scene.png',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length_x=5000.,
                     focal_length_y=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     ####################
                     ### PROX
                     render_results=False,
                     camera_mode='moving',
                     ## Depth
                     s2m=False,
                     s2m_weights=None,
                     m2s=False,
                     m2s_weights=None,
                     rho_s2m=1,
                     rho_m2s=1,
                     init_mode=None,
                     trans_opt_stages=None,
                     viz_mode='mv',
                     #penetration
                     sdf_penetration=False,
                     sdf_penetration_weights=0.0,
                     sdf_dir=None,
                     cam2world_dir=None,
                     #contact
                     contact=False,
                     rho_contact=1.0,
                     contact_loss_weights=None,
                     contact_angle=15,
                     contact_body_parts=None,
                     body_segments_dir=None,
                     load_scene=False,
                     scene_dir=None,
                     **kwargs):

    body_model.transl.requires_grad = True

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if visualize:
        pil_img.fromarray((img * 255).astype(np.uint8)).show()

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg


    keypoint_data = torch.tensor(keypoints[:6], dtype=dtype)
    gt_joints = keypoint_data[:, :, :, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, :, :, 2]

    betas_per_frame = []
    transl_per_frame = []
    left_hand_pose_per_frame = []
    reft_hand_pose_per_frame = []
    leye_pose_per_frame = []
    reye_pose_per_frame = []
    expression_per_frame = []
    pose_embedding_per_frame = []
    bodypose_per_frame = []
    global_orient_per_frame = []
    jaw_pose_per_frame = []


    for i in range(0, keypoint_data.shape[1]):
        batch_id = kwargs['batch_id']

        i2 = i # + 


        print('Per-frame param %d' % i2)
        import pickle as pkl

        try:
            # Per-frame
            with open(result_fn.replace('results/00000/000.pkl', 'results/%05d/000.pkl' % i2), 'rb') as fin:
                data_res = pkl.load(fin)            
            if np.isnan( np.min(data_res['pose_embedding']) ) or np.isnan( np.min(data_res['right_hand_pose']) ) or np.isnan( np.min(data_res['left_hand_pose']) ) or np.isnan( np.min(data_res['expression']) ):
                with open(result_fn.replace('results/00000/000.pkl', 'results/%05d/000.pkl' % 0), 'rb') as fin:
                    data_res = pkl.load(fin)            
        except:
            with open(result_fn.replace('results/00000/000.pkl', 'results/%05d/000.pkl' % 0), 'rb') as fin:
                data_res = pkl.load(fin)            

        betas_per_frame.append( data_res['betas'] )
        transl_per_frame.append( data_res['transl'] ) 

        left_hand_pose_per_frame.append( data_res['left_hand_pose'].reshape([-1, 12]) )
        reft_hand_pose_per_frame.append( data_res['right_hand_pose'].reshape([-1, 12]) )

        leye_pose_per_frame.append( data_res['leye_pose'] )
        reye_pose_per_frame.append( data_res['reye_pose'] )
        expression_per_frame.append( data_res['expression'] )
        jaw_pose_per_frame.append( data_res['jaw_pose'] )

        pose_embedding_per_frame.append( data_res['pose_embedding'] )
        bodypose_per_frame.append( data_res['body_pose'].reshape([-1, 63]) )

        global_orient_per_frame.append( data_res['global_orient'].reshape([-1, 3]) )

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding_per_frame = np.array( pose_embedding_per_frame )  
        pose_embedding = torch.from_numpy(pose_embedding_per_frame).to(device=device).reshape(1, -1)        
        pose_embedding = pose_embedding.type(dtype)
        pose_embedding.requires_grad = True

        bodypose = np.array( bodypose_per_frame ).reshape([-1, 63])

        transl_per_frame = np.array( transl_per_frame )
        transl_per_frame = torch.tensor(transl_per_frame, dtype=dtype, device=device, requires_grad=True).reshape(1, -1)

        left_hand_pose_per_frame = np.array( left_hand_pose_per_frame ) 
        left_hand_pose_per_frame = torch.tensor(left_hand_pose_per_frame, dtype=dtype, device=device, requires_grad=True).reshape(1, -1)

        reft_hand_pose_per_frame = np.array( reft_hand_pose_per_frame ) 
        reft_hand_pose_per_frame = torch.tensor(reft_hand_pose_per_frame, dtype=dtype, device=device, requires_grad=True).reshape(1, -1)

        leye_pose_per_frame = np.array( leye_pose_per_frame )
        leye_pose_per_frame = torch.tensor(leye_pose_per_frame, dtype=dtype, device=device, requires_grad=True).reshape(1, -1)

        reye_pose_per_frame = np.array( reye_pose_per_frame )
        reye_pose_per_frame = torch.tensor(reye_pose_per_frame, dtype=dtype, device=device, requires_grad=True).reshape(1, -1)

        expression_per_frame = np.array( expression_per_frame )
        expression_per_frame = torch.tensor(expression_per_frame, dtype=dtype, device=device, requires_grad=True).reshape(1, -1)

        global_orient_per_frame = np.array(global_orient_per_frame)
        global_orient_per_frame = torch.tensor(global_orient_per_frame, dtype=dtype, device=device, requires_grad=True).reshape(1, -1)

        jaw_pose_per_frame = np.array( jaw_pose_per_frame )
        jaw_pose_per_frame = torch.tensor(jaw_pose_per_frame, dtype=dtype, device=device, requires_grad=True).reshape(1, -1)

        betas_per_frame = np.array(betas_per_frame).squeeze()


        betas_per_frame = torch.tensor(betas_per_frame, dtype=dtype, device=device, requires_grad=True)


        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = torch.zeros([1, vposer_latent_dim*batch_size],
                                     dtype=dtype)
    else:
        pass
    

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    scan_tensors = scan

    # load pre-computed signed distance field
    sdf = None
    sdf_normals = None
    grid_min = None
    grid_max = None
    voxel_size = None
    if sdf_penetration:
        with open(osp.join(sdf_dir, scene_name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)
            grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
            grid_dim = sdf_data['dim']
        voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(osp.join(sdf_dir, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        sdf = torch.tensor(sdf, dtype=dtype, device=device)
        sdf_normals = np.load(osp.join(sdf_dir, scene_name + '_normals.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)
        sdf_normals = torch.tensor(sdf_normals, dtype=dtype, device=device)


    import os
    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        cam2world = np.array(json.load(f))
        R = torch.tensor(cam2world[:3, :3].reshape(3, 3), dtype=dtype, device=device)
        t = torch.tensor(cam2world[:3, 3].reshape(1, 3), dtype=dtype, device=device)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)


    # load vertix ids of contact parts
    contact_verts_ids  = ftov = None

    #if contact:
    contact_verts_ids = []
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
    contact_verts_ids = np.concatenate(contact_verts_ids)
    
    # Here, do the conversion
    # With extra samples from head
    with open('../models/wo_head.pkl', 'rb') as fin:
            import pickle as pkl
            wo_head = pkl.load(fin)
            wo_head = wo_head['wo_head']

    mapping = {}
    for i, j in enumerate(wo_head):
        mapping[j] = i
    for i in range( len(contact_verts_ids) ):            
        contact_verts_ids[i] = mapping[ contact_verts_ids[i] ]


    import open3d as o3d
    template_wohead = o3d.io.read_triangle_mesh('../models/000_template.ply')
    vertices_np = np.asarray( template_wohead.vertices )[wo_head]
    body_faces_np = np.asarray( template_wohead.triangles )

    if not os.path.exists('./body_faces_np_tmp.pkl'):
        body_faces_np_tmp = []
        for i, t in enumerate(body_faces_np):
            #if np.all( t in np.array(wo_head).astype(np.int32) ):
            if (t[0 ] in wo_head) and (t[1] in wo_head) and (t[2] in wo_head):
                tmp = [mapping[j] for j in t ]
                body_faces_np_tmp.append( tmp )
            if i % 5000 == 0:
                print('%d / %d' % (i, body_faces_np.shape[0]))
        body_faces_np = np.array( body_faces_np_tmp )

        with open('./body_faces_np_tmp.pkl', 'wb') as fout:
            pkl.dump({'body_faces_np': body_faces_np}, fout)

    else:
        with open('./body_faces_np_tmp.pkl', 'rb') as fin:
            body_faces_np = pkl.load(fin)['body_faces_np']


    m = Mesh(v=vertices_np, f=body_faces_np)

    ftov = m.faces_by_vertex(as_sparse_matrix=True)

    ftov = sparse.coo_matrix(ftov)
    indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).to(device)
    values = torch.FloatTensor(ftov.data).to(device)
    shape = ftov.shape
    ftov = torch.sparse.FloatTensor(indices, values, torch.Size(shape))


    # Read the scene scan if any
    scene_v = scene_vn = scene_f = None

    if scene_name is not None:
        if load_scene:
            scene = Mesh(filename=os.path.join(scene_dir, scene_name + '.ply'))

            scene.vn = scene.estimate_vertex_normals()

            scene_v = torch.tensor(scene.v[np.newaxis, :],
                                   dtype=dtype,
                                   device=device).contiguous()
            scene_vn = torch.tensor(scene.vn[np.newaxis, :],
                                    dtype=dtype,
                                    device=device)
            scene_f = torch.tensor(scene.f.astype(int)[np.newaxis, :],
                                   dtype=torch.long,
                                   device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights
    if s2m:
        opt_weights_dict['s2m_weight'] = s2m_weights
    if m2s:
        opt_weights_dict['m2s_weight'] = m2s_weights
    if sdf_penetration:
        opt_weights_dict['sdf_penetration_weight'] = sdf_penetration_weights

    if contact:
        opt_weights_dict['contact_loss_weight'] = contact_loss_weights[2:]



    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # load indices of the head of smpl-x model
    with open( osp.join(body_segments_dir, 'body_mask.json'), 'r') as fp:
        head_indx = np.array(json.load(fp))
    N = body_model.get_num_verts()
    body_indx = np.setdiff1d(np.arange(N), head_indx)
    head_mask = np.in1d(np.arange(N), head_indx)
    body_mask = np.in1d(np.arange(N), body_indx)

    tmp = [False] * len(wo_head)

    for w in wo_head:
        if body_mask[w]:
            idx =  mapping[ w ]
            tmp[idx] = True
    body_mask = tmp


    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get('body_tri_idxs')

    # which initialization mode to choose: similar traingles, mean of the scan or the average of both
    if init_mode == 'scan':
        init_t = init_trans
    elif init_mode == 'both':
        init_t = (init_trans.to(device) + fitting.guess_init(body_model, gt_joints, edge_indices,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    model_type=kwargs.get('model_type', 'smpl'),
                                    focal_length=focal_length_x, dtype=dtype) ) /2.0

    else:
        init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    model_type=kwargs.get('model_type', 'smpl'),
                                    focal_length=focal_length_x, dtype=dtype)

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               s2m=s2m,
                               m2s=m2s,
                               rho_s2m=rho_s2m,
                               rho_m2s=rho_m2s,
                               head_mask=head_mask,
                               body_mask=body_mask,
                               sdf_penetration=sdf_penetration,
                               voxel_size=voxel_size,
                               grid_min=grid_min,
                               grid_max=grid_max,
                               sdf=sdf,
                               sdf_normals=sdf_normals,
                               R=R,
                               t=t,
                               contact=contact,
                               contact_verts_ids=contact_verts_ids,
                               rho_contact=rho_contact,
                               contact_angle=contact_angle,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=1, visualize=visualize, viz_mode=viz_mode, **kwargs) as monitor:

        H, W, _ = 1080, 1920, 0

        # Reset the parameters to estimate the initial translation of the
        # body model
        if camera_mode == 'moving':
            body_model.reset_params(body_pose=body_mean_pose)
            with torch.no_grad():
                camera.translation[:] = init_t.view_as(camera.translation)
                camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

            # Re-enable gradient calculation for the camera translation
            camera.translation.requires_grad = True

            camera_opt_params = [camera.translation, body_model.global_orient]

        elif camera_mode == 'fixed':
            pass

        orientations = [body_model.global_orient.detach().cpu().numpy()]

        results = []
        body_transl = body_model.transl.clone().detach()
        # Step 2: Optimize the full model
        final_loss_val = 0

        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            mp = os.path.basename( pc[0]['mesh_path'] )

            with torch.no_grad():
                # Grasp pose, 01_20r
                if '1.obj' in mp or '5.obj' in mp or '8.obj' in mp:
                    flat_right = [-1.79910542,  0.53262842,  1.28660498,  0.21811475, -0.10176443, 0.27301444,  0.19337532, -0.49830927, -1.09764117, -0.10203813, -0.16491107,  0.15277187]
                    flat_left = [-1.79910542,  0.53262842,  1.28660498,  0.21811475, -0.10176443, 0.27301444,  0.19337532, -0.49830927, -1.09764117, -0.10203813, -0.16491107,  0.15277187]

                # Flat hand pose
                elif '10.obj' in mp or '11.obj' in mp:
                    flat_right = [1.3358688354492188, 0.8030816912651062, 0.15652795135974884, 0.8655192852020264, 0.23036262392997742, -0.1905936896800995, -0.25475624203681946, 0.17347466945648193, 1.1129142045974731, -0.3046478033065796, -0.36282816529273987, -1.2045189142227173, 0.2988683879375458, -0.5423060655593872, -0.01983018033206463, 1.4432138204574585, 1.9869898557662964, -1.3893120288848877]
                    flat_left = [1.0009729862213135, 0.37132811546325684, -0.3274688720703125, 0.9419571161270142, -0.33265185356140137, 0.3768235743045807, 0.521725594997406, -0.4584660828113556, -0.18133953213691711, 0.08398113399744034, -0.2741631269454956, -0.3443540334701538, 1.18972647190094, 0.4671509861946106, -0.9719769358634949, 1.0913945436477661, 1.478946328163147, -0.9113854765892029]


                if ('1.obj' in mp and '11.obj' not in mp) or '5.obj' in mp or '8.obj' in mp:
                    for fid in range(len(scan[0])):
                        for c in range(12):
                            reft_hand_pose_per_frame[0, c + fid*12] = flat_right[c]

                        for c in range(12):
                            left_hand_pose_per_frame[0, c + fid*12] = flat_left[c]

            # more than 1 instances
            betas_per_frame = betas_per_frame[:1]

            total = transl_per_frame.reshape([-1, 3]).shape[0]

            left_hand_pose_per_frame_aug = left_hand_pose_per_frame.reshape([-1, 12])
            reft_hand_pose_per_frame_aug = reft_hand_pose_per_frame.reshape([-1, 12])

            new_params = defaultdict(transl=transl_per_frame.reshape([-1, 3]),
                                     global_orient=global_orient_per_frame.reshape([-1, 3]),
                                     betas = betas_per_frame.repeat([total, 1]),
                                     body_pose=bodypose.reshape([-1, 63]),
                                     jaw_pose=jaw_pose_per_frame.reshape([-1, 3]),
                                     leye_pose=leye_pose_per_frame.reshape([-1, 3]),
                                     reye_pose=reye_pose_per_frame.reshape([-1, 3]),
                                     expression=expression_per_frame.reshape([-1, 10]),

                                     left_hand_pose= left_hand_pose_per_frame_aug,
                                     right_hand_pose= reft_hand_pose_per_frame_aug)


            body_model.reset_params(**new_params)                


            #for opt_idx, curr_weights in enumerate(tqdm(opt_weights[-1:], desc='Stage')):
            for opt_idx, curr_weights in enumerate(tqdm(opt_weights[-1:] * 2, desc='Stage')):
                body_model.transl.requires_grad = True

                if opt_idx == 0:
                    body_model.betas.requires_grad = True
                else:
                    body_model.betas.requires_grad = False

                body_model.left_hand_pose.requires_grad = True
                body_model.right_hand_pose.requires_grad = True

                body_params = list(body_model.parameters())

                final_params = list(                        
                    filter(lambda x: x.requires_grad, body_params))


                if opt_idx == 0:
                    pc = pc[:total]

                for j, p in enumerate(pc):
                    if opt_idx == 0:
                        pc[j]['obj_pose_ori'] = p['pose']
                        pc[j]['obj_trans_ori'] = p['trans']

                        pc[j]['pose'] = torch.from_numpy(p['pose']).type(torch.float32).to(device=device)
                        pc[j]['trans'] = torch.from_numpy(p['trans']).type(torch.float32).to(device=device)
                        pc[j]['pose'].requires_grad = True
                        pc[j]['trans'].requires_grad = True

                    final_params.append( pc[j]['pose'] )
                    final_params.append( pc[j]['trans'] )


                #pc[0]['ob'] = torch.from_numpy(pc[0]['ob'].vertices).type(torch.float32).to(device=device)
                if opt_idx == 0:
                    pc[0]['ob_v'] = torch.from_numpy(pc[0]['ob_v']).type(torch.float32).to(device=device)
                    pc[0]['ob_f'] = torch.from_numpy(pc[0]['ob_f']).to(device=device)

                    pc[0]['ob_normal'] = torch.from_numpy(pc[0]['ob_normal']).type(torch.float32).to(device=device)


                if use_vposer:
                    final_params.append(pose_embedding)


                if opt_idx == 0:
                    kwargs['maxiters'] = 100
                else:
                    kwargs['maxiters'] = 600


                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:76] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 76:] = curr_weights['face_weight']

                loss.reset_loss_weights(curr_weights)


                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    #camera=camera, gt_joints_whole=[gt_joints, keypoints[3], keypoints[4], keypoints[5]],
                    camera=camera, gt_joints_whole=gt_joints, pc=pc,

                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    scan_tensor=scan_tensors,
                    scene_v=scene_v, scene_vn=scene_vn, scene_f=scene_f,ftov=ftov, contact_angle=15, rho_contact=0.05,
                    return_verts=False, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()

                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer, maxiters=1,
                    use_vposer=use_vposer)


                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera[0].named_parameters()}

            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})


            if use_vposer:
                result['pose_embedding'] = pose_embedding.detach().cpu().numpy()
                body_pose = vposer.decode(
                    pose_embedding.view([-1, 32]),
                    output_type='aa').view(1, -1) if use_vposer else None
                result['body_pose'] = body_pose.detach().cpu().numpy()
            else:
                result['body_pose'] = body_model.body_pose.detach().cpu().numpy().reshape([-1, 63])

                for j, p in enumerate(pc):
                    pc[j]['pose'] = torch.from_numpy(p['pose']).type(torch.float32).to(device=device)
                    pc[j]['trans'] = torch.from_numpy(p['trans']).type(torch.float32).to(device=device)
                    pc[j]['pose'].requires_grad = True
                    pc[j]['trans'].requires_grad = True

            result['ob_pose'] = [p['pose'].detach().cpu().numpy() for p in pc]
            result['ob_trans'] = [p['trans'].detach().cpu().numpy() for p in pc]

            results.append({'loss': final_loss_val,
                            'result': result})

        print('Save result ')
        import pickle as pkl
        with open(result_fn.replace('00000/000.pkl', '%05d_temp.pkl' % 0), 'wb') as fout:
            pickle.dump(results[0]['result'], fout, protocol=2)


    if save_meshes or visualize:
        model_output = body_model(return_verts=True, body_pose=torch.tensor(result['body_pose'].reshape([-1, 63])).to(device=device))

        batch_id = kwargs['batch_id']

        vertices = model_output.vertices.detach().cpu().numpy().squeeze()
        if len(vertices.shape) < 3:
            vertices = vertices[None]

        for i in range(0, keypoint_data.shape[1]):
            #vertices = body_models_output[i][0].vertices
            #vertices = vertices.detach().cpu().numpy().squeeze()
            i2 = i 
            import trimesh

            out_mesh = trimesh.Trimesh(vertices[i], body_model.faces, process=False)
            out_mesh.export(mesh_fn.replace('000.ply', '%05d_second.ply' % i2))
            out_mesh.export(mesh_fn.replace('000.ply', '%05d_second.obj' % i2))


            import cv2
            vertices_trans = np.matmul(pc[0]['ob_v'].detach().cpu().numpy(), cv2.Rodrigues(pc[i]['pose'].detach().cpu().numpy().squeeze())[0].T) + pc[i]['trans'].detach().cpu().numpy()
            out_mesh_obj = trimesh.Trimesh(vertices_trans, pc[0]['ob_f'].cpu(), process=False)
            out_mesh_obj.export(mesh_fn.replace('000.ply', '%05d_second_obj.ply' % i2))

            vertices_trans = np.matmul(pc[0]['ob_v'].detach().cpu().numpy(), cv2.Rodrigues(pc[i]['obj_pose_ori'].squeeze())[0].T) + pc[i]['obj_trans_ori']
            out_mesh_obj = trimesh.Trimesh(vertices_trans, pc[0]['ob_f'].cpu(), process=False)
            out_mesh_obj.export(mesh_fn.replace('000.ply', '%05d_temp_obj_ori.ply' % i2))
