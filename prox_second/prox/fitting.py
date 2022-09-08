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

import open3d as o3d
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from psbody.mesh.visibility import visibility_compute
from psbody.mesh import Mesh
from AE_sep import Enc

import misc_utils as utils
import dist_chamfer as ext
distChamfer = ext.chamferDist()

global first_or_not
first_or_not = True
global vis_maps_global 
vis_maps_global = []

global scan_tensor_global
scan_tensor_global = []

global mintheta
global maxtheta
mintheta = None
maxtheta = None

global obj_sdf
obj_sdf = None



WO_PC_OR_NOT = False
VIEW_SIZE = 6

global POINT
global NORMAL

POINT = np.array([-0.07362306, 1.06334016, 2.88126607])
NORMAL = np.array([-0.02645398, 0.98627183, 0.16299712]) * -1

# Piont clouds loss wtihout hands
with open('../models/wo_head.pkl', 'rb') as fin:
    import pickle as pkl
    wo_head = pkl.load(fin)
    SMOOTH_MARKER_IDS = wo_head['marker_ids']
    wo_head = wo_head['wo_head']
    # Remove hands

with open('../models/wo_head_hands.pkl', 'rb') as fin:
    import pickle as pkl
    wo_head_ = pkl.load(fin)
    wo_head_ = wo_head_['hands_idx']

body_pcs_wohands = []
for i, j in enumerate(wo_head):
    if j not in wo_head_:
        body_pcs_wohands.append(i)


motion_smooth_model = None
Xmean_global_markers = None
Xstd_global_markers = None




@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 viz_mode='mv',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

        self.visualize = visualize
        self.viz_mode = viz_mode

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            if self.viz_mode == 'o3d':
                self.vis_o3d = o3d.Visualizer()
                self.vis_o3d.create_window()
                self.body_o3d = o3d.TriangleMesh()
                self.scan = o3d.PointCloud()
            else:
                self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            if self.viz_mode == 'o3d':
                self.vis_o3d.close()
            else:
                self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''

        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None


        for n in range(self.maxiters):
            print('Step %d' % n)

            loss = optimizer.step(closure)
            print('After %d' % n)
            print()

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    print('Less than ftol')
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            prev_loss = loss.item()

            break

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, pc=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               scan_tensor=None,
                               create_graph=False,
                               **kwargs):
        with open('./body_faces_np_tmp.pkl', 'rb') as fin:
            faces_tensor = pkl.load(fin)['body_faces_np']
        faces_tensor = torch.tensor(faces_tensor, dtype=torch.long)
        faces_tensor = faces_tensor.view(-1)



        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding.squeeze().reshape([-1, 32]), output_type='aa').view(
                    -1, 63) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_models_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)

            total_loss = loss(body_models_output, camera=camera,
                              gt_joints=gt_joints, pc=pc,
                              body_model=None,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              scan_tensor=scan_tensor,
                              visualize=self.visualize,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph, retain_graph=True)


            if self.visualize:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                if self.steps == 0 and self.viz_mode == 'o3d':

                    self.body_o3d.vertices = o3d.Vector3dVector(vertices.squeeze())
                    self.body_o3d.triangles = o3d.Vector3iVector(body_model.faces)
                    self.body_o3d.vertex_normals = o3d.Vector3dVector([])
                    self.body_o3d.triangle_normals = o3d.Vector3dVector([])
                    self.body_o3d.compute_vertex_normals()
                    self.vis_o3d.add_geometry(self.body_o3d)

                    if scan_tensor is not None:
                        self.scan.points = o3d.Vector3dVector(scan_tensor.detach().cpu().numpy().squeeze())
                        N = np.asarray(self.scan.points).shape[0]
                        self.scan.colors = o3d.Vector3dVector(np.tile([1.00, 0.75, 0.80], [N, 1]))
                        self.vis_o3d.add_geometry(self.scan)

                    self.vis_o3d.update_geometry()
                    self.vis_o3d.poll_events()
                    self.vis_o3d.update_renderer()
                elif self.steps % self.summary_steps == 0:
                    if self.viz_mode == 'o3d':
                        self.body_o3d.vertices = o3d.Vector3dVector(vertices.squeeze())
                        self.body_o3d.triangles = o3d.Vector3iVector(body_model.faces)
                        self.body_o3d.vertex_normals = o3d.Vector3dVector([])
                        self.body_o3d.triangle_normals = o3d.Vector3dVector([])
                        self.body_o3d.compute_vertex_normals()

                        self.vis_o3d.update_geometry()
                        self.vis_o3d.poll_events()
                        self.vis_o3d.update_renderer()
                    else:
                        self.mv.update_mesh(vertices.squeeze(),
                                        body_model.faces)
            self.steps += 1

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 s2m=False,
                 m2s=False,
                 rho_s2m=1,
                 rho_m2s=1,
                 s2m_weight=0.0,
                 m2s_weight=0.0,
                 head_mask=None,
                 body_mask=None,
                 sdf_penetration=False,
                 voxel_size=None,
                 grid_min=None,
                 grid_max=None,
                 sdf=None,
                 sdf_normals=None,
                 sdf_penetration_weight=0.0,
                 R=None,
                 t=None,
                 contact=False,
                 contact_loss_weight=0.0,
                 contact_verts_ids=None,
                 rho_contact=0.0,
                 contact_angle=0.0,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.s2m = s2m
        self.m2s = m2s

        self.s2m_robustifier = utils.GMoF(rho=rho_s2m)
        self.m2s_robustifier = utils.GMoF(rho=rho_m2s)

        self.shape_prior = shape_prior
        # Newly added
        self.body_pose_prior = body_pose_prior

        self.body_mask = body_mask
        self.head_mask = head_mask

        self.R = R
        self.t = t

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance


        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

        self.register_buffer('s2m_weight',
                             torch.tensor(s2m_weight, dtype=dtype))
        self.register_buffer('m2s_weight',
                             torch.tensor(m2s_weight, dtype=dtype))

        self.sdf_penetration = sdf_penetration
        if self.sdf_penetration:
            self.sdf = sdf
            self.sdf_normals = sdf_normals
            self.voxel_size = voxel_size
            self.grid_min = grid_min
            self.grid_max = grid_max
            self.register_buffer('sdf_penetration_weight',
                                 torch.tensor(sdf_penetration_weight, dtype=dtype))
        self.contact = contact
        if self.contact:
            self.contact_verts_ids = contact_verts_ids
            self.rho_contact = rho_contact
            self.contact_angle = contact_angle
            self.register_buffer('contact_loss_weight',
                                 torch.tensor(contact_loss_weight, dtype=dtype))
            self.contact_robustifier = utils.GMoF_unscaled(rho=self.rho_contact)

        self.rho_contact = rho_contact
        self.register_buffer('contact_loss_weight',
                             torch.tensor(contact_loss_weight, dtype=dtype))
        self.contact_robustifier = utils.GMoF_unscaled(rho=self.rho_contact)

    def reset_loss_weights(self, loss_weight_dict):
            for key in loss_weight_dict:
                if hasattr(self, key):
                    weight_tensor = getattr(self, key)
                    if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                        weight_tensor = loss_weight_dict[key].clone().detach()
                    else:
                        weight_tensor = torch.tensor(loss_weight_dict[key],
                                                     dtype=weight_tensor.dtype,
                                                     device=weight_tensor.device)
                    setattr(self, key, weight_tensor)

    #def forward(self, body_model_output, camera, gt_joints_whole, joints_conf,
    def forward(self, body_model_output, camera, gt_joints_whole, pc, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                scan_tensor=None, visualize=False,
                scene_v=None, scene_vn=None, scene_f=None,ftov=None,
                **kwargs):

        total = len(body_model_output.A)

        loss_all = []
        ob_transed_all = []
        total_loss = 0

        #opt_matrix_all = []
        global first_or_not
        global scan_tensor_global

        if not WO_PC_OR_NOT and first_or_not:
            for i in range(total):
                print('Pre-process %d' % i)
                body_vertices = body_model_output.vertices[i]
                body_vertices_mean = torch.mean(body_vertices, axis=0)

                per_frame_tmp = []
                for c in range( len(camera) ):
                    scan_tensor_points = scan_tensor[c][i]['points']

                    scan_tensor_points = torch.tensor(scan_tensor_points, device=body_model_output.vertices.device, dtype=body_model_output.vertices.dtype)#.unsqueeze(0)
                    scan_tensor_points = scan_tensor_points[:].unsqueeze(0)

                    per_frame_tmp.append( scan_tensor_points )

                scan_tensor_global.append( per_frame_tmp )

            
            with open('./body_faces_np_tmp.pkl', 'rb') as fin:
                body_faces_np = pkl.load(fin)['body_faces_np']

            globals()['body_faces_np'] = body_faces_np


        torch.cuda.empty_cache()

        def offset_(x_, r, t):
            res = torch.matmul(x_, r) + t
            return res

        for i in range(total):
            projected_joints = camera[0](body_model_output.joints[i, None, :, :])
            projected_joints_2 = camera[1](body_model_output.joints[i, None, :, :])
            projected_joints_3 = camera[2](body_model_output.joints[i, None, :, :])
            projected_joints_4 = camera[3](body_model_output.joints[i, None, :, :])

            projected_joints_5 = camera[4](body_model_output.joints[i, None, :, :])
            projected_joints_6 = camera[5](body_model_output.joints[i, None, :, :])


            # Calculate the weights for each joints
            weights = (joint_weights * joints_conf[0][i]
                       if self.use_joints_conf else
                       joint_weights).unsqueeze(dim=-1)
            weights_2 = (joint_weights * joints_conf[1][i]
                       if self.use_joints_conf else
                       joint_weights).unsqueeze(dim=-1)
            weights_3 = (joint_weights * joints_conf[2][i]
                       if self.use_joints_conf else
                       joint_weights).unsqueeze(dim=-1)
            weights_4 = (joint_weights * joints_conf[3][i]
                       if self.use_joints_conf else
                       joint_weights).unsqueeze(dim=-1)
            weights_5 = (joint_weights * joints_conf[4][i]
                       if self.use_joints_conf else
                       joint_weights).unsqueeze(dim=-1)
            weights_6 = (joint_weights * joints_conf[5][i]
                       if self.use_joints_conf else
                       joint_weights).unsqueeze(dim=-1)

            gt_joints = gt_joints_whole[:, i]


            # Calculate the distance of the projected joints from
            # the ground truth 2D detections
            def not_all_zero_test(joint):
                joint = joint.detach().cpu().numpy()
                res = np.any(joint)
                return res

            joint_diff = self.robustifier( (gt_joints[0] - projected_joints) if not_all_zero_test(gt_joints[0]) else (projected_joints - projected_joints) )
            joint_diff_2 = self.robustifier( (gt_joints[1] - projected_joints_2) if not_all_zero_test(gt_joints[1]) else (projected_joints_2 - projected_joints_2) )
            joint_diff_3 = self.robustifier( (gt_joints[2] - projected_joints_3) if not_all_zero_test(gt_joints[2]) else (projected_joints_3 - projected_joints_3) )
            joint_diff_4 = self.robustifier( (gt_joints[3] - projected_joints_4) if not_all_zero_test(gt_joints[3]) else (projected_joints_4 - projected_joints_4) )
            joint_diff_5 = self.robustifier( (gt_joints[4] - projected_joints_5) if not_all_zero_test(gt_joints[4]) else (projected_joints_5 - projected_joints_5) )
            joint_diff_6 = self.robustifier( (gt_joints[5] - projected_joints_6) if not_all_zero_test(gt_joints[5]) else (projected_joints_6 - projected_joints_6) )


            # 
            self.data_weight = self.data_weight * 1.0

            joint_loss = (torch.sum(weights ** 1 * joint_diff) *
                          self.data_weight ** 2)
            joint_loss_2 = (torch.sum(weights_2 ** 1 * joint_diff_2) *
                          self.data_weight ** 2)
            joint_loss_3 = (torch.sum(weights_3 ** 1 * joint_diff_3) *
                          self.data_weight ** 2)
            joint_loss_4 = (torch.sum(weights_4 ** 1 * joint_diff_4) *
                          self.data_weight ** 2)
            joint_loss_5 = (torch.sum(weights_5 ** 1 * joint_diff_5) *
                          self.data_weight ** 2)
            joint_loss_6 = (torch.sum(weights_6 ** 1 * joint_diff_6) *
                          self.data_weight ** 2)


            # Calculate the loss from the Pose prior
            if use_vposer:
                pprior_loss = torch.sum(self.body_pose_prior(
                    body_model_output.body_pose.reshape([1, -1, 63])[:, i],
                    body_model_output.betas)) * self.body_pose_weight ** 2
            else:
                pprior_loss = torch.sum(self.body_pose_prior(
                    body_model_output.body_pose.reshape([1, -1, 63])[:, i],
                    body_model_output.betas)) * self.body_pose_weight ** 2

            shape_loss = torch.sum(self.shape_prior(
                body_model_output.betas[0])) * self.shape_weight ** 2

            body_pose = body_model_output.full_pose[i, None, 3:66]
            angle_prior_loss = torch.sum(
                self.angle_prior(body_pose)) * self.bending_prior_weight ** 2

            # Apply the prior on the pose space of the hand
            left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0


            def hand_limit_prior(hand_pose):
                MINBOUND = -5.
                MAXBOUND = 5.

                minThetaVals = np.array([MINBOUND, MINBOUND, MINBOUND, # global rot
                    0, -0.15, 0.1, -0.3, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # index
                    MINBOUND, -0.15, 0.1, -0.5, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # middle
                    -1.5, -0.15, -0.1, MINBOUND, -0.5, -0.0, MINBOUND, MINBOUND, 0, # pinky
                    -0.5, -0.25, 0.1, -0.4, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # ring
                    0.0, -0.83, -0.0, -0.15, MINBOUND, 0, MINBOUND, -0.5, -1.57, ]) # thumb
                maxThetaVals = np.array([MAXBOUND, MAXBOUND, MAXBOUND, #global
                    0.45, 0.2, 1.8, 0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # index
                    MAXBOUND, 0.15, 2.0, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # middle
                    -0.2, 0.6, 1.6, MAXBOUND, 0.6, 2.0, MAXBOUND, MAXBOUND, 1.25, # pinky
                    -0.4, 0.10, 1.8, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # ring
                    2.0, 0.66, 0.5, 1.6, MAXBOUND, 0.5, MAXBOUND, 0, 1.08]) # thumb
                validThetaIDs = np.array([0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29,
                    30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 44, 46, 47], dtype=np.int32)


                global mintheta
                global maxtheta

                if mintheta == None and maxtheta == None:
                    mintheta = minThetaVals[validThetaIDs[3:]]
                    mintheta = torch.tensor(mintheta).to(device=hand_pose.device)

                    maxtheta = maxThetaVals[validThetaIDs[3:]]
                    maxtheta = torch.tensor(maxtheta).to(device=hand_pose.device)

                loss_max = mintheta - hand_pose[validThetaIDs[3:]-3]
                loss_min = hand_pose[validThetaIDs[3:]-3] - maxtheta

                loss_max = torch.square(loss_max[loss_max > 0])
                loss_min = torch.square(loss_min[loss_min > 0])

                loss = torch.sum(loss_max) + torch.sum(loss_min)

                return loss * (self.hand_prior_weight * 1e2) ** 2

            if self.use_hands and self.left_hand_prior is not None:
                left_hand_prior_loss = torch.sum(
                    self.left_hand_prior(
                        body_model_output.left_hand_pose.reshape([-1, 45])[i])) * self.hand_prior_weight ** 2  

            if self.use_hands and self.right_hand_prior is not None:
                right_hand_prior_loss = torch.sum(
                    self.right_hand_prior(
                        body_model_output.right_hand_pose.reshape([-1, 45])[i])) *  self.hand_prior_weight ** 2 

            tmp_vertices = body_model_output.vertices[i, None, :, :]

            expression_loss = 0.0
            jaw_prior_loss = 0.0
            if self.use_face:
                expression_loss = torch.sum(self.expr_prior(
                    #body_model_output[i][0].expression)) * \
                    body_model_output.expression[i])) * \
                    self.expr_prior_weight ** 2

                if hasattr(self, 'jaw_prior'):
                    jaw_prior_loss = torch.sum(
                        self.jaw_prior(
                            #body_model_output[i][0].jaw_pose.mul(
                            body_model_output.jaw_pose[i].mul(
                                self.jaw_prior_weight)))

            def opendr_visibility(vvv, fff):
                m = Mesh(v=vvv, f=fff)

                from random import randint
                TMP_FILE_PATH = '/tmp/temp_%d.ply' % randint(0, 1e10)
                
                import os
                if os.path.exists(TMP_FILE_PATH):
                    os.remove(TMP_FILE_PATH)

                m.write_ply(TMP_FILE_PATH)
                m = Mesh(filename=TMP_FILE_PATH)

                if os.path.exists(TMP_FILE_PATH):
                    os.remove(TMP_FILE_PATH)

                (vis, n_dot) = visibility_compute(v=m.v, f=m.f, cams=np.array([[0.0, 0.0, 0.0]]))
                return vis.squeeze()


            vis_maps = []

            #for c in [0, 1, 2, 3]:
            for c in [0, 1, 2, 3, 4, 5]:
                if WO_PC_OR_NOT:
                    break
                if not first_or_not:
                    break

                vertices_np = camera[c].transform(body_model_output.vertices[i][None, :, :]).detach().cpu().numpy().squeeze()
                body_faces_np = globals()['body_faces_np']

                res = opendr_visibility(vertices_np, body_faces_np)

                vis_maps.append(res)


            if first_or_not:
                vis_maps_global.append(vis_maps)
            else:
                vis_maps = vis_maps_global[i]



            s2m_dist = []
            m2s_dist = []

            if i % 200 == 0:
                print('%d / %d' % (i, total))

            for c in range(len(scan_tensor)):                    
                if WO_PC_OR_NOT:
                    break

                if self.s2m and self.s2m_weight > 0 and np.sum(vis_maps[c]) > 0:
                    tmp, _, _, _ = distChamfer( scan_tensor_global[i][c],
                            body_model_output.vertices[i][None, list( set(body_pcs_wohands).intersection(set( np.where(vis_maps[c] > 0)[0])) ), :])

                    tmp = self.s2m_robustifier(tmp.sqrt())
                    tmp = self.s2m_weight * tmp.sum()  * 1e8
                    s2m_dist.append(tmp)

                if self.m2s and self.m2s_weight > 0 and np.sum(vis_maps[c]) > 0:
                    _, tmp, _, _ = distChamfer(  scan_tensor_global[i][c],
                            body_model_output.vertices[i][None, list( set(body_pcs_wohands).intersection( set(np.where(np.logical_and(vis_maps[c] > 0, self.body_mask))[0]) ) ), :])



                    tmp = self.m2s_robustifier(tmp.sqrt())
                    tmp = self.m2s_weight * tmp.sum() * 1e8
                    m2s_dist.append(tmp)    


            # Turn off s2m loss
            if not WO_PC_OR_NOT:
                s2m_dist = torch.sum( torch.stack(s2m_dist) ) if len(s2m_dist) > 0 else 0.0            
                m2s_dist = torch.sum( torch.stack(m2s_dist) ) if len(m2s_dist) > 0 else 0.0

            # This is the interpenetration loss
            global obj_sdf

            import smplx
            opt_matrix = smplx.lbs.batch_rodrigues( pc[i]['pose'][None] )
            ob_transed = torch.matmul(pc[0]['ob_v'], opt_matrix.squeeze().T) + pc[i]['trans']
            ob_transed_all.append( ob_transed )

            if obj_sdf is None:
                from sdf_compute import compute_sdf
                #obj_sdf = compute_sdf(pc[0]['ob_v'].detach().cpu().numpy(), pc[0]['ob_f'].detach().cpu().numpy(), device=joint_loss.device)

                mp =  pc[0]['mesh_path'].split('/')[-1]

                if '08.obj' in mp or '09.obj' in mp:
                    obj_sdf = compute_sdf(pc[0]['ob_v'].detach().cpu().numpy(), pc[0]['ob_f'].detach().cpu().numpy(), device=joint_loss.device, flag=1)
                else:
                    obj_sdf = compute_sdf(pc[0]['ob_v'].detach().cpu().numpy(), pc[0]['ob_f'].detach().cpu().numpy(), device=joint_loss.device)

                obj_sdf['sdf'] = torch.tensor(obj_sdf['sdf']).to(device=joint_loss.device)
                #obj_sdf = compute_sdf(ob_transed.detach().cpu().numpy(), pc[0]['ob_f'].detach().cpu().numpy(), device=joint_loss.device)
                
            right_hand_obj = body_model_output.vertices[i] - pc[i]['trans']
            right_hand_obj = torch.matmul(right_hand_obj, opt_matrix.squeeze())

            nv = right_hand_obj.shape[0]
            
            grid_dim = obj_sdf['sdf'].shape[0]
            grid_min = np.array( obj_sdf['min'] )
            grid_max = np.array( obj_sdf['max'] )
            voxel_size = np.array( obj_sdf['voxel_size'] )
            sdf_normals = np.array( obj_sdf['normals'] )
            cur_sdf = obj_sdf['sdf']

            grid_dim = torch.tensor(grid_dim).to(device=joint_loss.device)
            grid_min = torch.tensor(grid_min).to(device=joint_loss.device)
            grid_max = torch.tensor(grid_max).to(device=joint_loss.device)
            voxel_size = torch.tensor(voxel_size).to(device=joint_loss.device)
            sdf_normals = torch.tensor(sdf_normals).to(device=joint_loss.device)
            #cur_sdf = torch.tensor(cur_sdf).to(device=joint_loss.device)


            sdf_ids = torch.round(
               (right_hand_obj.squeeze() - grid_min) / voxel_size).to(dtype=torch.long)
            sdf_ids.clamp_(min=0, max=grid_dim-1)

            norm_vertices = (right_hand_obj - grid_min) / (grid_max - grid_min) * 2 - 1

            #body_sdf = F.grid_sample(obj_sdf['sdf'].view(1, 1, grid_dim, grid_dim, grid_dim),
            body_sdf = F.grid_sample(cur_sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
                                     norm_vertices[:, [2, 1, 0]].view(1, nv, 1, 1, 3),
                                     padding_mode='border')
            sdf_normals = sdf_normals[sdf_ids[:,0], sdf_ids[:,1], sdf_ids[:,2]]

            # if there are no penetrating vertices then set sdf_penetration_loss = 0
            if body_sdf.lt(0).sum().item() < 1:
                sdf_penetration_loss = torch.tensor(0.0, dtype=joint_loss.dtype, device=joint_loss.device)
            else:
                mp =  pc[0]['mesh_path'].split('/')[-1]
                if '08.obj' in mp or '09.obj' in mp:
                    sdf_penetration_loss = 5e7 * (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() * sdf_normals[body_sdf.view(-1) < 0, :]).pow(2).sum(dim=-1).sqrt().sum()
                else:
                    sdf_penetration_loss = 5e3 * (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() * sdf_normals[body_sdf.view(-1) < 0, :]).pow(2).sum(dim=-1).sqrt().sum()


            contact_body_vertices = body_model_output.vertices[i][None, self.contact_verts_ids, :] # Ignore hand vertices

            contact_dist, _, idx1, _ = distChamfer(
                contact_body_vertices.contiguous(), ob_transed[None, :, :])

            body_model_faces = body_model_faces.to(device=joint_loss.device)


            body_triangles = torch.index_select(
                body_model_output.vertices[i], 0,
                body_model_faces).view(1, -1, 3, 3)

            # Calculate the edges of the triangles
            # Size: BxFx3
            edge0 = body_triangles[:, :, 1] - body_triangles[:, :, 0]
            edge1 = body_triangles[:, :, 2] - body_triangles[:, :, 0]
            # Compute the cross product of the edges to find the normal vector of
            # the triangle
            body_normals = torch.cross(edge0, edge1, dim=2)
            # Normalize the result to get a unit vector
            body_normals = body_normals / \
                torch.norm(body_normals, 2, dim=2, keepdim=True)
            # compute the vertex normals
            body_v_normals = torch.mm(ftov, body_normals.squeeze())
            body_v_normals = body_v_normals / \
                torch.norm(body_v_normals, 2, dim=1, keepdim=True)

            contact_body_verts_normals = body_v_normals[self.contact_verts_ids, :]


            contact_scene_normals = body_v_normals[idx1.squeeze().to(
                    dtype=torch.long), :].squeeze()

            angles = torch.asin(
                    torch.norm(torch.cross(contact_body_verts_normals, contact_scene_normals), 1, dim=1, keepdim=True)) *180 / np.pi

            valid_contact_mask = (angles.le(15.) + angles.ge(180 - 15.)).ge(1)

            valid_contact_ids = valid_contact_mask.squeeze().nonzero().squeeze()
            valid_contact_ids = valid_contact_ids.flatten()

            if sum(valid_contact_ids) > 0:
                contact_dist = self.contact_robustifier(contact_dist[:, :].sqrt())                                
                contact_loss = 1e5 * contact_dist.mean()
                total_loss += contact_loss / total


            # For rendering loss
            seg_loss = []
            for s in range( VIEW_SIZE ):
                seg = pc[i]['segs'][s]#.to(joint_loss.device)
                if not torch.is_tensor(seg) or torch.sum(seg) < 20:
                    continue

                seg = seg.to(joint_loss.device)
                image = pc[i]['renders_dyna'][s](ob_transed[None, :, :], pc[0]['ob_f'][None, :, :], mode='silhouettes')


                H, W = seg.shape[:2]

                tmp = torch.sum( (image.squeeze()[:H, :W] - seg) ** 2 )

                seg_loss.append(tmp * 1e0)


            if len(seg_loss) > 0:
                seg_loss = torch.stack(seg_loss)
                total_loss += 1e2 * torch.sum( seg_loss ) / total

            # Mapping loss
            if 'map_k' in pc[0].keys():
                map_loss = torch.sum( 1e5 * (body_model_output.vertices[i, pc[0]['map_k'], :] - ob_transed[ pc[0]['map_v'] ]) ** 2 )
                total_loss += map_loss
                
            # Transform vertices to world coordinates
            if self.R is not None and self.t is not None:
                vertices = tmp_vertices
                nv = vertices.shape[1]
                vertices.squeeze_()
                vertices = self.R.mm(vertices.t()).t() + self.t.repeat([nv, 1])
                vertices.unsqueeze_(0)

            # Compute scene penetration using signed distance field (SDF)
            sdf_penetration_loss = 0.0
            if self.sdf_penetration and self.sdf_penetration_weight > 0:
                grid_dim = self.sdf.shape[0]
                sdf_ids = torch.round(
                   (vertices.squeeze() - self.grid_min) / self.voxel_size).to(dtype=torch.long)
                sdf_ids.clamp_(min=0, max=grid_dim-1)

                norm_vertices = (vertices - self.grid_min) / (self.grid_max - self.grid_min) * 2 - 1
                body_sdf = F.grid_sample(self.sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
                                         norm_vertices[:, :, [2, 1, 0]].view(1, nv, 1, 1, 3),
                                         padding_mode='border')
                sdf_normals = self.sdf_normals[sdf_ids[:,0], sdf_ids[:,1], sdf_ids[:,2]]
                # if there are no penetrating vertices then set sdf_penetration_loss = 0
                if body_sdf.lt(0).sum().item() < 1:
                    sdf_penetration_loss = torch.tensor(0.0, dtype=joint_loss.dtype, device=joint_loss.device)
                else:
                    sdf_penetration_loss = self.sdf_penetration_weight * (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() * sdf_normals[body_sdf.view(-1) < 0, :]).pow(2).sum(dim=-1).sqrt().sum()

            total_loss += (joint_loss + joint_loss_2 + joint_loss_3 + joint_loss_4 + joint_loss_6 + pprior_loss + shape_loss +
                      left_hand_prior_loss + right_hand_prior_loss + m2s_dist + s2m_dist + sdf_penetration_loss + contact_loss) / (total * 1e5)


        # For acceleration loss
        if first_or_not:
            global POINT
            global NORMAL
            POINT = torch.from_numpy(POINT).to(joint_loss.device)
            NORMAL = torch.from_numpy(NORMAL).to(joint_loss.device).reshape([3, -1])

        if first_or_not:
            global motion_smooth_model
            global Xmean_global_markers
            global Xstd_global_markers
            motion_smooth_model = Enc(downsample=False, z_channel=64).to(body_model_output.vertices.device)
            weights = torch.load('./15217/Enc_last_model.pkl', map_location=lambda storage, loc: storage)
            motion_smooth_model.load_state_dict(weights)
            motion_smooth_model.eval()
            for param in motion_smooth_model.parameters():
                param.requires_grad = False

            preprocess_stats = np.load('./15217/preprocess_stats_smooth_withHand_global_markers.npz')
            Xmean_global_markers = torch.from_numpy(preprocess_stats['Xmean']).float().to(body_model_output.vertices.device)
            Xstd_global_markers = torch.from_numpy(preprocess_stats['Xstd']).float().to(body_model_output.vertices.device)

            first_or_not = False

        markers_smooth = body_model_output.vertices[:, SMOOTH_MARKER_IDS, :]  # [T, 67, 3]
        joints_3d = body_model_output.joints[:, :75]

        ##### transfrom to pelvis at origin, face y axis
        joints_frame0 = joints_3d[0].detach()  # [25, 3], joints of first frame
        x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
        x_axis[-1] = 0
        x_axis = x_axis / torch.norm(x_axis)
        z_axis = torch.tensor([0, 0, 1]).float().to(joint_loss.device)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)
        transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
        joints_3d = torch.matmul(joints_3d - joints_frame0[0], transf_rotmat)  # [T(/bs), 25, 3]
        markers_frame0 = markers_smooth[0].detach()
        markers_smooth = torch.matmul(markers_smooth - markers_frame0[0], transf_rotmat)  # [T(/bs), 67, 3]
        clip_img = markers_smooth.reshape(markers_smooth.shape[0], -1).unsqueeze(0)
        clip_img = (clip_img - Xmean_global_markers) / Xstd_global_markers
        clip_img = clip_img.permute(0, 2, 1).unsqueeze(1)  # [1, 1, d, T]

        ####### input res, encoder forward
        T = clip_img.shape[-1]
        clip_img_v = clip_img[:, :, :, 1:] - clip_img[:, :, :, 0:-1]
        ### input padding
        p2d = (8, 8, 1, 1)
        clip_img_v = F.pad(clip_img_v, p2d, 'reflect')
        ### forward
        motion_z, _, _, _, _, _ = motion_smooth_model(clip_img_v)


        ####### constraints on latent z
        motion_z_v = motion_z[:, :, :, 1:] - motion_z[:, :, :, 0:-1]
        motion_prior_smooth_loss = torch.mean(motion_z_v ** 2) * 1.0 #self.motion_prior_smooth_weight

        markers_vel = markers_smooth[1:] - markers_smooth[0:-1]
        markers_acc = markers_vel[1:] - markers_vel[0:-1]
        smooth_acc_loss = torch.mean(markers_acc ** 2)

        loss_acceleration_joint = (body_model_output.vertices[:(total-2)] + body_model_output.vertices[2:total] - 2*body_model_output.vertices[1:(total-1)]) * 1e3
        loss_acceleration_joint = torch.sum( torch.pow(loss_acceleration_joint, 2) ) / (total * 1. )

        ob_transed_all = torch.stack(ob_transed_all)
        loss_acceleration_obj = (ob_transed_all[:-2] + ob_transed_all[2:] - 2 * ob_transed_all[1:-1]) * 1e3
        loss_acceleration_obj = torch.sum( torch.pow(loss_acceleration_obj, 2) ) / (total * 1.)

        loss_acceleration = motion_prior_smooth_loss * 1e5 
        loss_acceleration = loss_acceleration * 1e10 / total
        total_loss = total_loss / 1e3 +  loss_acceleration  +  loss_acceleration_obj / total

        contact_ids = pc[0]['contact']
        obj_point = pc[0]['points']

        
        for cid in range(0, int( len(contact_ids)/2 )):

            sid = cid * 2
            eid = cid * 2 + 1

            sid = contact_ids[sid]-1
            eid = contact_ids[eid]

            if sid >= total:
                break
            if eid >= total+1:
                eid = total

            for fid in range(sid, eid):
                if pc[0]['mesh_path'].endswith('02.obj') or pc[0]['mesh_path'].endswith('03.obj'):
                    whole_palm_idx = [5618, 5619, 5620, 5621, 5622, 5623, 5624, 5625, 5626, 5627, 5628, 5629, 5630, 5631, 5632, 5633, 5634, 5635, 5636, 5637, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5650, 5651, 5652, 5653, 5654, 5655, 5656, 5657, 5658, 5659, 5660, 5661, 5662, 5663, 5664, 5665, 5666, 5667, 5668, 5669, 5670, 5671, 5672, 5673, 5674, 5675, 5676, 5677, 5678, 5679, 5680, 5681, 5682, 5683, 5684, 5685, 5686, 5687, 5688, 5689, 5690, 5691, 5692, 5695, 5696, 5697, 5698, 5699, 5700, 5701, 5702, 5703, 5704, 5705, 5706, 5707, 5708, 5709, 5710, 5711, 5712, 5713, 5714, 5715, 5716, 5718, 5719, 5720, 5721, 5722, 5723, 5724, 5725, 5739, 5741, 5742, 5743, 5744, 5745, 5746, 5747, 5748, 5749, 5750, 5751, 5752, 5753, 5754, 5755, 5756, 5757, 5758, 5759, 5760, 5761, 5762, 5763, 5764, 5765, 5766, 5767, 5768, 5769, 5770, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778, 5779, 5780, 5781, 5782, 5783, 5784, 5785, 5786, 5789, 5790, 5791, 5793, 5794, 5795, 5796, 5802, 5803, 5804, 5805, 5806, 5807, 5808, 5809, 5810, 5811, 5812, 5813, 5814, 5815, 5816, 5817, 5818, 5819, 5820, 5821, 5822, 5823, 5824, 5825, 5826, 5827, 5828, 5829, 5830, 5831, 5832, 5833, 5834, 5835, 5836, 5837, 5838, 5844, 5845, 5846, 5847, 5848, 5849, 5850, 5851, 5852, 5853, 5854, 5855, 5856, 5857, 5858, 5859, 5860, 5861, 5862, 5863, 5864, 5865, 5866, 5867, 5868, 5869, 5870, 5871, 5872, 5873, 5874, 5875, 5876]
                    whole_palm_idx_left = [2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3049, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3100, 5887, 5888, 5889, 5922, 5923, 5924, 5925, 5926, 5927, 5928, 5929, 5930, 5931, 5932, 5934, 5936, 5937, 5938, 5939, 5940, 5941, 5942, 5943, 5944, 5945, 5946, 5947, 5948, 5949, 5950, 5951, 5952, 5953, 5954, 5955, 5956, 5957, 5959, 5960, 5961, 5962, 5963, 5964, 5965, 5966, 5967, 5968, 5969, 5970, 5971, 5972, 5973, 5974, 5980, 5981, 5982, 5983, 5984, 5985, 5986, 5987, 5988, 5989, 5990, 5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999, 6000, 6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009, 6010, 6011]
                else:
                    whole_palm_idx = [4126, 4128, 4146, 4147, 4163, 4164, 4165, 4166, 4179, 4180, 4181, 4182, 4184, 4185, 4187, 4190, 4191, 4192, 4193, 4194, 4210, 4217, 4218, 4219, 4221, 4249, 4250, 4252, 4254, 4258, 4263, 4264, 4265, 4266, 4267, 4268, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4322, 4325, 4326, 4327, 4358, 4368, 4369, 4384, 4398, 4399, 4402, 4403, 4404, 4407, 4411, 4414, 4417, 4425, 4455, 4456, 4457, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4469, 4470, 4477, 4478, 4479, 4480, 4487, 4490, 4491, 4492, 4493, 4494, 4495, 4496, 4497, 4500, 4506, 4507, 4509, 4510, 4511, 4512, 4514, 4515, 4516, 4520, 4523, 4524, 4527, 4528, 4533, 4534, 4535, 4538, 4539, 4540, 4546, 4547, 4550, 4555, 4556, 4565, 4566, 4568, 4569, 4570, 4571, 4572, 4573, 4574, 4575, 4576, 4577, 4578, 4579, 4580, 4581, 4582, 4588, 4589, 4590, 4591, 4592, 4593, 4599, 4601, 4603, 4604, 4605, 4606, 4607, 4608, 4609, 4612, 4619, 4620, 4621, 4624, 4625, 4626, 4633, 4634, 4643, 4644, 4645, 4649, 4650, 4651, 4657, 4658, 4661, 4666, 4676, 4679, 4680, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689, 4690, 4691, 4692, 4700, 4701, 4702, 4703, 4710, 4714, 4715, 4716, 4717, 4719, 4720, 4731, 4732, 4734, 4737, 4738, 4739, 4740, 4741, 4742, 4743, 4744, 4747, 4748, 4751, 4752, 4755, 4756, 4758, 4759, 4761, 4762, 4767, 4768, 4776, 4777, 4778, 4779, 4793, 4799, 4802, 4803, 4804, 4807, 4808, 4809, 4812, 4813, 4814, 4816, 4817, 4818, 4819, 4820, 4821, 4822, 4823, 4824, 4825, 4826, 4827, 4828, 4829, 4830, 4831, 4832, 4834, 4836, 4837, 4841, 4848, 4850, 4851, 4856, 4857, 4866, 4867, 4869, 4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878, 4880, 4882, 4890, 4891, 4892, 4893, 4894, 4895, 4897, 4898, 4899, 4900, 4901, 4902, 4903, 4904, 4905, 4906, 4907, 4908, 4912, 4913]
                    whole_palm_idx_left = [1404, 1405, 1407, 1410, 1424, 1441, 1442, 1443, 1444, 1456, 1457, 1458, 1459, 1460, 1462, 1469, 1470, 1471, 1472, 1488, 1527, 1531, 1532, 1533, 1536, 1541, 1542, 1543, 1544, 1545, 1546, 1550, 1552, 1553, 1554, 1555, 1556, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1581, 1600, 1601, 1602, 1603, 1604, 1605, 1629, 1630, 1636, 1646, 1647, 1659, 1660, 1661, 1676, 1677, 1681, 1682, 1685, 1689, 1694, 1695, 1703, 1708, 1713, 1715, 1716, 1721, 1722, 1723, 1733, 1734, 1735, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1772, 1773, 1774, 1775, 1777, 1778, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1800, 1802, 1805, 1806, 1811, 1812, 1813, 1815, 1816, 1817, 1818, 1823, 1824, 1825, 1827, 1828, 1833, 1834, 1836, 1843, 1844, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1867, 1868, 1869, 1870, 1871, 1872, 1874, 1875, 1877, 1878, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1890, 1893, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1908, 1910, 1911, 1912, 1915, 1916, 1919, 1921, 1922, 1923, 1925, 1927, 1928, 1929, 1935, 1936, 1938, 1939, 1944, 1945, 1954, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1970, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1998, 1999, 2002, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2025, 2026, 2029, 2033, 2034, 2035, 2036, 2037, 2039, 2040, 2041, 2044, 2045, 2046, 2052, 2053, 2054, 2055, 2056, 2057, 2074, 2075, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2087, 2091, 2092, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2114, 2115, 2116, 2117, 2118, 2124, 2125, 2126, 2133, 2134, 2143, 2144, 2145, 2147, 2148, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2193]

                    # WRT to full/original/default body, after mapping
                    whole_palm_idx = [4500, 4502, 4520, 4521, 4537, 4538, 4539, 4540, 4553, 4554, 4555, 4556, 4558, 4559, 4561, 4564, 4565, 4566, 4567, 4568, 4584, 4591, 4592, 4593, 4595, 4623, 4624, 4626, 4628, 4632, 4637, 4638, 4639, 4640, 4641, 4642, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4666, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4696, 4699, 4700, 4701, 4732, 4742, 4743, 4758, 4772, 4773, 4776, 4777, 4778, 4781, 4785, 4788, 4791, 4799, 4829, 4830, 4831, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4843, 4844, 4851, 4852, 4853, 4854, 4861, 4864, 4865, 4866, 4867, 4868, 4869, 4870, 4871, 4874, 4880, 4881, 4883, 4884, 4885, 4886, 4888, 4889, 4890, 4894, 4897, 4898, 4901, 4902, 4907, 4908, 4909, 4912, 4913, 4914, 4920, 4921, 4924, 4929, 4930, 4939, 4940, 4942, 4943, 4944, 4945, 4946, 4947, 4948, 4949, 4950, 4951, 4952, 4953, 4954, 4955, 4956, 4962, 4963, 4964, 4965, 4966, 4967, 4973, 4975, 4977, 4978, 4979, 4980, 4981, 4982, 4983, 4986, 4993, 4994, 4995, 4998, 4999, 5000, 5007, 5008, 5017, 5018, 5019, 5023, 5024, 5025, 5031, 5032, 5035, 5040, 5050, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5061, 5062, 5063, 5064, 5065, 5066, 5074, 5075, 5076, 5077, 5084, 5088, 5089, 5090, 5091, 5093, 5094, 5105, 5106, 5108, 5111, 5112, 5113, 5114, 5115, 5116, 5117, 5118, 5121, 5122, 5125, 5126, 5129, 5130, 5132, 5133, 5135, 5136, 5141, 5142, 5150, 5151, 5152, 5153, 5167, 5173, 5176, 5177, 5178, 5181, 5182, 5183, 5186, 5187, 5188, 5190, 5191, 5192, 5193, 5194, 5195, 5196, 5197, 5198, 5199, 5200, 5201, 5202, 5203, 5204, 5205, 5206, 5208, 5210, 5211, 5215, 5222, 5224, 5225, 5230, 5231, 5240, 5241, 5243, 5244, 5245, 5246, 5247, 5248, 5249, 5250, 5251, 5252, 5254, 5256, 5264, 5265, 5266, 5267, 5268, 5269, 5271, 5272, 5273, 5274, 5275, 5276, 5277, 5278, 5279, 5280, 5281, 5282, 5286, 5287]
                    whole_palm_idx_left = [1773, 1774, 1776, 1779, 1793, 1810, 1811, 1812, 1813, 1825, 1826, 1827, 1828, 1829, 1831, 1838, 1839, 1840, 1841, 1857, 1896, 1900, 1901, 1902, 1905, 1910, 1911, 1912, 1913, 1914, 1915, 1919, 1921, 1922, 1923, 1924, 1925, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1950, 1969, 1970, 1971, 1972, 1973, 1974, 1998, 1999, 2005, 2015, 2016, 2028, 2029, 2030, 2045, 2046, 2050, 2051, 2054, 2058, 2063, 2064, 2072, 2077, 2082, 2084, 2085, 2090, 2091, 2092, 2102, 2103, 2104, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2141, 2142, 2143, 2144, 2146, 2147, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2169, 2171, 2174, 2175, 2180, 2181, 2182, 2184, 2185, 2186, 2187, 2192, 2193, 2194, 2196, 2197, 2202, 2203, 2205, 2212, 2213, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2236, 2237, 2238, 2239, 2240, 2241, 2243, 2244, 2246, 2247, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2259, 2262, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2277, 2279, 2280, 2281, 2284, 2285, 2288, 2290, 2291, 2292, 2294, 2296, 2297, 2298, 2304, 2305, 2307, 2308, 2313, 2314, 2323, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2339, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2367, 2368, 2371, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2394, 2395, 2398, 2402, 2403, 2404, 2405, 2406, 2408, 2409, 2410, 2413, 2414, 2415, 2421, 2422, 2423, 2424, 2425, 2426, 2443, 2444, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2456, 2460, 2461, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2483, 2484, 2485, 2486, 2487, 2493, 2494, 2495, 2502, 2503, 2512, 2513, 2514, 2516, 2517, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2562]


                contact_loss_2, _, _, _ = distChamfer(body_model_output.vertices[fid, whole_palm_idx][None], ob_transed_all[fid][obj_point][None])                     
                contact_loss_2_left, _, _, _ = distChamfer(body_model_output.vertices[fid, whole_palm_idx_left][None], ob_transed_all[fid][obj_point][None])                     



                if pc[0]['mesh_path'].endswith('03.obj') or pc[0]['mesh_path'].endswith('07.obj'):
                    contact_loss_2 = torch.min(contact_loss_2)
                    contact_loss_2_left = torch.min(contact_loss_2_left)


                if pc[0]['mesh_path'].endswith('09.obj'):
                    if contact_loss_2.mean() > contact_loss_2_left.mean():
                        total_loss += contact_loss_2_left.mean() * 9e9/ total
                    else:
                        total_loss += contact_loss_2.mean() * 9e9 / total
                elif pc[0]['mesh_path'].endswith('08.obj'):
                    if contact_loss_2.mean() > contact_loss_2_left.mean():
                        total_loss += contact_loss_2_left.mean() * 9e6/ total
                    else:
                        total_loss += contact_loss_2.mean() * 9e6 / total
                elif pc[0]['mesh_path'].endswith('07.obj') or pc[0]['mesh_path'].endswith('10.obj'):
                    if contact_loss_2.mean() > contact_loss_2_left.mean():
                        total_loss += contact_loss_2_left.mean() * 5e5 / total
                    else:
                        total_loss += contact_loss_2.mean() * 5e5 / total
                else:
                    if contact_loss_2.mean() > contact_loss_2_left.mean():
                        total_loss += contact_loss_2_left.mean() * 9e7/ total
                    else:
                        total_loss += contact_loss_2.mean() * 9e7 / total


        # Plane intepenetraion loss
        loss_plane_body = torch.mm(body_model_output.vertices.view([-1, 3]) - POINT, NORMAL)
        loss_foot_contact = torch.pow(torch.min(loss_plane_body.reshape([-1, 6117]), axis=1)[0], 2) * 1e5
        loss_foot_contact = torch.sum( loss_foot_contact )

        loss_plane_body = torch.sum( torch.pow(loss_plane_body[loss_plane_body<0.], 2) ) * 1e9

        loss_plane_object = torch.mm(ob_transed_all.view([-1, 3]) - POINT, NORMAL)
        loss_plane_object = torch.sum( torch.pow(loss_plane_object[loss_plane_object<0.], 2) ) * 1e9

        total_loss = total_loss + loss_plane_body/1e3 + loss_plane_object / 1e1
        total_loss = total_loss + loss_foot_contact

        loss_plane_body = 0.0



        print('total:{:.2f}, joint_loss:{:0.2f},  joint_loss_2:{:0.2f}, joint_loss_3:{:0.2f}, joint_loss_4:{:0.2f}, joint_loss_5:{:0.2f}, joint_loss_6:{:0.2f}, loss_s2m:{:0.2f}, loss_m2s:{:0.2f},  loss_smoothness:{:0.2f}, penetration:{:0.2f}, contact:{:0.2f}, plane_body:{:0.2f}'.format(total_loss.item(), joint_loss.item(), joint_loss_2.item(),joint_loss_3.item(), joint_loss_4.item(), joint_loss_5.item(), joint_loss_6.item(), s2m_dist, m2s_dist, loss_acceleration, torch.tensor(sdf_penetration_loss).item(), torch.tensor(contact_loss).item(), torch.tensor(loss_plane_body).item()))

        return total_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2,
                 camera_mode='moving',
                 dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype
        self.camera_mode = camera_mode

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, body_model,
                **kwargs):

        projected_joints = camera[0](body_model_output.joints)
        projected_joints_2 = camera[1](body_model_output.joints)
        projected_joints_3 = camera[2](body_model_output.joints)
        projected_joints_4 = camera[3](body_model_output.joints)
        projected_joints_5 = camera[4](body_model_output.joints)
        projected_joints_6 = camera[5](body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints[0], 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        joint_error_2 = torch.pow(
            torch.index_select(gt_joints[1], 1, self.init_joints_idxs) -
            torch.index_select(projected_joints_2, 1, self.init_joints_idxs),
            2)
        joint_loss_2 = torch.sum(joint_error_2) * self.data_weight ** 2

        joint_error_3 = torch.pow(
            torch.index_select(gt_joints[2], 1, self.init_joints_idxs) -
            torch.index_select(projected_joints_3, 1, self.init_joints_idxs),
            2)
        joint_loss_3 = torch.sum(joint_error_3) * self.data_weight ** 2

        joint_error_4 = torch.pow(
            torch.index_select(gt_joints[3], 1, self.init_joints_idxs) -
            torch.index_select(projected_joints_4, 1, self.init_joints_idxs),
            2)
        joint_loss_4 = torch.sum(joint_error_4) * self.data_weight ** 2


        joint_error_5 = torch.pow(
            torch.index_select(gt_joints[4], 1, self.init_joints_idxs) -
            torch.index_select(projected_joints_5, 1, self.init_joints_idxs),
            2)
        joint_loss_5 = torch.sum(joint_error_5) * self.data_weight ** 2

        joint_error_6 = torch.pow(
            torch.index_select(gt_joints[5], 1, self.init_joints_idxs) -
            torch.index_select(projected_joints_6, 1, self.init_joints_idxs),
            2)
        joint_loss_6 = torch.sum(joint_error_6) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            if self.camera_mode == 'moving':
                depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                     camera.translation[:,2] - self.trans_estimation[:, 2]).pow(2))

            elif self.camera_mode == 'fixed':
                depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                    body_model.transl[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + joint_loss_2 + joint_loss_3 + joint_loss_4 + joint_loss_5 + joint_loss_6 
