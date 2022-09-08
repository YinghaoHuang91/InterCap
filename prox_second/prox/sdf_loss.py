import torch
import torch.nn as nn
import numpy as np

from sdf import SDF

class SDFLoss(nn.Module):

    def __init__(self, faces_body, faces_object, grid_size=32, robustifier=None, debugging=False):
        super(SDFLoss, self).__init__()
        self.sdf = SDF()
        self.register_buffer('faces_body', torch.tensor(faces_body.astype(np.int32)))
        self.register_buffer('faces_object', torch.tensor(faces_object.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier
        self.debugging = debugging

    @torch.no_grad()
    def get_bounding_boxes(self, vertices_body, vertices_object):
        num_people = 2
        boxes = torch.zeros(num_people, 2, 3, device=vertices_body.device)
        boxes[0, 0, :] = vertices_body.min(dim=0)[0]
        boxes[1, 1, :] = vertices_object.max(dim=0)[0]

        return boxes

    @torch.no_grad()
    def check_overlap(self, bbox1, bbox2):
        # check x
        if bbox1[0,0] > bbox2[1,0] or bbox2[0,0] > bbox1[1,0]:
            return False
        #check y
        if bbox1[0,1] > bbox2[1,1] or bbox2[0,1] > bbox1[1,1]:
            return False
        #check z
        if bbox1[0,2] > bbox2[1,2] or bbox2[0,2] > bbox1[1,2]:
            return False
        return True

    def filter_isolated_boxes(self, boxes):

        num_people = boxes.shape[0]
        isolated = torch.zeros(num_people, device=boxes.device, dtype=torch.uint8)
        for i in range(num_people):
            isolated_i = False
            for j in range(num_people):
                if j != i:
                    isolated_i |= not self.check_overlap(boxes[i], boxes[j])
            isolated[i] = isolated_i
        return isolated

    def forward(self, vertices_body, vertices_object, scale_factor=0.2):
        num_people = 2
        loss = torch.tensor(0., device=vertices_body.device)
        boxes = self.get_bounding_boxes(vertices_body, vertices_object)
        overlapping_boxes = ~self.filter_isolated_boxes(boxes)

        # If no overlapping voxels return 0
        if overlapping_boxes.sum() == 0:
            return loss

        # Filter out the isolated boxes
        #boxes = boxes[overlapping_boxes]

        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
        boxes_scale = (1+scale_factor) * 0.5*(boxes[:,1] - boxes[:,0]).max(dim=-1)[0][:,None,None]

        with torch.no_grad():
            vertices_body_centered = vertices_body - boxes_center[0]
            vertices_object_centered = vertices_object - boxes_center[1]

            vertices_body_scaled = vertices_body_centered / boxes_scale[0]
            vertices_object_scaled = vertices_object_centered / boxes_scale[1]

            assert(vertices_body_scaled.min() >= -1)
            assert(vertices_object_scaled.max() <= 1)

            #phi_body = self.sdf(self.faces_body, vertices_body_scaled)
            self.faces_object = self.faces_object.repeat(2, 1, 1)
            vertices_object_scaled = vertices_object_scaled[None, :, :].repeat(2, 1, 1)

            phi_object = self.sdf(self.faces_object, vertices_object_scaled)

            #assert(phi_body.min() >= 0)
            assert(phi_object.min() >= 0)


        valid_people = 2
        weights = torch.ones(valid_people, 1, device=vertices.device)
        weights[1,0] = 0.
        # Change coordinate system to local coordinate system of each person
        vertices_local = (vertices_object - boxes_center[1].unsqueeze(dim=0)) / boxes_scale[1].unsqueeze(dim=0)
        vertices_grid = vertices_local.view(1,-1,1,1,3)
        # Sample from the phi grid
        phi_val = nn.functional.grid_sample(phi_object[None, None], vertices_grid).view(valid_people, -1)
        # ignore the phi values for the i-th shape
        cur_loss = weights[1] * phi_val
        if self.debugging:
            import ipdb;ipdb.set_trace()
        # robustifier
        if self.robustifier:
            frac = (cur_loss / self.robustifier) ** 2
            cur_loss = frac / (frac + 1)
        loss += cur_loss.sum() / valid_people ** 2

        return loss
