import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from bvh_distance_queries import BVH
from psbody.mesh.geometry.tri_normals import TriNormals, TriNormalsScaled

class SignedDistance(nn.Module):

    def __init__(self,
                 sort_points_by_morton: bool = True,
                 queue_size: int = 128) -> None:
        ''' Constructor for the point to mesh residual module

            Parameters
            ----------
                sort_points_by_morton: bool, optional
                    Sort input points by their morton code. Helps improve query
                    speed. Default is true
                queue_size: int, optional
                    The size of the data structure used to store intermediate
                    distance computations
        '''
        super(SignedDistance, self).__init__()
        self.search_tree = BVH(sort_points_by_morton=sort_points_by_morton,
                               queue_size=queue_size)

    def forward(self,
                triangles,
                face_normals,
                points
                ):
        ''' Forward pass of the search tree

            Parameters
            ----------
                triangles: torch.tensor
                    A BxFx3x3 PyTorch tensor that contains the triangle
                    locations.
                points: torch.tensor
                    A BxQx3 PyTorch tensor that contains the query point
                    locations.
            Returns
            -------
                signed_distances: Distance (not squared)
        '''
        output = self.search_tree(triangles, points)
        distances, _, closest_faces, closest_bcs = output

        closest_bcs = torch.clamp(closest_bcs, 0, 1)

        batch_size, num_triangles = triangles.shape[:2]
        num_points = points.shape[1]

        closest_faces_idxs = (
            torch.arange(
                0, batch_size, device=triangles.device, dtype=torch.long) *
            num_triangles
        ).view(batch_size, 1)

        closest_triangles = triangles.view(-1, 3, 3)[
            closest_faces_idxs + closest_faces].view(
                batch_size, num_points, 3, 3)
        closest_points = (
            closest_triangles[:, :, 0] *
            closest_bcs[:, :, 0].unsqueeze(dim=-1) +
            closest_triangles[:, :, 1] *
            closest_bcs[:, :, 1].unsqueeze(dim=-1) +
            closest_triangles[:, :, 2] *
            closest_bcs[:, :, 2].unsqueeze(dim=-1)
        )

        residual = closest_points - points
        # sign compute
        # import ipdb; ipdb.set_trace()

        closest_face_normals = face_normals[:,closest_faces.flatten(),:]
        norm = torch.norm(residual, 2, dim=-1, keepdim=True)
        norm[norm==0] = 1.0
        residual_norm = residual/norm
        dot_pr = (residual_norm * closest_face_normals).sum(dim=-1)

        sign = torch.ones_like(dot_pr)
        sign[0 < dot_pr] = -1

        distances = residual.pow(2).sum(dim=-1).sqrt()
        signed_distances = sign * distances

        return signed_distances, residual_norm, closest_points, closest_faces, closest_bcs

def compute_sdf(v, f, device, flag=0):

    dim = 96

    if flag == 1:
        dim = 256

    n_points_per_batch = 1000000
    badding_val = 0.5

    face_normals = torch.tensor(TriNormalsScaled(v, f), dtype=torch.float32, device=device).reshape(
        1, -1, 3)

    vertices = torch.tensor(v, dtype=torch.float32, device=device)
    faces = torch.tensor(f.astype(np.int64),
                         dtype=torch.long,
                         device=device)
    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0)

    bbox = np.array([[np.max(v[:, 0]), np.max(v[:, 1]), np.max(v[:, 2])],
                     [np.min(v[:, 0]), np.min(v[:, 1]), np.min(v[:, 2])]])

    xmin, ymin, zmin = np.amin(v, axis=0) - badding_val
    xmax, ymax, zmax = np.amax(v, axis=0) + badding_val
    voxel_size = (np.array([xmax, ymax, zmax]) - np.array([xmin, ymin, zmin])) / dim
    X, Y, Z = np.mgrid[xmin: xmax:complex(0, dim), ymin:ymax:complex(0, dim), zmin:zmax:complex(0, dim)]
    query_points_np = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    query_points = torch.tensor(query_points_np, dtype=torch.float32, device=device)

    query_points_batches = torch.split(query_points, n_points_per_batch)
    nn_semantics_list = []
    distances_list = []
    norm_list = []
    # todo: for debug only
    closest_faces_list = []
    for i in tqdm(range(len(query_points_batches))):
        # m = bvh_distance_queries.BVH()
        m = SignedDistance()
        torch.cuda.synchronize()
        signed_distances, residual_norm, closest_points, closest_faces, closest_bcs = m(triangles, face_normals,query_points_batches[i].unsqueeze(0))

        distances_list.append(signed_distances.detach().cpu().numpy().squeeze())
        norm_list.append(residual_norm.detach().cpu().numpy().squeeze())
        closest_faces_list.append(closest_faces.detach().cpu().numpy().squeeze())

    SDF = np.concatenate(distances_list).flatten().astype(np.float32)
    SDF = SDF.reshape([dim, dim, dim])

    normals = np.concatenate(norm_list).reshape(-1,3).astype(np.float32)
    normals = normals.reshape([dim, dim, dim, 3])

    sdf_dict = {'sdf':SDF, 'min':np.array([xmin, ymin, zmin]), 'max':np.array([xmax, ymax, zmax]), 'dim':dim, 'bbox':bbox.tolist(), 'voxel_size':voxel_size.tolist(), 'normals':normals}

    #return SDF, grid_min, grid_max, dim, bbox
    return sdf_dict
