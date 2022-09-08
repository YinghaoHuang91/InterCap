import numpy as np
import cv2
import os
import torch
import trimesh
import torch
import torch.nn as nn
import open3d as o3d
#from utils import rodrigues, create_renderers, load_targets, GMoF
from utils import create_renderers, load_targets, GMoF
from pytorch3d.transforms import axis_angle_to_matrix

import neural_renderer as nr
from torch.nn.functional import upsample
from pytorch3d.loss import chamfer_distance

import argparse
from configparser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('path', action='store', type=str, help='The text to parse.')
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

POINT = np.array([-0.07362306, 1.06334016, 2.88126607])
NORMAL = np.array([-0.02645398, 0.98627183, 0.16299712]) * -1


POINT = torch.from_numpy(POINT).to(device)
NORMAL = torch.from_numpy(NORMAL).to(device)

VIEW_SIZE = 6


#configParser = ConfigParser.RawConfigParser()   
config = ConfigParser()
config.read( os.path.join(args.path, 'config.ini') )

OBJ_MAJOR_PATH = './objs'
         #0   1           2        3         4   5           6         7, 8          9        10,       11     12  13 
TYPES = ['', 'suitcase', 'skate', 'sports', '', 'umbrella', 'tennis', '', 'suitcase', 'chair', 'bottle', 'cup', '', 'chair']
#IDX = int(config.get('mesh' , 'mesh_path'))
IDX = int( args.path.split('_')[1][6:] )

mesh_path = os.path.join(OBJ_MAJOR_PATH, '%02d' % IDX + '.ply')
mesh_type = TYPES[ IDX ]
mesh = trimesh.load_mesh(mesh_path)

# Capure name
capture_name = args.path
res_path = os.path.join(capture_name, 'Res')
if not os.path.exists( res_path ):
    os.mkdir( res_path )

# Starting frame
start_idx = int( config.get('init_frame_id' , 'id') )

init_rot_config =  config.get('init_rot' , 'rot') 
init_trans_config =  config.get('init_trans' , 'trans') 
init_rot_config = [float(i) for i in  init_rot_config.split(' ')]
init_trans_config = [float(i) for i in  init_trans_config.split(' ')]

global_idx = 0
global_step = 0

vertices = mesh.vertices.astype(np.float32)
vertices = torch.from_numpy(vertices).to(device)
faces = torch.from_numpy(mesh.faces).to(device)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        init_rot = np.array( init_rot_config, dtype=np.float32)
        init_rot = torch.from_numpy(init_rot)

        init_trans = np.array(init_trans_config, dtype=np.float32)
        init_trans = torch.from_numpy(init_trans)


        self.global_rot = nn.Parameter(init_rot, requires_grad=True)
        self.global_trans = nn.Parameter(init_trans, requires_grad=True)

        self.vertices = vertices[None, :, :]
        self.vertices_final = vertices[None, :, :]
        self.faces = faces[None, :, :]

        self.renderer, self.func_renderer_dyna = create_renderers(capture_name)

    def load_target(self, idx):
        targets, bbs = load_targets(idx, capture_name, mesh_type)

        for i in range( int( len(targets) / 2 ) ):
            if len( targets[i].shape ) < 3:
                targets[i] = targets[i][None]
    
        global_mat = axis_angle_to_matrix(self.global_rot.reshape([1, 3]))

        vertices_final = torch.matmul( global_mat.squeeze(),self.vertices.T[:, :, 0]).T + self.global_trans
        self.vertices_final = vertices_final

        np_imgs = []

        for i in range( len(self.renderer) ):
            image = self.renderer[i](vertices_final[None, :, :], self.faces, mode='silhouettes')            
            image = image.detach().cpu().numpy()[0]
            image = cv2.resize(image, (1920, 1080))

            if idx == 0:
                #kernel = np.ones((20,20),np.uint8)
                kernel = np.ones((5,5),np.uint8)
                image = cv2.dilate(image, kernel, iterations=5)

            image = (image > 0).astype(np.float32)

            regions = []
            for m in targets[i]:
                if idx > 60:
                    tmp = np.sum( image * m )
                else:
                    tmp = np.sum( m )
                regions.append(tmp)

            print('Count regions')
            print(regions)
            print()

            if len(regions) > 0:
                max_index = regions.index(np.max(regions))
                tmp = targets[i][max_index].astype(np.float32)
                bbs[i] = bbs[i][max_index]
            else:
                tmp = np.zeros([1080, 1920], dtype=np.float32)

            #tmp = cv2.resize(tmp, (1920, 1920))
            targets[i] = torch.from_numpy(tmp)

        # Remove background depth
        for i in range( len(self.renderer) ):
            targets[i + len(self.renderer)][ targets[i] == 0 ] = 0


        self.image_ref = targets
        self.image_ref_crop = [None] * len(targets)

        self.bbs = bbs
        self.img_sizes = []
        for b in self.bbs:
            S = np.max([b[3]-b[1], b[2]-b[0]])
            self.img_sizes.append( S )

        # Create dyna renderer        
        self.renderer_dyna = self.func_renderer_dyna(self.bbs)


    def forward(self):
        global global_idx
        global global_step

        #if global_idx > 60:
        global_mat = axis_angle_to_matrix(self.global_rot.reshape([1, 3]))

        vertices_final = torch.matmul( global_mat.squeeze(),self.vertices.T[:, :, 0]).T + self.global_trans
        self.vertices_final = vertices_final


        loss = []
        for i in range( len(self.renderer_dyna) ):
            if ( torch.sum(self.image_ref[i].flatten()) < 20 ):
                continue


            image = self.renderer_dyna[i](vertices_final[None, :, :], self.faces, mode='silhouettes')            
            image_depth = self.renderer_dyna[i](vertices_final[None, :, :], self.faces, mode='depth')
            image_depth[image_depth == 100.] = 0.

            img_ref = self.image_ref[i][self.bbs[i][1]:self.bbs[i][1] + self.img_sizes[i], self.bbs[i][0]:self.bbs[i][0] + self.img_sizes[i]].to(device)
            depth_ref = self.image_ref[i+VIEW_SIZE][self.bbs[i][1]: self.bbs[i][1] + self.img_sizes[i] , self.bbs[i][0]:self.bbs[i][0] + self.img_sizes[i]].to(device)
            H, W = img_ref.shape[:2]

            # Only save cropped regions
            self.image_ref_crop[i] = self.image_ref[i][self.bbs[i][1]:self.bbs[i][1] + self.img_sizes[i], self.bbs[i][0]:self.bbs[i][0] + self.img_sizes[i]]
            self.image_ref_crop[i+VIEW_SIZE] = self.image_ref[i+VIEW_SIZE][self.bbs[i][1]: self.bbs[i][1] + self.img_sizes[i] , self.bbs[i][0]:self.bbs[i][0] + self.img_sizes[i]]

            # To multiply by mask or not
            tmp = torch.sum( ( ( image.squeeze()[:H, :W] * img_ref - img_ref ))  ** 2 )


            loss.append(tmp)
            loss.append(tmp_depth)

        # Above plane loss
        if False:
            gap = vertices_final - POINT
            dis = torch.mm(gap, NORMAL.view(3, -1))
            loss_plane = torch.sum( torch.pow(dis[dis<0.], 2) ) * 1e5

        return loss



model = Model()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.Adam( iter([model.global_trans]), lr=1.0e-3)


#optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100)

#total = 1000
total = 5000
#total = 250

res = []

#for fid in range(60, total):
#for fid in range(164, total):
for fid in range(start_idx, total):
#for fid in range(140, 600):
    fid = fid * 1 

    try:
        model.load_target(fid)
    except Exception as e:
        print(e)
        print('Error!')
        break
    model.cuda()

    print("Frame %d" % fid)
    print(model.global_trans.detach().cpu().numpy())
    print(model.global_rot.detach().cpu().numpy())


    for i in range(150 if fid > start_idx else 400):
        loss = model()        
        #loss = [loss[4]]

        # No masks
        if len(loss) == 0:
            print('Zeor losses')
            break

        final_loss = torch.sum(torch.stack(loss))
        


        global_step = i

        if i % 50 == 0:
            print('Len of loss %d, Step %d ' % (len(loss), i))

            print(final_loss.detach().cpu().numpy())
            for k in loss:
                print( k.detach().cpu().numpy() )
            print()


        optimizer.zero_grad()

        final_loss.backward(retain_graph=True)
        optimizer.step()


    if True:
        res_ = {'pose': model.global_rot.detach().cpu().numpy(), 'trans': model.global_trans.detach().cpu().numpy(), 'segs': [j.detach().cpu().numpy() for j in model.image_ref], 'bbs': model.bbs,'img_sizes': model.img_sizes}

        if True and (fid-start_idx) % 10 == 0:
            for kk in range(len(res_['segs'][:5])):
                break
                import os
                if not os.path.exists('Object_%d' % kk):
                    os.mkdir('Object_%d' % kk)
                if not os.path.exists('Vis_%d' % kk):
                    os.mkdir('Vis_%d' % kk)

                obj_silhouette = cv2.resize(res_['segs'][kk], (1920, 1080))
                obj_silhouette[obj_silhouette > 0] = 1
                cv2.imwrite('./Object_%d/%05d.png' % (kk, fid), obj_silhouette)
                cv2.imwrite('./Vis_%d/%05d.png' % (kk, fid), (obj_silhouette*255).astype(np.uint8) )

            for vid in range(VIEW_SIZE):
                tmp1 = model.renderer[vid](model.vertices_final[None, :, :], model.faces, mode='silhouettes') * 255
                tmp1 = tmp1.detach().cpu().numpy().squeeze()
                tmp1 = cv2.resize(tmp1, (1920, 1080))

                tmp2 = model.image_ref[vid].squeeze() * 255
                tmp2 = tmp2.detach().cpu().numpy()

                combined = np.concatenate((tmp1, tmp2), axis=1)

                cv2.imwrite('./%s/%05d_ref_%05d.png' % (res_path, fid, vid), combined)

        res.append( {'pose': model.global_rot.detach().cpu().numpy(), 'trans': model.global_trans.detach().cpu().numpy(), 'segs': [j.detach().cpu().numpy() if j != None else j for j in model.image_ref_crop], 'bbs': model.bbs[:], 'img_sizes': model.img_sizes[:] } )

res[0]['mesh_path'] = mesh_path.replace('.ply', '.obj')

with open('./%s/res.pkl' % res_path, 'wb') as fout:
    import pickle as pkl
    pkl.dump(res, fout)
