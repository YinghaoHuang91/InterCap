import numpy as np
import glob
import os
import json
import open3d as o3d
import pickle as pkl

# TODO: point to RICH path
VAL_PATH = './val'

sub_folders = glob.glob(os.path.join(VAL_PATH, '*'))
sub_folders = sorted(sub_folders)

# Don't consider contact between feet and floor
def get_idxs():
    mesh = o3d.io.read_triangle_mesh('./smpl.ply')
    colors = np.asarray(mesh.vertex_colors)

    res = []
    for i in range(6890):
        c = colors[i]
        if np.array_equal(c, np.array([0., 0., 0.])):
            continue
        else:
            res.append(i)

    return res

IDXs = get_idxs()

# Get 2D keypoint detection result
def read_dets(det_folder):
    jsons = glob.glob( os.path.join(det_folder, '*_*/*json') )
    jsons = sorted(jsons)

    res = []
    res_idx = []
    res_valid = []
    for j in jsons:

        try:
            with open(j) as fin:
                data = json.load(fin)

            human = np.zeros([530, 730])
            obj = np.zeros([530, 730])

            for d in data:
                m = np.array(d['mask'])
                if 'person' in d['class']:
                    human = human + m
                else:
                    obj = obj + m
            
            obj = obj > 0
            human = human > 0

            flag =  sum((obj * human).flatten()) > 10
            res.append(flag)

            idx = int( j.split('/')[-2].split('_')[0] )
            res_idx.append(idx)

            if sum(obj.flatten()) > 10:
                res_valid.append(1)
            else:
                res_valid.append(0)
        except:
            res_valid.append(0)
            continue

    return res, res_idx, res_valid

# Get ground-truth annotation of RICH
def get_gt( folder_name ):

    # TODO: point to ground-truth of RICH
    pkl_file = os.path.join('./val_hsc/', folder_name)
    #pkl_files = glob.glob(os.path.join(pkl_file, '*/016.pkl'))
    pkl_files = glob.glob(os.path.join(pkl_file, '*/*.pkl'))
    pkl_files = sorted( pkl_files )

    res = []
    res_idx = []
    for p in pkl_files:
        try:
            with open(p, 'rb') as fin:
                data = pkl.load(fin)

            if sum(data['contact'][IDXs]) > 10:
                res.append(True)
            else:
                res.append(False)

            tmp = int(p.split('/')[-2])
            res_idx.append( tmp )
        except:
            continue

    return np.array(res), res_idx


# Compute contact label
def compute_contact(folder_path):
    cams = glob.glob( os.path.join(folder_path, 'cam*') )
    cams = [c for c in cams if 'det' in c]

    dets = []
    dets_idx = []
    res_valid_ = []

    for c in cams:
        tmp, idx, res_valid = read_dets(c)
        dets.append(tmp)
        dets_idx.append(idx)
        res_valid_.append(res_valid)
        print('Done %s' % c)

    dets = np.asarray(dets)
    if type(dets) is list:
        return None, None

    try:
        dets = np.sum(dets, axis=0) >= 2
        res_valid_ = np.sum(res_valid_, axis=0) >= 2
    except:
        return None, None

    gt, gt_idx = get_gt( os.path.basename(folder_path) )

    res = []

    if False:
        res = []
        for i, k in enumerate(dets_idx[0]):
            if k in gt_idx:
                j = gt_idx.index(k)

                if dets[i] == gt[j]:
                    res.append(1)
                else:
                    res.append(0)
    else:
        total = min(len(dets), len(gt))
        for i in range(total):
            if dets[i] == gt[i]:
                res.append(1)
            else:
                res.append(0)

    return res, res_valid_

res = {}
res_valid_ = {}
for sf in sub_folders:
   tmp, res_valid = compute_contact(sf)

   if tmp is None:
       continue
   res['%s' % os.path.basename(sf)] =  tmp
   res_valid_['%s' % os.path.basename(sf)] =  res_valid

with open('res.pkl', 'wb') as fout:
    pkl.dump(res, fout)
with open('res_valid.pkl', 'wb') as fout:
    pkl.dump(res_valid_, fout)
