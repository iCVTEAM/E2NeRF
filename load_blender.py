import os
import torch
import numpy as np
import imageio 
import json


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir):
    splits = ['train', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transform_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    imgs = []
    poses = []
    test_imgs = []
    test_poses = []
    for s in splits:
        meta = metas[s]

        if s == 'train':
            num = 0
            for frame in meta['frames'][0:200:2]:
                fname = os.path.join(basedir, "./train/r_{}.png".format(num))
                num = num + 2
                imgs.append(imageio.imread(fname))
                poses.append(np.array([frame['transform_matrix'][1], frame['transform_matrix'][5], frame['transform_matrix'][9], frame['transform_matrix'][13], frame['transform_matrix'][17]]))
            imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
        else:
            num = 0
            for frame in meta['frames'][0:200:1]:
                fname = os.path.join(basedir, "./test/r_{}.png".format(num))
                num = num + 1
                test_imgs.append(imageio.imread(fname))
                test_poses.append(np.array(frame['transform_matrix'][1]))


            test_imgs = (np.array(test_imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
            test_poses = np.array(test_poses).astype(np.float32)


    H, W = imgs[0].shape[:2]
    meta = metas['train']
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 180 + 1)[:-1]], 0)

    return imgs, poses, test_imgs, test_poses, render_poses, [H, W, focal]
