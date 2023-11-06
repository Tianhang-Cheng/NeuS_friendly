import numpy as np
import os
import shutil
import torch
import imageio.v2 as imageio
import cv2 as cv
from scipy.spatial.transform import Rotation
from termcolor import colored
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
np.set_printoptions(suppress=True)

def load_K_Rt_from_P(P=None):
    raise ValueError("This function is deprecated. Use decomposeP instead.")
    P = P[:3]
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32) 
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def decomposeP(P):
    M = P[0:3,0:3]
    Q = np.eye(3)[::-1]
    P_b = Q @ M @ M.T @ Q
    K_h = Q @ np.linalg.cholesky(P_b) @ Q
    K = K_h / K_h[2,2]
    A = np.linalg.inv(K) @ M
    l = (1/np.linalg.det(A)) ** (1/3)
    R = l * A
    t = l * np.linalg.inv(K) @ P[0:3,3]
    w2c = np.concatenate([R, t[:, None]], axis=1)
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0) # [4, 4]
    K_out = np.eye(4)
    K_out[0:3,0:3] = K
    return K_out, w2c

def near_far_from_sphere( rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far
 

def gen_rays_at(H, W, intrinsic, c2w, inv_scale_mat, resolution_level=1):
    """
    Generate rays at world space from one camera.
    Args:
        pose: [4, 4] camera pose matrix from camera to world (c2w)
        intrinsic: [4, 4] camera intrinsic matrix
        inv_scale_mat: [4, 4] scale matrix from bbox to unit sphere
    """
    intrinsic = torch.from_numpy(intrinsic).float()
    intrinsic = torch.inverse(intrinsic)
    c2w = torch.from_numpy(c2w).float()
    inv_scale_mat = torch.from_numpy(inv_scale_mat).float()

    l = resolution_level
    tx = torch.linspace(0, W - 1, W // l)
    ty = torch.linspace(0, H - 1, H // l)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
    p = torch.matmul(intrinsic[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    rays_v = torch.matmul(c2w[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = (inv_scale_mat @ c2w)[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

def draw_p(x, c='blue'):
    points = x  
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[...,0],
             points[...,1],
             points[...,2],
            cmap='Accent', 
            c=c, 
            alpha=1,
            marker=".") 
    plt.title('Point Cloud') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def min_max_to_bbox(x_min, y_min, z_min, x_max, y_max, z_max):
    return np.array([[x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_min], [x_min, y_max, z_max],
                    [x_max, y_min, z_min], [x_max, y_min, z_max], [x_max, y_max, z_min], [x_max, y_max, z_max]])

def draw_bbox_3d(point_cloud, x_min, y_min, z_min, x_max, y_max, z_max, title=None, set_lim=True):
    """
    point_cloud: [N, 3]
    x_min, y_min, z_min, x_max, y_max, z_max: float
    """ 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 

    # 绘制点云
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o', label='Point Cloud')

    bbox_points = min_max_to_bbox(x_min, y_min, z_min, x_max, y_max, z_max)

    # 绘制3D边界框
    edges = [
        [bbox_points[0], bbox_points[1], bbox_points[3], bbox_points[2]],
        [bbox_points[4], bbox_points[5], bbox_points[7], bbox_points[6]],
        [bbox_points[0], bbox_points[1], bbox_points[5], bbox_points[4]],
        [bbox_points[2], bbox_points[3], bbox_points[7], bbox_points[6]],
        [bbox_points[0], bbox_points[2], bbox_points[6], bbox_points[4]],
        [bbox_points[1], bbox_points[3], bbox_points[7], bbox_points[5]]
    ] 
    ax.add_collection3d(Poly3DCollection(edges, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)) 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') 
    ax.legend()
    # 通过set_box_aspect函数设置坐标轴刻度一致
    ax.set_box_aspect([1, 1, 1])  # 参数为X, Y, Z刻度的缩放因子
    if title is not None:
        ax.title.set_text(title)
    if set_lim:
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(z_min, z_max)
    plt.show()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--colmap_txt_dir', type=str, default='E:/dataset/hotdog/colmap', help='colmap output dir, where "points3D.txt", "cameras.txt", "images.txt" are located')
    parser.add_argument('--raw_image_dir', type=str, default='E:/dataset/hotdog', help='where images are located (the images that are used to run colmap)')
    parser.add_argument('--raw_mask_dir', type=str, default=None, help='where masks are located (optional)')
    parser.add_argument('--output_dir', type=str, default='E:/code/NeuS/public_data/hotdog', help='processed output dir, where "cameras_sphere.npz" will be saved')
    parser.add_argument('--viz_bbox', action='store_true', help='visualize bbox')
    args = parser.parse_args()
 
    colmap_txt_dir = args.colmap_txt_dir
    raw_image_dir = args.raw_image_dir
    raw_mask_dir = args.raw_mask_dir
    output_dir = args.output_dir
    viz_bbox = args.viz_bbox
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    
    ########################################################################
    # manually filter points cloud to find interesting area
    ########################################################################
    
    # 1. save as .xyz file
    xyz_file = os.path.join(colmap_txt_dir, 'points3D.xyz')
    points_list = []
    with open(os.path.join(colmap_txt_dir, 'points3D.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines[3:]:
            info = line.split(' ')
            p3d = [float(info[1]), float(info[2]), float(info[3]), ] # 7 is error
            points_list.append(p3d)
    points_list = np.array(points_list)
    if not os.path.exists(xyz_file):
        with open(xyz_file, 'w') as f:
            # 写入点云的总点数
            f.write(str(len(points_list)) + '\n')
            # 逐行写入每个点的坐标
            for point in points_list:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        print(colored('"points3D.xyz" is created. Please clean the point cloud "points3D.xyz" by hand (etc. Meshlab) and run this code again', 'red'))
        exit()

    # 2. read .xyz file to get bbox
    points_list = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        points_list = []
        for line in lines[1:]:
            info = line.strip().split(' ')
            p3d = [float(info[0]), float(info[1]), float(info[2])]
            points_list.append(p3d)
    points_list = np.array(points_list)
    points_list
    x_min = np.min(points_list[:, 0])
    y_min = np.min(points_list[:, 1])
    z_min = np.min(points_list[:, 2])
    x_max = np.max(points_list[:, 0])
    y_max = np.max(points_list[:, 1])
    z_max = np.max(points_list[:, 2])
    print(colored(f"bbox: x_min={x_min}, y_min={y_min}, z_min={z_min}, x_max={x_max}, y_max={y_max}, z_max={z_max}", 'green'))
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
    print(colored(f"scene center: {center}", 'green'))
    radius = np.linalg.norm(points_list - center, ord=2, axis=-1).max()
    print(colored(f"scene max radius: {radius}", 'green'))

    if viz_bbox:
        draw_bbox_3d(points_list, x_min=x_min, y_min=y_min, z_min=z_min, x_max=x_max, y_max=y_max, z_max=z_max, title='Bbox in Colmap space')

    # 3. calculate the scale matrix
    pred_object_bbox_min = np.array([x_min, y_min, z_min]) # predicted bbox min
    pred_object_bbox_max = np.array([x_max, y_max, z_max]) # predicted bbox max
    target_object_bbox_min = np.array([-1.01, -1.01, -1.01]) # target bbox min
    target_object_bbox_max = np.array([ 1.01,  1.01,  1.01]) # target bbox max

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -center
    # radius = max(z_max-z_min, y_max-y_min, x_max-x_min)
    # 用相同的缩放因子缩放x、y和z坐标，简化计算
    inv_scale_matrix = np.diag([1.01/radius, 1.01/radius, 1.01/radius, 1.0]).astype(np.float32) # from bbox to unit sphere
    inv_scale_matrix = np.dot(inv_scale_matrix, translation_matrix)

    raw_bbox = min_max_to_bbox(*pred_object_bbox_min, *pred_object_bbox_max)
    target_bbox = min_max_to_bbox(*target_object_bbox_min, *target_object_bbox_max)

    points_list_ = np.concatenate([points_list, np.ones((len(points_list), 1))], axis=1)
    if viz_bbox:
        draw_bbox_3d((inv_scale_matrix @ points_list_.T).T[:, :3], *target_object_bbox_min, *target_object_bbox_max, title='Bbox in unit sphere')
    
    ########################################################################
    # save intrinsics and camera poses into "cameras_sphere.npz"
    ########################################################################

    # 1. read intrinsics
    print(colored('Here we use "PINHOLE" model. You can change to other models.', 'green'))
    f_intrinsic  = open(os.path.join(colmap_txt_dir, 'cameras.txt'), 'r')
    lines_intrinsic = f_intrinsic.readlines()
    f_intrinsic.close()
        
    print(colored(lines_intrinsic[2], 'green'))
    n_cam = int(float(lines_intrinsic[2].split(' ')[-1]))
    assert len(lines_intrinsic[3:]) == n_cam, 'number of cameras is not equal to the number of lines'

    intrinsics = np.zeros((n_cam, 4, 4))
    valid_camera_id_to_index = {}
    for i, line in enumerate(lines_intrinsic[3:]):
        line = line.strip().split(' ')
        camera_id = int(line[0])
        valid_camera_id_to_index[camera_id] = i
        if line[1] == 'PINHOLE':
            fx = float(line[4])
            fy = float(line[5])
            cx = float(line[6])
            cy = float(line[7])
            intrinsics[i,0,0] = fx
            intrinsics[i,1,1] = fy
            intrinsics[i,0,2] = cx
            intrinsics[i,1,2] = cy
            intrinsics[i,2,2] = 1
            intrinsics[i,3,3] = 1
        else:
            raise NotImplementedError
    
    # 2.read camera poses (world to camera)
    # we should sort the camera poses by camera_id!
    f_pose = open(os.path.join(colmap_txt_dir, 'images.txt'), 'r')
    lines_pose = f_pose.readlines()
    f_pose.close()
 
    camera_list = np.zeros([n_cam, 7])
    for line in lines_pose[4::2]:
        info = line.strip().split(' ')
        
        camera_id = int(info[-2])
        mapped_index = valid_camera_id_to_index[camera_id] # this index is used for training

        camera_list[mapped_index] = np.array([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7])])

        image_name = info[-1] 

        # 3. copy images to processed_output_dir
        source_image_path = os.path.join(raw_image_dir, image_name)
        target_image_path = os.path.join(output_dir, 'image', str(mapped_index).zfill(3) + '.png')
        shutil.copy(source_image_path, target_image_path)
        print('copy image from %s to %s' % (source_image_path, target_image_path))

        # 4. copy masks to processed_output_dir (optional)
        if raw_mask_dir is None:
            if mapped_index == 0:
                H, W = imageio.imread(source_image_path).shape[:2]
                fake_mask = np.ones((H, W), dtype=np.uint8) * 255
            imageio.imwrite(os.path.join(output_dir, 'mask', str(mapped_index).zfill(3) + '.png'), fake_mask)
        else:
            source_mask_path = os.path.join(raw_mask_dir, image_name)
            target_mask_path = os.path.join(output_dir, 'mask', str(mapped_index).zfill(3) + '.png')
            shutil.copy(source_mask_path, target_mask_path)
            print('copy mask from %s to %s' % (source_mask_path, target_mask_path))
    if raw_mask_dir is None:
        print(colored('No mask is provided. Set all pixels as 255 as the fake mask', 'red', attrs=['bold']))

    quaternion = np.concatenate([camera_list[:, 1:4], camera_list[:, :1]], axis=-1) # WXYZ -> XYZW
    cam_pos = camera_list[:, 4:, None] # [N, 3, 1]
    rotation = Rotation.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    w2c = np.concatenate([rotation_matrix, cam_pos], axis=2) # [N, 3, 4]
    w2c = np.concatenate([w2c, np.array([[[0, 0, 0, 1]]]).repeat(len(w2c), axis=0)], axis=1) # [N, 4, 4]
    c2w = np.linalg.inv(w2c) # [N, 4, 4]
    assert len(w2c) == n_cam, 'number of cameras poses is not equal to the number of camera intrinsics' 
 
    # 5. combine intrinsics and camera poses
    # In NueS, The project matrix is defined as: intrinsic @ camera_matrix @ scale_mat @ world_point
    # since world_point is in the unit sphere (radius=1), the scale_mat should map these world_point to the range of colmap's bbox
    # so the "scale_mat" here is the scale matrix from unit sphere to bbox
    scale_mat = np.linalg.inv(inv_scale_matrix)

    camera_dict = {}
    for i in range(n_cam):
        # be consistent with NueS
        camera_dict['world_mat_%d' % i] = intrinsics[i] @ w2c[i]  # world_mat is a projection matrix from world to image @ w2c[i]
        camera_dict['scale_mat_%d' % i] = scale_mat # shared by all cameras
        camera_dict['world_mat_inv_%d' % i] = np.linalg.inv(camera_dict['world_mat_%d' % i])
        camera_dict['scale_mat_inv_%d' % i] = np.linalg.inv(scale_mat) # or inv_scale_matrix
    np.savez(os.path.join(output_dir, 'cameras_sphere.npz'), **camera_dict) # save as npz

    print(colored('Data processing is done!', 'blue', attrs=['bold']))

    # test 
    rays_o, rays_d = gen_rays_at(H=H, W=W, intrinsic=intrinsics[0],
                                 c2w=np.linalg.inv(w2c[0]), # np.linalg.inv(w2c[i] @ scale_mat)
                                 inv_scale_mat=inv_scale_matrix,
                                 resolution_level=4)

    # K1, Rt1 = load_K_Rt_from_P(intrinsics[0] @ w2c[i] @ scale_mat)
    K2, Rt2 = decomposeP(intrinsics[0] @ w2c[0])

    rays_o = rays_o.reshape(-1, 3)[::30]
    rays_d = rays_d.reshape(-1, 3)[::30]
    assert np.allclose(np.linalg.norm(rays_d, ord=2, axis=-1), 1.0), 'rays_d should be normalized' 
    z_vals = torch.linspace(0, 1, 5)
    near, far = near_far_from_sphere(rays_o, rays_d)

    z_vals = torch.linspace(0.0, 1.0, 5)
    z_vals = near + (far - near) * z_vals[None, :]
    points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None] # [N_rays, N_samples, 3]
    draw_bbox_3d(points.reshape(-1, 3), *target_object_bbox_min, *target_object_bbox_max, title='sampled points of camera 0', set_lim=False)