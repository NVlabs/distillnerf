import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.spatial.transform import Rotation as R
from scipy import spatial, interpolate
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

class PoseInterpolator:
    '''
    Interpolates the poses to the desired time stamps. The translation component is interpolated linearly,
    while spherical linear interpolation (SLERP) is used for the rotations. https://en.wikipedia.org/wiki/Slerp

    Args:
        poses (np.array): poses at given timestamps in a se3 representation [n,4,4]
        timestamps (np.array): timestamps of the known poses [n]
        ts_target (np.array): timestamps for which the poses will be interpolated [m,1]
    Out:
        (np.array): interpolated poses in se3 representation [m,4,4]
    '''
    def __init__(self, poses, timestamps):

        self.slerp = spatial.transform.Slerp(timestamps, R.from_matrix(poses[:,:3,:3]))
        self.f_x = interpolate.interp1d(timestamps, poses[:,0,3])
        self.f_y = interpolate.interp1d(timestamps, poses[:,1,3])
        self.f_z = interpolate.interp1d(timestamps, poses[:,2,3])

        self.last_row = np.array([0,0,0,1], dtype=np.float32).reshape(1,1,-1)

    def interpolate_to_timestamps(self, ts_target):
        x_interp = self.f_x(ts_target).reshape(-1,1,1).astype(np.float32)
        y_interp = self.f_y(ts_target).reshape(-1,1,1).astype(np.float32)
        z_interp = self.f_z(ts_target).reshape(-1,1,1).astype(np.float32)
        R_interp = self.slerp(ts_target).as_matrix().reshape(-1,3,3).astype(np.float32)

        t_interp = np.concatenate([x_interp,y_interp,z_interp],axis=-2)

        return np.concatenate((np.concatenate([R_interp,t_interp],axis=-1), np.tile(self.last_row,(R_interp.shape[0],1,1))), axis=1)


def transform_point_cloud(pc, T):
    ''' Transform the point cloud with the provided transformation matrix
    Args:
        pc (np.array): point cloud coordinates (x,y,z) [n,3]
        T (np.array): se3 transformation matrix [4,4]

    Out:
        (np array): transformed point cloud coordinated [n,3]
    '''
    return (T[:3,:3] @ pc[:,:3].transpose() + T[:3,3:4]).transpose()



def world_points_2_pixel_py(points, cam_metadata, iterate=False, no_rs=False):

    ''' Projects the points in the global coordinate system to the image plane by compensating for the rollign shutter effect
        on the rolling shutter times, and its effect on point projection.

    Args:
        points (np.array): point coordinates in the global coordinate system [n,3]
        camera_metadata (dict): camera metadata

    Out:
        points_img (np.array): pixel coordinates of the image projections [m,2]
        valid (np.array): array of boolean flags. True for point that project to the image plane
    '''


    img_width = cam_metadata['img_width']
    img_height = cam_metadata['img_height']
    exposure_time = cam_metadata['exposure_time']
    rs_direction = cam_metadata['rolling_shutter_direction']


    t_sof, t_eof = cam_metadata['ego_pose_timestamps']
    T_global_cam_sof = np.linalg.inv(cam_metadata['T_cam_rig']) @ np.linalg.inv(cam_metadata['ego_pose_s'])
    T_global_cam_eof = np.linalg.inv(cam_metadata['T_cam_rig']) @ np.linalg.inv(cam_metadata['ego_pose_e'])
    pose_interpolator = PoseInterpolator(np.stack([T_global_cam_sof, T_global_cam_eof]), np.array([t_sof, t_eof]))

    # Transform the point cloud to the cam coordinate system based on the last pose
    points_cam = transform_point_cloud(points, (T_global_cam_eof + T_global_cam_sof)/2)

    # Preform an initial projection
    initial_proj, initial_valid_idx = project_camera_rays_2_img(points_cam, cam_metadata)

    # # sanity-checking the pytorch version, bp should match initial_proj
    # pcam = torch.FloatTensor(points_cam)
    # intrinsic = torch.FloatTensor(cam_metadata['intrinsic']).unsqueeze(0)
    # bp = batched_project_camera_rays_2_img(pcam, intrinsic, cam_metadata['img_width'], cam_metadata['img_height'], cam_metadata['camera_model'])

    initial_proj = initial_proj[initial_valid_idx,:]
    valid_pts = points[initial_valid_idx,:]

    # Get the time of the acquisition of the first and last row/column
    first_t = t_sof + exposure_time/2
    last_t = t_eof - exposure_time/2
    dt_first_last  = last_t - first_t

    if no_rs:
        # no rolling shutter correction
        T_global_cam = pose_interpolator.interpolate_to_timestamps( (first_t+last_t) / 2)[0]
        return initial_proj, T_global_cam, valid_pts, initial_valid_idx


    optimized_proj = []
    valid_idx = []
    trans_matrices = []

    # TODO: IMPLEMENT ITERATIVE APPROACH ()
    for pt_idx, point in enumerate(initial_proj):
        # TODO: ADAPT THIS FOR ALL ROLLING SHUTTER DIRECTIONS (not a priority as all datasets up to now have either 1 or 2)
        if rs_direction == 1:
            t_h = first_t + np.floor(point[1]) * dt_first_last / (img_height - 1)
        elif rs_direction == 2:
            t_h = first_t + np.floor(point[0]) * dt_first_last / (img_width - 1)
        elif rs_direction == 4:
            t_h = first_t + (img_width - np.ceil(point[0])) * dt_first_last / (img_width - 1)
        else:
            raise ValueError(f'Rolling shutter direction {rs_direction} not valid or not implemented.')

        pix_pose = pose_interpolator.interpolate_to_timestamps(t_h)[0]
        trans_matrices.append(pix_pose)
        tmp_point = transform_point_cloud(valid_pts[pt_idx].reshape(1,-1), pix_pose)

        new_proj, _ = project_camera_rays_2_img(tmp_point, cam_metadata)

        if new_proj.shape[0] > 0:
            optimized_proj.append(new_proj[0])
            valid_idx.append(initial_valid_idx[pt_idx])

        return np.stack(optimized_proj), np.stack(trans_matrices), np.stack(valid_idx)


def numericallyStable2Norm2D(x, y):
    absX = abs(x)
    absY = abs(y)
    minimum = min(absX, absY)
    maximum = max(absX, absY)

    if maximum <= np.float32(0.0):
        return np.float32(0.0)

    oneOverMaximum = np.float32(1.0) / maximum
    minMaxRatio = np.float32(minimum) * oneOverMaximum
    return maximum * np.sqrt(np.float32(1.0) + minMaxRatio * minMaxRatio)



def project_camera_rays_2_img(points, cam_metadata):
    ''' Projects the points in the camera coordinate system to the image plane

    Args:
        points (np.array): point coordinates in the camera coordinate system [n,3]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        img_width (float): image width in pixels
        img_height (float): image hight in pixels
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']
    Out:
        points_img (np.array): pixel coordinates of the image projections [m,2]
        valid (np.array): array of boolean flags. True for point that project to the image plane
    '''

    intrinsic = cam_metadata['intrinsic']
    camera_model = cam_metadata['camera_model']
    img_width = cam_metadata['img_width']
    img_height = cam_metadata['img_height']

    if camera_model == "pinhole":

        # Camera coordinates system is FLU and image is RDF
        normalized_points = -points[:,1:3] / points[:,0:1]
        f_u, f_v, c_u, c_v, k1, k2, k3, k4, k5 = intrinsic
        u_n = normalized_points[:,0]
        v_n = normalized_points[:,1]

        r2 = np.square(u_n) + np.square(v_n)
        r4 = r2 * r2
        r6 = r4 * r2

        r_d = 1.0 + k1 * r2 + k2 * r4 + k5 * r6

        # If the radial distortion is too large, the computed coordinates will be unreasonable
        kMinRadialDistortion = 0.8
        kMaxRadialDistortion = 1.2

        invalid_idx = np.where(np.logical_or(np.less_equal(r_d,kMinRadialDistortion),np.greater_equal(r_d,kMaxRadialDistortion)))[0]

        u_nd = u_n * r_d + 2.0 * k3 * u_n * v_n + k4 * (r2 + 2.0 * u_n * u_n)
        v_nd = v_n * r_d + k3 * (r2 + 2.0 * v_n * v_n) + 2.0 * k4 * u_n * v_n

        u_d = u_nd * f_u + c_u
        v_d = v_nd * f_v + c_v

        valid_flag = np.ones_like(u_d)
        valid_flag[points[:,0] <0] = 0

        # Replace the invalid ones
        r2_sqrt_rcp = 1.0 / np.sqrt(r2)
        clipping_radius = np.sqrt(img_width**2 + img_height**2)
        u_d[invalid_idx] = u_n[invalid_idx] * r2_sqrt_rcp[invalid_idx] * clipping_radius + c_u
        v_d[invalid_idx] = v_n[invalid_idx] * r2_sqrt_rcp[invalid_idx] * clipping_radius + c_v
        valid_flag[invalid_idx] = 0

        # Change the flags of the pixels that project outside of an image
        valid_flag[u_d < 0 ] = 0
        valid_flag[v_d < 0 ] = 0
        valid_flag[u_d > img_width] = 0
        valid_flag[v_d > img_height] = 0

        return np.concatenate((u_d[:,None], v_d[:,None]),axis=1),  np.where(valid_flag == 1)[0]

    elif camera_model == "f_theta":

        # Initialize the forward polynomial
        fw_poly = Polynomial(intrinsic[9:14])

        xy_norm = np.zeros((points.shape[0], 1))

        for i, point in enumerate(points):
            xy_norm[i] = numericallyStable2Norm2D(point[0], point[1])

        cos_alpha = points[:, 2:] / np.linalg.norm(points, axis=1, keepdims=True)
        alpha = np.arccos(np.clip(cos_alpha, -1 + 1e-6, 1 - 1e-6))
        delta = np.zeros_like(cos_alpha)
        valid = alpha <= intrinsic[16]

        delta[valid] = fw_poly(alpha[valid])

        # For outside the model (which need to do linear extrapolation)
        delta[~valid] = (intrinsic[14] + (alpha[~valid] - intrinsic[16]) * intrinsic[15])

        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0

        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * points

        # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= 0.0).squeeze()
        pixel[edge_case_cond, :] = points[edge_case_cond, :]
        points_img = pixel
        points_img[:, :2] += intrinsic[0:2]

        # Mark the points that do not fall on the camera plane as invalid
        x_ok = np.logical_and(0 <= points_img[:, 0], points_img[:, 0] < img_width-1)
        y_ok = np.logical_and(0 <= points_img[:, 1], points_img[:, 1] < img_height-1)
        z_ok = points_img[:,2] > 0.0

        valid = np.logical_and(np.reshape(valid, (valid.shape[0])), np.logical_and(np.logical_and(x_ok, y_ok), z_ok))

        return points_img, np.where(valid==True)[0]


def backwards_polynomial(pixel_norms, intrinsic):
    ret = 0
    for k, coeff in enumerate(intrinsic):
        ret += coeff * pixel_norms**k
    return ret

def pixel_2_camera_ray(pixel_coords, intrinsic, camera_model):
    ''' Convert the pixel coordinates to a 3D ray in the camera coordinate system.

    Args:
        pixel_coords (np.array): pixel coordinates of the selected points [n,2]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']

    Out:
        camera_rays (np.array): rays in the camera coordinate system [n,3]
    '''

    camera_rays = np.ones((pixel_coords.shape[0],3))

    if camera_model == 'pinhole':
        camera_rays[:,0] = (pixel_coords[:,0] + 0.5 - intrinsic[2]) / intrinsic[0]
        camera_rays[:,1] = (pixel_coords[:,1] + 0.5 - intrinsic[5]) / intrinsic[4]

    elif camera_model == "f_theta":
        pixel_offsets = np.ones((pixel_coords.shape[0],2))
        pixel_offsets[:,0] = pixel_coords[:,0] - intrinsic[0]
        pixel_offsets[:,1] = pixel_coords[:,1] - intrinsic[1]

        pixel_norms = np.linalg.norm(pixel_offsets, axis=1, keepdims=True)

        alphas = backwards_polynomial(pixel_norms, intrinsic[4:9])
        camera_rays[:,0:1] = (np.sin(alphas) * pixel_offsets[:,0:1]) / pixel_norms
        camera_rays[:,1:2] = (np.sin(alphas) * pixel_offsets[:,1:2]) / pixel_norms
        camera_rays[:,2:3] = np.cos(alphas)

        # special case: ray is perpendicular to image plane normal
        valid = (pixel_norms > np.finfo(np.float32).eps).squeeze()
        camera_rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

    return camera_rays


def batched_backwards_polynomial(pixel_norms, intrinsic):
    ret = 0

    for k in range(intrinsic.shape[1]):
        ret += intrinsic[:, k:k+1] * torch.pow(pixel_norms,k)
    return ret

def batched_pixel_2_camera_ray(pixel_coords, intrinsic, camera_model):
    ''' Convert the pixel coordinates to a 3D ray in the camera coordinate system.

    Args:
        pixel_coords (FloatTensor): pixel coordinates of the selected points [B,n,2]
        intrinsic (FloatTensor): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']

    Out:
        camera_rays (FloatTensor): rays in the camera coordinate system [B,n,3]
    '''
    B, n, _ = pixel_coords.shape
    camera_rays = torch.ones((B, n, 3)).to(pixel_coords.device)

    if camera_model == 'pinhole':
        pass
    elif camera_model == "f_theta":

        pixel_offsets = torch.ones((B, n, 2)).to(pixel_coords.device)
        pixel_offsets[:,:,0] = pixel_coords[:,:,0] - intrinsic[:,0:1]
        pixel_offsets[:,:,1] = pixel_coords[:,:,1] - intrinsic[:,1:2]

        pixel_norms = torch.norm(pixel_offsets, dim=2)
        alphas = batched_backwards_polynomial(pixel_norms, intrinsic[:, 4:9]).unsqueeze(-1)
        pixel_norms = pixel_norms.unsqueeze(-1)
        camera_rays[:,:,0:1] = (torch.sin(alphas) * pixel_offsets[:,:,0:1]) / pixel_norms
        camera_rays[:,:,1:2] = (torch.sin(alphas) * pixel_offsets[:,:,1:2]) / pixel_norms
        camera_rays[:,:,2:3] = torch.cos(alphas)

        # special case: ray is perpendicular to image plane normal
        valid = (pixel_norms > np.finfo(np.float32).eps).squeeze(-1)
        camera_rays[~valid, :] = torch.FloatTensor([0, 0, 1]).to(pixel_coords.device)  # This is what DW sets these rays to

    return camera_rays


def filter_by_mask(img, mask):
    if mask is not None:
        return img[mask]
    else:
        return img


def batched_fw_poly(intrins, alpha):
    """
    intrins is (N, 5), alpha is (N, 1)
    """
    return intrins[:, 0] + intrins[:, 1] * alpha.pow(1) + intrins[:, 2] * alpha.pow(2) + intrins[:, 3] * alpha.pow(3) + intrins[:, 4] * alpha.pow(4)

def batched_project_camera_rays_2_img(points, intrinsic, camera_model, fixed_topview=False):
    ''' Projects the points in the camera coordinate system to the image plane

    Args:
        points: (N, 3)
        intrinsic: (B, 3)
        points (np.array): point coordinates in the camera coordinate system [n,3]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']
    Out:
        points_img (np.array): pixel coordinates of the image projections [m,2]
        valid (np.array): array of boolean flags. True for point that project to the image plane
    '''
    if fixed_topview:
        distortion = 2.0

        # Initialize the forward polynomial
        xy_norm = torch.linalg.norm(points[:, :2], dim=1)
        cos_alpha = points[:, 2:] / torch.linalg.norm(points, dim=1, keepdim=True)
        alpha = torch.arccos(torch.clip(cos_alpha,  -1 + 1e-6, 1 - 1e-6))
        delta = torch.zeros_like(alpha)
        distort_bxpxk = [torch.ones_like(cos_alpha), cos_alpha]

        fw_poly = intrinsic[:, 9:14]
        Torder = fw_poly.shape[1]
        for nn in range(2, Torder):
            distort_bxpxk.append(cos_alpha**nn)

        distort_bxpxk = torch.cat(distort_bxpxk, dim=2)
        f_theta_bxpx1 = (distort_bxpxk * fw_poly.unsqueeze(1)).sum(dim=2,
                                                                 keepdim=True)
        # reduce distortion
        f_theta_max = f_theta_bxpx1[theta_bxpx1 <= 3.14159 / 2].max()
        f_theta_bxpx1 = f_theta_bxpx1 / f_theta_max
        f_theta_bxpx1 = f_theta_bxpx1**distortion
        f_theta_bxpx1 = f_theta_bxpx1 * f_theta_max
            #
            # costheta_bxpx1 = -z_bxpx1 / dist_bxpx1
            # theta_bxpx1 = torch.acos(torch.clamp(costheta_bxpx1, -1 + 1e-7, 1 - 1e-7))
            # distort_bxpxk = [torch.ones_like(theta_bxpx1), theta_bxpx1]
            #
            # Torder = T_bxk.shape[1]
            # for nn in range(2, Torder):
            #     distort_bxpxk.append(theta_bxpx1**nn)
            # distort_bxpxk = torch.cat(distort_bxpxk, dim=2)
            # f_theta_bxpx1 = (distort_bxpxk * T_bxk.unsqueeze(1)).sum(dim=2,
            #                                                          keepdim=True)
            #
            # # reduce distortion
            # f_theta_max = f_theta_bxpx1[theta_bxpx1 <= 3.14159 / 2].max()
            # f_theta_bxpx1 = f_theta_bxpx1 / f_theta_max
            # f_theta_bxpx1 = f_theta_bxpx1**distortion
            # f_theta_bxpx1 = f_theta_bxpx1 * f_theta_max
            #
            # XY_bxpx2 = XYZ_bxpx3[:, :, 0:2]
            # Rp_bxpx1 = XY_bxpx2.norm(dim=2, keepdim=True) + 1e-10
            # cossinphi_bxpx2 = XY_bxpx2 / Rp_bxpx1
            # projection_bxpx2 = f_theta_bxpx1 * cossinphi_bxpx2 + cxcy_bx2.unsqueeze(1)
            #
            # projection_max = f_theta_bxpx1[theta_bxpx1 <= 3.14159 / 2].max(
            # ) * cossinphi_bxpx2 + cxcy_bx2.unsqueeze(1)
            #
            # # y range is +-1, x will be scaled
            # # if the rendering ratio changes
            # # if the new aspect ratio is the same as old, it is 1
            # # if the new aspect ratio is lower(h increases), it is larger than 1
            # rationew_bx1 = ratio_bx1  * newaspectratio
            # # ratio_bx2 = F.pad(ratio_bx1, [0, 1], 'constant', newaspectratio)
            # ratio_bx2 = torch.cat([ratio_bx1, rationew_bx1], dim=-1)
            # projection_bxpx2 = projection_bxpx2 / ratio_bx2.unsqueeze(1)
            #
            # return projection_bxpx2, theta_bxpx1
    if camera_model == 'pinhole':
        points_img = intrinsic.matmul(points.unsqueeze(-1)).squeeze(-1)
        points_img = torch.cat([points_img[:,:2] / torch.clamp(points_img[:,2:3].detach(), 0.01, 10000), points_img[:, 2:3]], dim=1)
        valid = points_img[:,2:3] > 0
        return points_img, valid.squeeze(1)

    elif camera_model == 'f_theta':
        # Initialize the forward polynomial
        xy_norm = torch.linalg.norm(points[:, :2], dim=1)
        cos_alpha = points[:, 2:] / torch.linalg.norm(points, dim=1, keepdim=True) #torch.clamp(torch.linalg.norm(points, dim=1, keepdim=True).detach(), 0.01, 10000)
        alpha = torch.arccos(torch.clip(cos_alpha,  -1 + 1e-6, 1 - 1e-6))
        delta = torch.zeros_like(alpha)

        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        # xy_norm[bad_norm] = 1
        xy_norm = torch.ones_like(xy_norm) * bad_norm.float() + xy_norm * (1-bad_norm.float()) # for grad
        delta[bad_norm] = 0

        if intrinsic[:, 16] == 0:
            # if the last three dims of intrinsic is not given
            intrinsic[:, 16] = 2.29925
            intrinsic[:, 15] = 398.842
            intrinsic[:, 14] = 1145.298

        valid = alpha <= intrinsic[:, 16]


        delta[valid] = batched_fw_poly(intrinsic[:, 9:14], alpha[valid])
        # For outside the model (which need to do linear extrapolation)
        delta[~valid] = (intrinsic[:, 14] + (alpha[~valid] - intrinsic[:, 16]) * intrinsic[:, 15])


        # compute pixel relative to center
        scale = delta / xy_norm.unsqueeze(1)
        # scale = delta / torch.clamp(xy_norm.unsqueeze(1).detach(), 0.01, 10000)
        pixel = scale * points
        # pixel = torch.clamp(pixel, -20000, 20000)

        # # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= 0.0).squeeze()

        pixel = pixel.to(points.dtype)

        pixel[edge_case_cond, :] = points[edge_case_cond, :]
        points_img = pixel
        points_img[:, :2] += intrinsic[:, 0:2]
        return points_img, valid
