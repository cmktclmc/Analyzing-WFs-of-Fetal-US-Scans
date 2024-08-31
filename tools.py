import numpy as np
import torch
from torchvision import *

from utils import *

train_on_gpu = torch.cuda.is_available()
device = 'cuda' if train_on_gpu else 'cpu'

# -----------------------------------------------------------------------------


#euler batch*4
# #output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):

    batch = euler.shape[0]

    # c1 = torch.cos(euler[:, 0]).view(batch, 1)  #batch*1
    # s1 = torch.sin(euler[:, 0]).view(batch, 1)  #batch*1
    # c2 = torch.cos(euler[:, 2]).view(batch, 1)  #batch*1
    # s2 = torch.sin(euler[:, 2]).view(batch, 1)  #batch*1
    # c3 = torch.cos(euler[:, 1]).view(batch, 1)  #batch*1
    # s3 = torch.sin(euler[:, 1]).view(batch, 1)  #batch*1

    # # XZY
    # row1 = torch.cat((c2 * c3,                 -s2,      c2 * s3                 ), 1).view(-1, 1, 3)  #batch*1*3
    # row2 = torch.cat((c1 * s2 * c3 + s1 * s3,  c1 * c2,  c1 * s2 * s3 - s1 * c3  ), 1).view(-1, 1, 3)  #batch*1*3
    # row3 = torch.cat((s1 * s2 * c3 - c1 * s3,  s1 * c2,  s1 * s2 * s3 + c1 * c3  ), 1).view(-1, 1, 3)  #batch*1*3

    cx=torch.cos(euler[:,0]).view(batch,1)#batch*1
    sx=torch.sin(euler[:,0]).view(batch,1)#batch*1
    cy=torch.cos(euler[:,2]).view(batch,1)#batch*1
    sy=torch.sin(euler[:,2]).view(batch,1)#batch*1
    cz=torch.cos(euler[:,1]).view(batch,1)#batch*1
    sz=torch.sin(euler[:,1]).view(batch,1)#batch*1

    # if order == 'xyz':
    row1=torch.cat((cy*cz,           -cy*sz,          sy              ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((cx*sz+cz*sx*sy,  cx*cz-sx*sy*sz,  -cy*sx          ), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((sx*sz-cx*cz*sy,  cz*sx+cx*sy*sz,  cx*cy           ), 1).view(-1,1,3) #batch*1*3
    # elif order == 'xzy':    # default in the provided function (intrinsic)
    #     row1=torch.cat((cz*cy,           -sz,             cz*sy           ), 1).view(-1,1,3) #batch*1*3
    #     row2=torch.cat((sx*sy+cx*cy*sz,  cx*cz,           cx*sz*sy-cy*sx  ), 1).view(-1,1,3) #batch*1*3
    #     row3=torch.cat((cy*sx*sz-cx*sy,  cz*sx,           cx*cy+sx*sz*sy  ), 1).view(-1,1,3) #batch*1*3
    # elif order == 'yxz':
    #     row1=torch.cat((cy*cz+sy*sx*sz,  cz*sy*sx-cy*sz,  cx*sy           ), 1).view(-1,1,3) #batch*1*3
    #     row2=torch.cat((cx*sz,           cx*cz,           -sx             ), 1).view(-1,1,3) #batch*1*3
    #     row3=torch.cat((cy*sx*sz,        cy*cz*sx+sy*sz,  cy*cx           ), 1).view(-1,1,3) #batch*1*3
    # elif order == 'yzx':
    #     row1=torch.cat((cy*cz,           sy*sx-cy*cx*sz,  cx*sy+cy*sz*sx  ), 1).view(-1,1,3) #batch*1*3
    #     row2=torch.cat((sz,              cz*cx,           -cz*sx          ), 1).view(-1,1,3) #batch*1*3
    #     row3=torch.cat((-cz*sy,          cy*sx+cx*sy*sz,  cy*cx-sy*sz*sx  ), 1).view(-1,1,3) #batch*1*3
    # elif order == 'zxy':
    #     row1=torch.cat((cz*cy-sz*sx*sy,  -cx*sz,          cz*sy+cy*sz*sx  ), 1).view(-1,1,3) #batch*1*3
    #     row2=torch.cat((cy*sz+cz*sx*sy,  cz*cx,           sz*sy-cz*cy*sx  ), 1).view(-1,1,3) #batch*1*3
    #     row3=torch.cat((-cx*sy,          sx,              cx*sy           ), 1).view(-1,1,3) #batch*1*3
    # elif order == 'zyx':
    #     row1=torch.cat((cz*cy,           -cz*sy*sx-cx*sz, sz*sx+cz*cx*sy  ), 1).view(-1,1,3) #batch*1*3
    #     row2=torch.cat((cy*sz,           cz*cx+sz*sy*sx,  cx*sz*sy-cz*sx  ), 1).view(-1,1,3) #batch*1*3
    #     row3=torch.cat((-sy,             cy*sx,           cy*cx           ), 1).view(-1,1,3) #batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  #batch*3*3

    return matrix


# -----------------------------------------------------------------------------


#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):

    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    out_euler = torch.zeros(batch, 3).to(device)
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular

    return out_euler


# -----------------------------------------------------------------------------


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    
    # print(m1[0])
    # print(m2[0])

    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  #batch*3*3

    # extract svd of a rotation matrix and replace diagonal by identity to get a perfect rotation (U*transpose(V))
    # u, s, v = LA.svd(m)
    # m2 = torch.bmm(u, v)

    # print("Before SVD")
    # print(m[0])
    # print("Before SVD")
    # print(m2[0])
    # print("Difference")
    # print(m[0]-m2[0])

    # diff = m == m2
    # diff = diff.all(-1)
    # print(diff[0])

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    # OPTION 1
    cos = torch.min(cos, torch.ones(batch).to(device))
    cos = torch.max(cos, torch.ones(batch).to(device) * -1)
    # OPTION 2
    # cos = torch.clamp(cos, -1 + 1e-7, 1 - 1e-7)

    theta = torch.acos(cos)
    #theta = torch.min(theta, 2*np.pi - theta)
    
    # print(theta[0])

    return theta


# -----------------------------------------------------------------------------

# def convert6D2R(x):

# '''
# Krittin version of the T
# '''

#     T = torch.eye(x).to(device)

#     b1 = x[0, 0:3] / x[0, 0:3].norm()
#     b2 = x[0, 3:6] - (b1.dot(x[0, 3:6])) * b1
#     b2 = b2 / b2.norm()
#     b3 = b1.cross(b2)

#     T[0:3, 0] = b1
#     T[0:3, 1] = b2
#     T[0:3, 2] = b3
#     T[0:3, 3] = x[0, 6:9]

#     return T    # T is a 4x4 transformation

# -----------------------------------------------------------------------------


def compute_T_from_ortho9d(ortho9d):

    '''
    input: 9 parameters computed by the network
    output: 4x4 transformation matrix for rotation and translation
    '''

    rot_6d = ortho9d[:, 0:5]
    transl_3d = ortho9d[:, 6:8]

    rot_mat = compute_rotation_matrix_from_ortho6d(rot_6d)

    T = torch.cat((rot_mat, transl_3d), 2)
    T = torch.cat((T, [0, 0, 0, 1]), 1)

    return T  # T is a 4x4 transformation


# -----------------------------------------------------------------------------

def compute_T_from_transl_and_rotmat(x_transl, x_rotmat):

    '''
    input: translation vector and rotation matrix
    output: 4x4 transformation matrix for rotation and translation
    '''

    T = torch.cat((x_rotmat, x_transl), 2)
    T = torch.cat((T, [0, 0, 0, 1]), 1)

    return T


# -----------------------------------------------------------------------------


def compute_quaternion_from_euler_angles(euler):

    batch = euler.shape[0]

    z = euler[:, 2]
    y = euler[:, 1]
    x = euler[:, 0]

    z = z / 2.0
    y = y / 2.0
    x = x / 2.0

    cz = torch.cos(z).view(batch, 1)  #batch*1
    sz = torch.sin(z).view(batch, 1)  #batch*1
    cy = torch.cos(y).view(batch, 1)  #batch*1
    sy = torch.sin(y).view(batch, 1)  #batch*1
    cx = torch.cos(x).view(batch, 1)  #batch*1
    sx = torch.sin(x).view(batch, 1)  #batch*1

    qw = (cx * cy * cz - sx * sy * sz).view(batch, 1)
    qx = (cx * sy * sz + cy * cz * sx).view(batch, 1)
    qy = (cx * cz * sy - sx * cy * sz).view(batch, 1)
    qz = (cx * cy * sz + sx * cz * sy).view(batch, 1)

    quat = torch.cat([qw, qx, qy, qz]).view(batch, 4)

    return quat


# -----------------------------------------------------------------------------


#T_poses num*3
#r_matrix batch*3*3
def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch = r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = r_matrix.view(batch, 1, 3, 3).expand(
        batch, joint_num, 3, 3).contiguous().view(batch * joint_num, 3, 3)
    src_poses = T_pose.view(1, joint_num, 3, 1).expand(
        batch, joint_num, 3, 1).contiguous().view(batch * joint_num, 3, 1)

    out_poses = torch.matmul(r_matrices, src_poses)  #(batch*joint_num)*3*1

    return out_poses.view(batch, joint_num, 3)


# -----------------------------------------------------------------------------


# batch*n
def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.FloatTensor([1e-8]).to(device))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


# -----------------------------------------------------------------------------


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  #batch*3

    return out


# -----------------------------------------------------------------------------


#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(ortho6d):

    x_raw = ortho6d[:, 0:3]  #batch*3
    y_raw = ortho6d[:, 3:6]  #batch*3

    x = normalize_vector(x_raw)  #batch*3
    z = cross_product(x, y_raw)  #batch*3
    z = normalize_vector(z)  #batch*3
    y = cross_product(z, x)  #batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  #batch*3*3
    # print(matrix.view(-1, 1, 9))

    return matrix


# -----------------------------------------------------------------------------


#in batch*6
#out batch*5
def stereographic_project(a):
    dim = a.shape[1]
    a = normalize_vector(a)
    out = a[:, 0:dim - 1] / (1 - a[:, dim - 1])
    return out


# -----------------------------------------------------------------------------


#in a batch*5, axis int
def stereographic_unproject(a, axis=None):

    """
	Inverse of stereographic projection: increases dimension by one.
	"""
    batch = a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a, 2).sum(1)  #batch
    ans = torch.zeros(batch, a.shape[1] + 1).to(device)  #batch*6
    unproj = 2 * a / (s2 + 1).view(batch, 1).repeat(1, a.shape[1])  #batch*5
    if (axis > 0):
        ans[:, :axis] = unproj[:, :axis]  #batch*(axis-0)
    ans[:, axis] = (s2 - 1) / (s2 + 1)  #batch
    ans[:, axis +
        1:] = unproj[:,
                     axis:]  #batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used

    return ans


# -----------------------------------------------------------------------------
#a batch*5
#out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):

    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2) + 1, np.sqrt(2) + 1, np.sqrt(2)])  #3
    proj_scale = (torch.FloatTensor(proj_scale_np).to(device)).view(
        1, 3).repeat(batch, 1)  #batch,3

    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)  #batch*4
    norm = torch.sqrt(torch.pow(u[:, 1:], 2).sum(1))  #batch
    u = u / norm.view(batch, 1).repeat(1, u.shape[1])  #batch*4
    b = torch.cat((a[:, 0:2], u), 1)  #batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix


# -----------------------------------------------------------------------------


#quaternion batch*4
def compute_rotation_matrix_from_quaternion(quaternion):

    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                     1)  #batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                     1)  #batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy),
                     1)  #batch*3

    matrix = torch.cat(
        (row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(
            batch, 1, 3)), 1)  #batch*3*3

    return matrix


# -----------------------------------------------------------
#quaternions batch*4,
#matrices batch*4*4 or batch*3*3
def compute_quaternions_from_rotation_matrices(matrices):

    batch = matrices.shape[0]

    w = torch.sqrt(1.0 + matrices[:, 0, 0] + matrices[:, 1, 1] +
                   matrices[:, 2, 2]) / 2.0
    w = torch.max(w, torch.zeros(batch).to(device) + 1e-8)  #batch
    w4 = 4.0 * w
    x = (matrices[:, 2, 1] - matrices[:, 1, 2]) / w4
    y = (matrices[:, 0, 2] - matrices[:, 2, 0]) / w4
    z = (matrices[:, 1, 0] - matrices[:, 0, 1]) / w4

    quats = torch.cat(
        (w.view(batch, 1), x.view(batch, 1), y.view(batch, 1), z.view(
            batch, 1)), 1)
    # quats = torch.cat((x.view(batch, 1), y.view(batch, 1), z.view(batch, 1), w.view(batch, 1)), 1) # UNITY FORMAT

    return quats


# -----------------------------------------------------------


#axisAngle batch*4 angle, x,y,z
def compute_rotation_matrix_from_axisAngle(axisAngle):

    batch = axisAngle.shape[0]

    theta = torch.tanh(axisAngle[:, 0]) * np.pi  #[-180, 180]
    sin = torch.sin(theta * 0.5)
    axis = normalize_vector(axisAngle[:, 1:4])  #batch*3
    qw = torch.cos(theta * 0.5)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                     1)  #batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                     1)  #batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy),
                     1)  #batch*3

    matrix = torch.cat(
        (row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(
            batch, 1, 3)), 1)  #batch*3*3

    return matrix


# -----------------------------------------------------------------------------

#axisAngle batch*3 (x,y,z)*theta
def compute_rotation_matrix_from_Rodriguez(rod):

    batch = rod.shape[0]

    axis, theta = normalize_vector(rod, return_mag=True)

    sin = torch.sin(theta)

    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                     1)  #batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                     1)  #batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy),
                     1)  #batch*3

    matrix = torch.cat(
        (row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(
            batch, 1, 3)), 1)  #batch*3*3

    return matrix


# -----------------------------------------------------------------------------


#axisAngle batch*3 a,b,c
def compute_rotation_matrix_from_hopf(hopf):

    batch = hopf.shape[0]

    theta = (torch.tanh(hopf[:, 0]) + 1.0) * np.pi / 2.0  #[0, pi]
    phi = (torch.tanh(hopf[:, 1]) + 1.0) * np.pi  #[0,2pi)
    tao = (torch.tanh(hopf[:, 2]) + 1.0) * np.pi  #[0,2pi)

    qw = torch.cos(theta / 2) * torch.cos(tao / 2)
    qx = torch.cos(theta / 2) * torch.sin(tao / 2)
    qy = torch.sin(theta / 2) * torch.cos(phi + tao / 2)
    qz = torch.sin(theta / 2) * torch.sin(phi + tao / 2)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                     1)  #batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                     1)  #batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy),
                     1)  #batch*3

    matrix = torch.cat(
        (row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(
            batch, 1, 3)), 1)  #batch*3*3

    return matrix


# -----------------------------------------------------------------------------


#euler_sin_cos batch*6
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):

    batch = euler_sin_cos.shape[0]

    s1 = euler_sin_cos[:, 0].view(batch, 1)
    c1 = euler_sin_cos[:, 1].view(batch, 1)
    s2 = euler_sin_cos[:, 2].view(batch, 1)
    c2 = euler_sin_cos[:, 3].view(batch, 1)
    s3 = euler_sin_cos[:, 4].view(batch, 1)
    c3 = euler_sin_cos[:, 5].view(batch, 1)

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  #batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3),
                     1).view(-1, 1, 3)  #batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3),
                     1).view(-1, 1, 3)  #batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  #batch*3*3

    return matrix


# -----------------------------------------------------------------------------


# #matrices batch*3*3
# #both matrix are orthogonal rotation matrices
# #out theta between 0 to 180 degree batch
def compute_angle_from_r_matrices(m):

    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    # cos = torch.min(cos, torch.ones(batch).to(device))
    # cos = torch.max(cos, torch.ones(batch).to(device) * -1)

    cos = torch.clamp(cos, -1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos)

    return theta


# -----------------------------------------------------------------------------


def get_sampled_rotation_matrices_by_quat(batch):

    # quat = euler_2_quat(batch).to(device)
    quat = torch.randn(batch.shape[0], 4).to(device)
    print(quat)
    matrix = compute_rotation_matrix_from_quaternion(quat)
    return matrix


# -----------------------------------------------------------------------------


def get_sampled_rotation_matrices_by_hpof(batch):

    theta = torch.FloatTensor(np.random.uniform(0, 1, batch) * np.pi).to(
        device)  #[0, pi]
    phi = torch.FloatTensor(np.random.uniform(0, 2, batch) * np.pi).to(
        device)  #[0,2pi)
    tao = torch.FloatTensor(np.random.uniform(0, 2, batch) * np.pi).to(
        device)  #[0,2pi)

    qw = torch.cos(theta / 2) * torch.cos(tao / 2)
    qx = torch.cos(theta / 2) * torch.sin(tao / 2)
    qy = torch.sin(theta / 2) * torch.cos(phi + tao / 2)
    qz = torch.sin(theta / 2) * torch.sin(phi + tao / 2)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                     1)  #batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                     1)  #batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy),
                     1)  #batch*3

    matrix = torch.cat(
        (row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(
            batch, 1, 3)), 1)  #batch*3*3

    return matrix


# -----------------------------------------------------------------------------


#axisAngle batch*4 angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle(batch, return_quaternion=False):

    theta = torch.FloatTensor(np.random.uniform(-1, 1, batch) * np.pi).to(
        device)  #[0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.randn(batch, 3).to(device)
    axis = normalize_vector(axis)  #batch*3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    quaternion = torch.cat((qw.view(batch, 1), qx.view(
        batch, 1), qy.view(batch, 1), qz.view(batch, 1)), 1)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                     1)  #batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                     1)  #batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy),
                     1)  #batch*3

    matrix = torch.cat(
        (row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(
            batch, 1, 3)), 1)  #batch*3*3

    if (return_quaternion == True):
        return matrix, quaternion
    else:
        return matrix


# -----------------------------------------------------------------------------


def remove_bs(image_batch, label_batch):

    for i, l in zip(image_batch, label_batch):
        i = torch.squeeze(i)

    return i


# -----------------------------------------------------------------------------


def format_func_y(value, tick_number):
    percent = value * 100
    if (percent > 99.99):
        percent = 100
    elif (percent < 0.000001):
        percent = 0
    out = "%.2f" % percent + "%"
    return out


def format_func_x(value, tick_number):
    out = "%d" % (int(value)) + '°'
    return out


def format_func_x_percentile_logit(value, tick_number):

    percent = value * 100

    out = "%.2f" % percent + "%"
    return out


def format_func_x_percentile(value, tick_number):
    out = str(int(value)) + "%"
    return out


def format_func_y_degree_logit(value, tick_number):

    score = value * 180

    out = str(score) + "°"
    return out


def format_func_y_degree_log(value, tick_number):

    score = np.exp(value)

    out = str(int(score)) + "°"
    return out


def format_func_y_degree(value, tick_number):
    out = str(int(value)) + "°"
    return out

def format_func_y_degree2(value, tick_number):
    out = str(value) + "°"
    return out


def format_func_y_perc(value, tick_number):
    out = str(int(value)) + "%"
    return out


def format_func_x2(value, tick_number):
    out = "%i" % (value / 1000) + 'k'
    return out


def format_func_y2(value, tick_number):
    out = "%d" % value + '°'
    return out
