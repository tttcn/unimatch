import numpy as np
from pytransform3d import transformations as pt

# 这是一个摄像机坐标系中的单位向量，指向z轴正向，在我们的定义中，从摄像机光心指向外部。
# 我们将会把这个向量转换到世界坐标系中用于计算摄像头的朝向（也是用户的朝向）。

Z_UNIT_VEC = np.array([0, 0, 1, 1])


class MapPose(object):
    def __init__(self, world2cam_pose) -> None:
        """
        world2cam_pose: (x,y,z,qw,qx,qy,qz)
        """
        self.timestamp = None
        self.world2cam_pose = world2cam_pose
        self.world2cam_mat = pt.transform_from_pq(self.world2cam_pose)
        self.cam2world_mat = pt.invert_transform(self.world2cam_mat)
        self.cam2world_pose = pt.pq_from_transform(self.cam2world_mat)
        self.xyz = self.cam2world_pose[:3]
        self.quaternion = self.cam2world_pose[3:7]
        self.z_axis = (self.cam2world_mat @ Z_UNIT_VEC)[:3] - self.xyz
        # c是一个复数，用于描述用户在XY平面中的方向。
        self.c = np.complex(self.z_axis[0], self.z_axis[1])
        self.theta = np.angle(self.c, deg=True)
