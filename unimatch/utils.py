class Feature:
    """
    this class record the necessary information of every SIFT feature.
    """

    def __init__(self,
                 image_id_,
                 feature_id_,
                 des_,
                 kp_,
                 depth_is_valid_=False,
                 local_xyz_=None):
        self.img_id = image_id_  # img_id = image name
        self.feature_id = feature_id_
        self.des = des_
        self.kp_dic = Feature.kp_convertor(kp_)
        self.local_xyz = local_xyz_
        self.depth_is_valid = depth_is_valid_

    @staticmethod
    def kp_convertor(kp):
        """
        helper function that converts a opencv keypoint struct into dict
        """
        return {
            "pt": kp.pt,
            "size": kp.size,
            "angle": kp.angle,
            "octave": kp.octave,
            "class_id": kp.class_id,
        }
