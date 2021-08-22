import sys

import cv2
import numpy as np
import pyrealsense2 as rs

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(
            "usage: capture.py images_directory prefix"
        )
        print("absolute path is recommended, directory should end with /")
        print("rgb image i: {images_directory}rgb/{prefix}_{i}_c.png")
        print("depth image i: {images_directory}depth/{prefix}_{i}_d.png")
        exit()

    images_directory = sys.argv[1]
    prefix = sys.argv[2]

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == "L500":
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 10  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    image_id = 0
    cmd = None

    # Streaming loop
    try:
        while True:

            cmd = input()

            if cmd == "e":
                break

            # Get frame pair of color and depth
            frames = pipeline.wait_for_frames()
            timestamp = frames.get_timestamp()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            # aligned_depth_frame is a 640x480 depth image
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            cname = images_directory + "rgb/" + prefix + '_' + str(image_id) + "_c.png"
            dname = images_directory + "depth/" + prefix + '_' + str(image_id) + "_d.png"
            print(cname, dname)
            cv2.imwrite(cname, color_image)
            cv2.imwrite(dname, depth_image)

            image_id += 1

    finally:
        pipeline.stop()
