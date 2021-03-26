# Run this code by
# python stream.py --render

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import argparse
import os.path as osp
import torch
import pyrealsense2 as rs
import time

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import draw_2d_skeleton, draw_3d_skeleton
from hand_shape_pose.util import renderer

class Hand_Graph_CNN():
    def __init__(self, args, cam_img_width=640, cam_img_height=480):
        self.args = args
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()

        output_dir = osp.join(cfg.EVAL.SAVE_DIR, args.config_file)
        logger = setup_logger("hand_shape_pose_inference", output_dir, filename='eval-' + get_logger_filename())
        logger.info(cfg)


        # 1. Load network model
        self.model = ShapePoseNetwork(cfg, output_dir)
        self.device = cfg.MODEL.DEVICE
        self.model.to(self.device)
        self.model.load_model(cfg)

        # 3. Inference
        self.model.eval()
        self.cam_param = torch.Tensor([[2210.0759, 2437.6676, 664.2237, 383.9007]]).to(self.device)
        self.bbox = torch.Tensor([[320., 240., 100., 100.]]).to(self.device)
        self.pose_root = torch.Tensor([[0, 0, 0]]).to(self.device)
        self.pose_scale = torch.Tensor([5.]).to(self.device)
        self.h_ratio = float(640/256)
        self.v_ratio = float(480/256)
        self.render_output = args.render

        if (self.args.device_type == 'normal'):
            self.cap = cv2.VideoCapture(self.args.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_img_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_img_height)
            
        elif (self.args.device_type == 'realsense'):
            self.pipeline = rs.pipeline()

            # Create a config and configure the pipeline to stream
            #  different resolutions of color and depth streams
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))

            # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.depth, cam_img_width, cam_img_height, rs.format.z16, 30)
            if device_product_line == 'L500':
                config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            else:
                config.enable_stream(rs.stream.color, cam_img_width, cam_img_height, rs.format.bgr8, 30)

            # Start streaming
            profile = self.pipeline.start(config)

            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away
            clipping_distance_in_meters = 1 #1 meter
            clipping_distance = clipping_distance_in_meters / depth_scale

            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)


    def get_frames(self):
        if (self.args.device_type == 'normal'):
            ret, frame = self.cap.read()
            if ret:
                return frame, None
            
        elif (self.args.device_type == 'realsense'):
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames() # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # # Validate that both frames are valid
            # if not aligned_depth_frame:# or not color_frame:
            #     continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)

            return color_image, depth_image 

    def detect(self, image, depth=None):
        # check if the image is RGBD
        use_depth = False
        if(depth is not None and use_depth):
            depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            disp_image = np.hstack((image, depth))

        # Our operations on the frame starts here
        input_image = cv2.resize(image, (256, 256))
        input_image = torch.Tensor(input_image).reshape(1, 256, 256, 3).to(self.device)
        
        est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz = \
            self.model(input_image, self.cam_param, self.bbox, self.pose_root, self.pose_scale)

        # u and v pixel position
        pose_uv = est_pose_uv[0].detach().cpu().numpy()
        pose_uv[:, 0] = self.h_ratio * pose_uv[:, 0]
        pose_uv[:, 1] = self.v_ratio * pose_uv[:, 1]

        # x y and z points 
        pose_xyz = est_pose_cam_xyz[0].detach().cpu().numpy()
        
        # estimate the position of the wrist coordinate
        temp_pose_mcp = [0, 0, 0]
        temp_pose_pip = [0, 0, 0]
        for i in range(5, 20, 4):
            temp_pose_mcp += pose_xyz[i]
            temp_pose_pip += pose_xyz[i+1]
        diff = (0.25 * (temp_pose_pip - temp_pose_mcp))
        pose_xyz[0] = (0.25 * temp_pose_mcp) - 2 * diff

        if(self.render_output):
            skeleton_overlay = draw_2d_skeleton(image, pose_uv)
            skeleton_3d = draw_3d_skeleton(pose_xyz, input_image[0].detach().cpu().numpy().shape[:2])
            return pose_uv, pose_xyz, skeleton_overlay, skeleton_3d

        return pose_uv, pose_xyz, None, None

    def end_stream(self):
        if(self.args.device_type == 'normal'):
            self.cap.release()
            cv2.destroyAllWindows()
        
        elif(self.args.device_type == 'realsense'):        
            self.pipeline.stop()

def main(args):
    hg = Hand_Graph_CNN(args)
    
    while True:
        # calculate time
        start_time = time.time()

        # get frames
        image, depth = hg.get_frames()
        
        # get data
        pose_uv, pose_xyz, skeleton_overlay, skeleton_3d = hg.detect(image)

        # show results
        if(args.render):
            cv2.imshow('UV Frame', skeleton_overlay)
            cv2.imshow('XYZ Frame', skeleton_3d)

        end_time = time.time()
        FPS = (1 / (end_time - start_time))
        print ("FPS : {:.2f}".format(FPS))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hg.end_stream()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
    ap.add_argument("--config-file", default="configs/eval_real_world_testset.yaml",
        metavar="FILE", help="path to config file")
    ap.add_argument("opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER)
    ap.add_argument("--device-type", default='normal', type=str,
    	help="input device")
    ap.add_argument("--device-id", default=0, type=int,
        help="device id")
    ap.add_argument("--render", action='store_true',
        help="view the rendered output")
    args = ap.parse_args()

    main(args)