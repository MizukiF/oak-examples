import os
from dotenv import load_dotenv

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
import cv2
import numpy as np

from utils.arguments import initialize_argparser
from utils.input import create_input_node

load_dotenv(override=True)

_, args = initialize_argparser()

if args.api_key:
    os.environ["DEPTHAI_HUB_API_KEY"] = args.api_key

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()
print(f"Platform: {platform}")

# HostNode to filter person detections, add circle indicator, and show distance
class PersonDetectionWithDepth(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.video_out = self.createOutput()
        self.detection_out = self.createOutput()
    
    def build(self, video_input, detection_input, depth_input):
        self.link_args(video_input, detection_input, depth_input)
        return self
    
    def process(self, video_frame, detection_packet, depth_frame):
        frame = video_frame.getCvFrame()
        depth_map = depth_frame.getFrame()
        
        # Check for person detections (class 0 in COCO)
        person_detected = False
        filtered_detections = []
        
        if hasattr(detection_packet, 'detections'):
            for det in detection_packet.detections:
                if det.label == 0:  # Person class only
                    person_detected = True
                    filtered_detections.append(det)
                    
                    # Calculate distance at the center of the bounding box
                    x_center = int((det.xmin + det.xmax) / 2 * frame.shape[1])
                    y_center = int((det.ymin + det.ymax) / 2 * frame.shape[0])
                    
                    # Make sure coordinates are within bounds
                    x_center = max(0, min(x_center, depth_map.shape[1] - 1))
                    y_center = max(0, min(y_center, depth_map.shape[0] - 1))
                    
                    # Get depth value in mm
                    depth_mm = depth_map[y_center, x_center]
                    distance_m = depth_mm / 1000.0  # Convert to meters
                    
                    # Draw distance on frame
                    if distance_m > 0:
                        text = f"Person: {distance_m:.2f}m"
                        cv2.putText(frame, text, (x_center - 50, y_center - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw circle in upper left corner
        if person_detected:
            cv2.circle(frame, (30, 30), 15, (255, 0, 0), -1)  # Blue (BGR)
        else:
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)  # Red (BGR)
        
        # Send modified frame
        modified_frame = dai.ImgFrame()
        modified_frame.setCvFrame(frame, video_frame.getType())
        modified_frame.setTimestamp(video_frame.getTimestamp())
        self.video_out.send(modified_frame)
        
        # Send filtered detections
        filtered_packet = dai.ImgDetections()
        filtered_packet.detections = filtered_detections
        filtered_packet.setTimestamp(detection_packet.getTimestamp())
        self.detection_out.send(filtered_packet)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # Create stereo cameras
    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    
    # Request outputs from stereo cameras (lower resolution to save resources)
    left_out = left_cam.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=args.fps_limit)
    right_out = right_cam.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=args.fps_limit)
    
    # Create stereo depth node with minimal resource usage
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 400)
    left_out.link(stereo.left)
    right_out.link(stereo.right)

    # model
    model_description = dai.NNModelDescription(f"yolov6_nano_r2_coco.{platform}.yaml")
    if model_description.model != args.model:
        model_description = dai.NNModelDescription(args.model, platform=platform)
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    # media/camera input
    input_node = create_input_node(
        pipeline,
        platform,
        args.media_path,
    )

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    # Add person detection filter with depth
    person_filter = pipeline.create(PersonDetectionWithDepth).build(
        nn_with_parser.passthrough,
        nn_with_parser.out,
        stereo.depth
    )

    # Visualization
    visualizer.addTopic("Video", person_filter.video_out, "images")
    visualizer.addTopic("Detections", person_filter.detection_out, "images")
    visualizer.addTopic("Depth", stereo.depth, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
