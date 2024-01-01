""" cameratest.py
Test file for camera.
Tests all functions called by main.py.
Procedure:
    * generates neural networks for face and gaze detection
    * connects with camera
    * while user does not quit:
        ** video is captured by camera  /!\ camera must be faced correctly with sufficient light level
        ** if gaze detected and delay exceeded:
            *** estimates gaze direction
        ** displays video, direction estimation, and possible errors
Caller files: None
Called files: MultiMsgSync.py, script.py
"""

# coding=utf-8
import blobconverter
import cv2
import depthai as dai
import numpy as np
from MultiMsgSync import TwoStageHostSeqSync
from depthai_sdk.visualize.bbox import BoundingBox
from djitellopy import Tello
import time

VIDEO_SIZE = (1072, 1072)

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
openvino_version = '2021.4'

def create_output(name: str, output: dai.Node.Output):
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(name)
    output.link(xout.input)

cam = pipeline.create(dai.node.ColorCamera)
# For ImageManip rotate you need input frame of multiple of 16
cam.setPreviewSize(1072, 1072)
cam.setVideoSize(VIDEO_SIZE)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setPreviewNumFramesPool(20)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

create_output('color', cam.video)

# ImageManip that will crop the frame before sending it to the Face detection NN node
face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
face_det_manip.setMaxOutputFrameSize(300*300*3)
cam.preview.link(face_det_manip.inputImage)

#=================[ FACE DETECTION ]=================

print("Creating Face Detection Neural Network...")
face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(
    name="face-detection-retail-0004",
    shaves=6,
    version=openvino_version
))
# Link Face ImageManip -> Face detection NN node
face_det_manip.out.link(face_det_nn.input)

create_output('detection', face_det_nn.out)

#=================[ SCRIPT NODE ]=================

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'age_gender_manip' to crop the initial frame
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

face_det_nn.out.link(script.inputs['face_det_in'])
face_det_nn.passthrough.link(script.inputs['face_pass'])

cam.preview.link(script.inputs['preview'])

with open("script.py", "r") as f:
    script.setScript(f.read())

#=================[ HEAD POSE ESTIMATION ]=================

headpose_manip = pipeline.create(dai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
script.outputs['headpose_cfg'].link(headpose_manip.inputConfig)
script.outputs['headpose_img'].link(headpose_manip.inputImage)

headpose_nn = pipeline.create(dai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(
    name="head-pose-estimation-adas-0001",
    shaves=6,
    version=openvino_version
))
headpose_manip.out.link(headpose_nn.input)

headpose_nn.out.link(script.inputs['headpose_in'])
headpose_nn.passthrough.link(script.inputs['headpose_pass'])

#=================[ LANDMARKS DETECTION ]=================

landmark_manip = pipeline.create(dai.node.ImageManip)
landmark_manip.initialConfig.setResize(48, 48)
script.outputs['landmark_cfg'].link(landmark_manip.inputConfig)
script.outputs['landmark_img'].link(landmark_manip.inputImage)

landmark_nn = pipeline.create(dai.node.NeuralNetwork)
landmark_nn.setBlobPath(blobconverter.from_zoo(
    name="landmarks-regression-retail-0009",
    shaves=6,
    version=openvino_version
))
landmark_manip.out.link(landmark_nn.input)

landmark_nn.out.link(script.inputs['landmark_in'])
landmark_nn.passthrough.link(script.inputs['landmark_pass'])

create_output('landmarks', landmark_nn.out)

#=================[ LEFT EYE CROP ]=================

left_manip = pipeline.create(dai.node.ImageManip)
left_manip.initialConfig.setResize(60, 60)
left_manip.inputConfig.setWaitForMessage(True)
script.outputs['left_manip_img'].link(left_manip.inputImage)
script.outputs['left_manip_cfg'].link(left_manip.inputConfig)
left_manip.out.link(script.inputs['left_eye_in'])

#=================[ RIGHT EYE CROP ]=================

right_manip = pipeline.create(dai.node.ImageManip)
right_manip.initialConfig.setResize(60, 60)
right_manip.inputConfig.setWaitForMessage(True)
script.outputs['right_manip_img'].link(right_manip.inputImage)
script.outputs['right_manip_cfg'].link(right_manip.inputConfig)
right_manip.out.link(script.inputs['right_eye_in'])

#=================[ GAZE ESTIMATION ]=================

gaze_nn = pipeline.create(dai.node.NeuralNetwork)
gaze_nn.setBlobPath(blobconverter.from_zoo(
    name="gaze-estimation-adas-0002",
    shaves=6,
    version=openvino_version,
    compile_params=['-iop head_pose_angles:FP16,right_eye_image:U8,left_eye_image:U8']
))
script.outputs['to_gaze'].link(gaze_nn.input)

create_output('gaze', gaze_nn.out)

#==================================================

with dai.Device(pipeline) as device:
    sync = TwoStageHostSeqSync()

    queues = {}
    
    frame = None
    
    # Delays of drone processing
    delay_init = time.process_time()
    delay = time.process_time() - delay_init
    delay_process = 5   # arbitrary
    
    # Text display
    text_last_direction = "****"
    text_curr_direction = "****"
    text_error = ""
    textFont = cv2.FONT_HERSHEY_SIMPLEX
    textFontScale = 1
    textColor = (255, 255, 255)
    textThickness = 2
    
    # Create output queues
    for name in ["color", "detection", "landmarks", "gaze"]:
        queues[name] = device.getOutputQueue(name)
    
    # Main loop
    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, detections and gaze estimations) to the Sync class.
            if q.has():
                msg = q.get()
                sync.add_msg(msg, name)
                if name == "color":
                    frame = msg.getCvFrame()
                    frame = cv2.flip(frame,1)
                    frame_dimension = frame.shape[0:2]
                    frame_center = (round(frame_dimension[0]/2),round(frame_dimension[1]/2))
 
        msgs = sync.get_msgs()
        
        if (msgs is not None) and (delay > delay_process):
            # Frame update
            frame = msgs["color"].getCvFrame()
            frame = cv2.flip(frame,1)
            # Text update
            text_last_direction = text_curr_direction
            text_error = ""
            
            dets = msgs["detection"].detections
            for i, detection in enumerate(dets):
                try:
                    """ GAZE DETECTION PARAMETERS
                    Getting all coordinates and thresholds for gaze.
                    """
                    gaze = np.array(msgs["gaze"][i].getFirstLayerFp16())
                    # Gaze cartesian coordinates
                    gaze_x, gaze_y = (gaze * 100).astype(int)[:2]
                    # Gaze polar coordinates
                    gaze_distance = np.sqrt(gaze_x**2 + gaze_y**2)
                    gaze_angle = np.arctan2(gaze_y, gaze_x)     # in radians
                    # Gaze thresholds
                    gaze_distance_th = 20   # used to detect forward looking
                    gaze_angle_th = np.pi/8 # <=> angle of tolerance is 22.5 degrees
                    
                    """ GAZE DIRECTION PROCESSING
                    Estimating direction of gaze (forward, left, right, top, bottom)
                    and sends corresponding command to the drone.
                    Directions are estimated with an interval of pi/8 radians around horizontal/vertical,
                    it means that 45 degree gaze will not be interpreted as a direction.
                    """
                    # Looking forward
                    if gaze_distance < gaze_distance_th:
                        text_curr_direction = "CENTER"
                    # Looking to the left
                    elif (gaze_angle >= -gaze_angle_th) and (gaze_angle < gaze_angle_th):
                        text_curr_direction = "LEFT"
                    # Looking to the right
                    elif (gaze_angle >= np.pi - gaze_angle_th) or (gaze_angle < -np.pi + gaze_angle_th):
                        text_curr_direction = "RIGHT"
                    # Looking to the top
                    elif (gaze_angle >= np.pi/2 - gaze_angle_th) and (gaze_angle < np.pi/2 + gaze_angle_th):
                        text_curr_direction = "TOP"
                    # Looking to the bottom
                    elif (gaze_angle >= -np.pi/2 - gaze_angle_th) and (gaze_angle < -np.pi/2 + gaze_angle_th):
                        text_curr_direction = "BOTTOM"
                    # No direction detected
                    else:
                        break
                except:
                    text_error = "No gaze detected."
            delay_init = time.process_time()
        delay = time.process_time() - delay_init
        
        """ DISPLAY ON THE SCREEN
        Displaying video, directions, and possible errors.
        """
        if frame is not None:
            if delay > delay_process:   # drone is not processing
                cv2.putText(frame, "Currrent direction: ****", (30,100), textFont, textFontScale, textColor, textThickness)
                cv2.rectangle(frame, (0,0), (frame_dimension[0], frame_dimension[1]), (0,255,0), 20)
            else:                       # drone is processing
                cv2.putText(frame, "Currrent direction: " + text_curr_direction, (30,100), textFont, textFontScale, textColor, textThickness)
                cv2.rectangle(frame, (0,0), (frame_dimension[0], frame_dimension[1]), (0,0,255), 20)
            cv2.putText(frame, f"Delay: {delay:.1f}", (30,50), textFont, textFontScale, textColor, textThickness)
            cv2.putText(frame, "Last direction: " + text_last_direction, (30,150), textFont, textFontScale, textColor, textThickness)
            cv2.putText(frame, text_error, (30,200), textFont, textFontScale, textColor, textThickness)
            cv2.imshow("Video", frame)
        
        # Exit condition
        if cv2.waitKey(1) == ord('q'):
            break