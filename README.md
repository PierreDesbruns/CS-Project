# CS-Project

Authors: Agate COMELLO, Pierre DESBRUNS.

## Files

- [main.py](main.py): main file for drone control by gaze estimation.
- [cameratest.py](cameratest.py): test file for camera.
- [dronetest.py](dronetest.py): test file for drone.
- [main_host_side.py](main_host_side.py): used for gaze estimation, see below.
- [MultiMsgSync.py](MultiMsgSync.py): used for gaze estimation, see below.
- [script.py](script.py): used for gaze estimation, see below.
- [requirements.txt](requirements.txt): Requirements installing file.

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py
```

## [Gen2] Gaze estimation

This example demonstrates how to run 3 stage inference (3-series, 2-parallel) using DepthAI.

[![Gaze Example Demo](https://github.com/luxonis/depthai-experiments/assets/18037362/5d7eea47-d5be-4d8a-92b2-7b5ff01d443d)](https://tinyurl.com/5h3dycc5)

(Click on the gif for full-res video)

Original OpenVINO demo, on which this example was made, is official [here](https://docs.openvinotoolkit.org/2021.1/omz_demos_gaze_estimation_demo_README.html) from Intel and implemented with the NCS2 with nice diagrams and explanation, [here](https://github.com/LCTyrell/Gaze_pointer_controller), from @LCTyrell.

![graph](https://user-images.githubusercontent.com/32992551/103378235-de4fec00-4a9e-11eb-88b2-621180f7edef.jpeg)

Figure: @LCTyrell

### Device-side vs host-side

[main.py](main.py) script will run everything on the device, heavily utilizing script node (which runs [script.py](script.py) on the device). Benefits for that include lower bandwidth usage (as data isn't being sent back and forth) and lower latency.

[main_host_side.py](main_host_side.py) is the same application, but it send frames/NN data to/from host for each inference. All the AI compute still happens on the device, but host computer does the decoding and preparing data for next NN inference.
