from picamera2 import Picamera2, Preview
import time
import json

tuning =  Picamera2.load_tuning_file('imx296_mono_custom.json',dir="./")           
picam2 = Picamera2(tuning=tuning)
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
print(camera_config)

picam2.start_preview(Preview.QTGL)
picam2.start()
picam2.capture_file("test.jpg")
while(1):
    pass

"""
{
 'use_case': 'preview', 
 'transform': <libcamera.Transform 'identity'>,
  'colour_space': <libcamera.ColorSpace 'sYCC'>, 
  'buffer_count': 4, 
  'queue': True, 
    'main': {
        'format': 'XBGR8888', 
        'size': (640, 480), 
        'stride': 2560, 
        'framesize': 1228800
    }, 
    'lores': None, 
    'raw': {
        'format': 'SBGGR10_CSI2P', 
        'size': (1456, 1088), 
        'stride': 1824, 
        'framesize': 1984512
    }, 
    'controls': {
        'NoiseReductionMode': <NoiseReductionModeEnum.Minimal: 3>, 
        'FrameDurationLimits': (100, 83333)
    }, 
    'sensor': {}, 
    'display': 'main', 
    'encode': 'main'
}
"""
