from picamera2 import Picamera2, Preview
import time
import json

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
print(camera_config)
with open('camera_config.json', 'w') as f:
    json.dump(camera_config, f, ensure_ascii=False)

picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(2)
while 1:
    picam2.capture_file("test.jpg")
    time.sleep(1)

    