from bottle import run, route, template , static_file
import time

@route('/')
def index():
    return template('index')

@route('/camera/<filename>')
def camera(filename):
    #pos = filename.find('?')
    #filename = filename[:pos]
    print(filename)
    picam2.capture_file("./img/"+filename)
    request = picam2.capture_request()
    metadata = request.get_metadata()
    print("ColourTemperature = %d" % metadata["ColourTemperature"])
    print("%.5f , %.5f , %.5f," % (metadata["ColourCorrectionMatrix"][0],metadata["ColourCorrectionMatrix"][1],metadata["ColourCorrectionMatrix"][2]))
    print("%.5f , %.5f , %.5f," % (metadata["ColourCorrectionMatrix"][3],metadata["ColourCorrectionMatrix"][4],metadata["ColourCorrectionMatrix"][5]))
    print("%.5f , %.5f , %.5f" % (metadata["ColourCorrectionMatrix"][6],metadata["ColourCorrectionMatrix"][7],metadata["ColourCorrectionMatrix"][8]))

    request.release()  # requests must always be returned to libcamera
    #picam2.start_and_capture_file("./img/"+filename)
    return static_file(filename, root='./img/')

@route('/img/<filename>')
def show_img(filename):
    return static_file(filename, root='./img/')


from picamera2 import Picamera2, Preview

picam2 = Picamera2(tuning=Picamera2.load_tuning_file('imx219_calibration_1gain.json',dir="."))
camera_config = picam2.create_still_configuration()
#camera_config["main"]["size"] = ( 1280,720 )
picam2.configure( camera_config )
picam2.start_preview(Preview.NULL)

picam2.start()

time.sleep(2)


run(host='0.0.0.0', port=8080, debug=True)