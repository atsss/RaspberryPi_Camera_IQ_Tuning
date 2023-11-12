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

    #picam2.start_and_capture_file("./img/"+filename)
    return static_file(filename, root='./img/')

@route('/img/<filename>')
def show_img(filename):
    return static_file(filename, root='./img/')


from picamera2 import Picamera2, Preview
picam2 = Picamera2()
picam2.start_preview(Preview.NULL)
picam2.start()
time.sleep(2)


run(host='0.0.0.0', port=8080, debug=True)