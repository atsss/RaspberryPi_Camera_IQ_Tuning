from bottle import run, route, template

@route('/')
def index():
    return template('index')

@route('/camera')
def camera():
    from picamera2 import Picamera2, Preview
    picam2 = Picamera2()
    picam2.start_preview(Preview.NULL)
    picam2.start_and_capture_file("test.jpg")
    return "Hello World!"


run(host='192.168.100.9', port=8080, debug=True)