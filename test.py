from bottle import run, route, template , static_file

@route('/')
def index():
    return template('index')

@route('/camera/<filename>')
def camera(filename):
    picam2.start_and_capture_file("./img/"+filename)
    return static_file(filename, root='./img/')

@route('/img/<filename>')
def show_img(filename):
    return static_file(filename, root='./img/')


from picamera2 import Picamera2, Preview
picam2 = Picamera2()
picam2.start_preview(Preview.NULL)

run(host='192.168.100.9', port=8080, debug=True)