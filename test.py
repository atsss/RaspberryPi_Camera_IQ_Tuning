from bottle import route, run

@route('/')
def root():
    return "Hello World!"

@route('/camera')
def camera():
    from picamera2 import Picamera2
    picam2 = Picamera2()
    #カメラ画像を保存する
    picam2.start_and_capture_file("test.jpg")
    return "Hello World!"


run(host='192.168.100.9', port=8080, debug=True)