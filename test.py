from bottle import route, run

@route('/hello')
def hello():
    return "Hello World!"

run(host='192.168.100.9', port=8080, debug=True)