from bottle import route, run, response

@route('/')
def index():
    return "hey! works!"

@route('/image.png')
def myimage():
    response.headers["Content-Type"] = "image/png"
    with open("image.png", "rb") as fp:
        return fp.read()

run(host='0.0.0.0', port=8080)
