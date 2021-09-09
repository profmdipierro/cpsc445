import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect(("127.0.0.1", 9001))

message = b"Hello"
for k in range(10):
    sock.send(message)
sock.send(b"")
