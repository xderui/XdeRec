'''
Descripttion: server
version: 1.0.0
Author: Rigel Ma
Date: 2023-08-24 11:55:28
LastEditors: Rigel Ma
LastEditTime: 2023-11-25 15:45:12
'''
from socketserver import BaseRequestHandler, ThreadingTCPServer
import tempfile
import socket
import random
import subprocess
import os

def check_port(host, port):
    s = socket.socket()
    try:
        s.connect((host, port))
        return True
    except:
        return False
    finally:
        s.close()

class Handler(BaseRequestHandler):

    def handle(self) -> None:
        self.request.settimeout(100)
        address, pid = self.client_address

        print(f'{address} connected!')
        # random port
        allocate_port = random.randint(1024,65535)
        while check_port('127.0.0.1', allocate_port):
            allocate_port = random.randint(1024,65535)
        self.request.sendall(str(allocate_port).encode())

        print('allocate:', allocate_port)
        # launch the log_server
        fd, path = tempfile.mkstemp()
        print('temp_file:,', path)
        listen_cmd = "python ./log_server.py --log_file {} --port {}".format(path, allocate_port)
        listen_process = subprocess.Popen(listen_cmd, shell = True)
        
        invalid_cnt = 0
        while True:
            try:
                data = self.request.recv(1024)
            except:
            # if len(data) <= 0:
                print(f"address {address} close!")
                subprocess.Popen("kill -9 {pid}".format(pid=listen_process.pid), shell=True)
                # listen_process.kill()
                # listen_process.wait()
                break
            recv_data = data.decode('utf-8', 'ignore')
            if len(recv_data) == 0:
                invalid_cnt += 1
                if invalid_cnt > 10:
                    break
            else:
                invalid_cnt = 0
            print('recv:', recv_data)
            with open(path, 'a+') as f:
                f.write(recv_data)
            f.close()
        self.request.close()


        

if __name__ == '__main__':
    server = ThreadingTCPServer(('0.0.0.0', 8000), Handler)
    print("Listening")
    server.serve_forever()
