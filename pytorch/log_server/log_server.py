from flask import Flask,request,render_template
import time
import logging as lg
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, required=True)
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()
log_file_path = args.log_file
port = args.port

def construct_logger():
    logger = lg.getLogger('logger')
    logger.setLevel(lg.INFO)
    fmt = lg.Formatter("[%(asctime)s] - %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S") 

    global log_file_path
    fh = lg.FileHandler(log_file_path, encoding='utf-8')
    fh.setFormatter(fmt=fmt)
    logger.addHandler(fh)
    sh = lg.StreamHandler()
    sh.setFormatter(fmt=fmt)
    logger.addHandler(sh)
    fh.close()

    return logger

def generate_log():
    logger = construct_logger()
    logger.info('listening')

def read_logs():
    global log_file_path
    with open(log_file_path, 'r') as f:
        log_contnet = f.readlines()

    return log_contnet

    

app = Flask(__name__)

line_number = [0] 

@app.route('/get_log',methods=['GET','POST'])
def get_log():
    log_data = read_logs() 
    if len(log_data) - line_number[0] > 0:
        log_type = 2 
        log_difference = len(log_data) - line_number[0] 
        log_list = [] 

        for i in range(log_difference):
            log_i = log_data[-(i+1)] 
            log_list.insert(0,log_i) 
    else:
        log_type = 3
        log_list = ''

    _log = {
        'log_type' : log_type,
        'log_list' : log_list
    }
    line_number.pop() 
    line_number.append(len(log_data)) 
    return _log


@app.route('/generation_log',methods=['GET','POST'])
def generation_log_():
    if request.method == 'POST':
        generate_log()
    return ''

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port) 
