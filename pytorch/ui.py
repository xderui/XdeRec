from heapq import merge
from logging import exception
from pprint import pprint
import gradio as gr
import gc
import torch
import os
import os.path
import argparse

import csv
import subprocess
import tkinter as tk
from tkinter import filedialog
import json
import threading
import time
import psutil
from threading import Timer
import signal
import shlex
from multiprocessing import  Process
from zipfile import ZipFile
import zipfile
import shutil
import mimetypes
from queue import Queue
import tempfile
import webbrowser
import socket
import socketserver
import random
import platform


# Global Settings
root_path = './'
model_type_path = os.path.join(root_path, './model/model')
model_config_path = os.path.join(root_path, './model/config')
dataset_path = os.path.join(root_path, './dataset')

model_type_list = os.listdir(model_type_path)
dataset_list = os.listdir(dataset_path)

# listening client
listen_client = None
internet_ip = "127.0.0.1"
server_port = 8000
listen_display = None

# Global Parameters
training_flag = False
training_process = None
template_config_path = os.path.join(root_path, './config/train.json')
template_params_dict = json.load(open(template_config_path, 'r'))
widget_dict = {}
widget_num_dict = {}

model_config_widgets = []



def save_config(config_dict, config_path):
    json.dump(config_dict, open(config_path, 'w'))

def load_config(config_path):
    return json.load(open(config_path, 'r'))

def UI():
    with gr.Blocks(analytics_enabled=False) as ui:
        global widget_dict, widget_num_dict
        ''' Training Setting Module'''
        widget_dict['train_config'] = {}
        with gr.Blocks():
            gr.Markdown(
                '''
                ## Training Setting
                You can set the training details here.
                '''
            )
            with gr.Row():
                model_type = gr.Dropdown(
                    label='Model Type',
                    choices=model_type_list
                )
                def Type_Model(model_type):
                    model_list = os.listdir(os.path.join(model_type_path, model_type))
                    model_list = [os.path.splitext(model_name)[0] for model_name in model_list if '_' not in model_name]
                    return gr.Dropdown(choices=model_list, value=model_list[0] if model_list else None, interactive=True)
                
                model = gr.Dropdown(
                    label='Model Name',
                )

                model_type.change(Type_Model, inputs=model_type, outputs=model)

            ''' Parameters Module'''
            ''' The parameters of Model is different, so the layerout should be built adaptively'''
            gr.Markdown(
                '''
                ## Model Parameters
                '''
            )
            
            @gr.render(inputs=[model_type, model])
            def Model_Params(model_type, model_name):
                config_path = os.path.join(model_config_path, model_type)
                config_list = os.listdir(config_path)
                models = [os.path.splitext(model_name)[0] for model_name in config_list]
                
                config_index = models.index(model_name)
                model_config_file = os.path.join(config_path, config_list[config_index])
                model_config_dict = json.load(open(model_config_file, 'r'))
                print(model_config_dict)
                
                global model_config_widgets
                param_components = []
                with gr.Row() as ModelParams_:
                    for key, value in model_config_dict.items():
                        if isinstance(value, str):
                            current_widget = gr.Textbox(label=key, value=value, interactive=True)
                        else:
                            current_widget = gr.Number(label=key, value=value, interactive=True)
                        param_components.append(current_widget)
                        model_config_widgets.append(current_widget)
                print(param_components)
                return param_components

            ''' Left config '''
            gr.Markdown(
                '''
                ## Training Parameters
                '''
            )
            with gr.Row():
                for k,v in template_params_dict['train_config'].items():
                    try:
                        eval(k)
                    except: # not exsits
                        if isinstance(v, str):
                            current_widget = gr.Textbox(label=k, value=v, interactive=True)
                        else:
                            current_widget = gr.Number(label=k, value=v, interactive=True)
                        exec(f"{k}=current_widget")
                    
                
                # save state to widget dict
                widget_dict['train_config'] = {}
                for k,v in template_params_dict['train_config'].items():
                    widget_dict['train_config'][k] = eval(k)
                    print(widget_dict['train_config'][k])


        ''' Data Selection Module'''
        with gr.Blocks():
            gr.Markdown(
                '''
                ## Dataset
                You can set the reading and processing parameters of the dataset here.
                '''
            )

            with gr.Row():
                dataset = gr.Dropdown(
                    label='Dataset',
                    choices=dataset_list
                )
                
                file = gr.Dropdown(
                    label='Interactions',
                )
            
                def Dataset_Files(dataset):
                    file_list = os.listdir(os.path.join(dataset_path, dataset))
                    return gr.Dropdown(choices=file_list, value=file_list[0] if file_list else None, interactive=True)

                dataset.change(Dataset_Files, inputs=dataset, outputs=file)

            with gr.Row() as DatasetParams:
                def File_Preview(dataset, filename):
                    file_path = os.path.join(dataset_path, dataset, filename)
                    all_content = open(file_path, 'r', encoding='ISO-8859-1')
                    content = ""
                    n = 10
                    for i, line in enumerate(all_content):
                        if i<n:
                            content += line

                    return gr.Textbox(value=content)
                

                ''' Left config '''
                with gr.Column():
                    with gr.Row():
                        for k,v in template_params_dict['dataset_config'].items():
                            try:
                                eval(k)
                            except:
                                if isinstance(v, str):
                                    current_widget = gr.Textbox(label=k, value=v, interactive=True)
                                elif isinstance(v, list):
                                    # check the type of each element!
                                    all_str_flag = True
                                    for elem in v:
                                        if not isinstance(elem, str):
                                            all_str_flag = False
                                            break
                                    if all_str_flag:
                                        current_widget = gr.CheckboxGroup(label=k, choices=v)
                                    else:
                                        current_widget_list = []
                                        with gr.Row():
                                            for idx, elem in enumerate(v):
                                                sub_widget = gr.Number(label=f'{k}_{idx}', value=elem, interactive=True)
                                                current_widget_list.append(sub_widget)
                                        current_widget = current_widget_list
                                else:
                                    current_widget = gr.Number(label=k, value=v, interactive=True)
                                exec(f"{k}=current_widget")
                    
                        # save state to widget dict
                        widget_dict['dataset_config'] = {}
                        for k,v in template_params_dict['dataset_config'].items():
                            widget_dict['dataset_config'][k] = eval(k)
                            print(widget_dict['dataset_config'][k])


                # preview file
                with gr.Column():
                    PreviewContent = gr.Textbox(
                        label='Preview',
                    )

                

                file.change(File_Preview, inputs=[dataset, file], outputs=PreviewContent)

        ''' other settings'''
        for other_k, other_v in template_params_dict.items():
            if other_k not in widget_dict.keys():
                with gr.Blocks():
                    gr.Markdown(
                        f'''
                        ## {other_k} 
                        '''
                    )
                    with gr.Row():
                        # print(other_v)
                        for k, v in other_v.items():
                            if isinstance(v, str):
                                current_widget = gr.Textbox(label=k, value=v, interactive=True)
                            elif isinstance(v, list):
                                # check the type of each element!
                                all_str_flag = True
                                for elem in v:
                                    if not isinstance(elem, str):
                                        all_str_flag = False
                                        break
                                if all_str_flag:
                                    current_widget = gr.CheckboxGroup(label=k, choices=v)
                                else:
                                    current_widget_list = []
                                    with gr.Row():
                                        for idx, elem in enumerate(v):
                                            sub_widget = gr.Number(label=f'{k}_{idx}', value=elem, interactive=True)
                                            current_widget_list.append(sub_widget)
                                    current_widget = current_widget_list
                            else:
                                current_widget = gr.Number(label=k, value=v, interactive=True)
                            exec(f"{k}=current_widget")
                    
                    # save state to widget dict
                    widget_dict[other_k] = {}
                    for k,v in template_params_dict[other_k].items():
                        widget_dict[other_k][k] = eval(k)
                        print(widget_dict[other_k][k])

        
        with gr.Row():
            train_result = gr.Textbox(label='Training results', placeholder='(if there is no orange bounding box, it means the training is finished!)', interactive=False)
        with gr.Row():
            train_btn = gr.Button(elem_id='train_button', value='Train!', variant='primary')
            stop_train_btn = gr.Button(elem_id='stop_train_button', value='Stop training')


        # bind function
        input_widgets = []
        for config_name, config_item in widget_dict.items():
            widget_num_dict[config_name] = {}
            for item_key, item_value in config_item.items():
                if isinstance(item_value, list):
                    widget_num_dict[config_name][item_key] = len(item_value)
                    for item_value_value in item_value:
                        input_widgets.append(item_value_value)
                else:
                    widget_num_dict[config_name][item_key] = 1
                    input_widgets.append(item_value)

        train_btn.click(
            fn=train_model,
            inputs=input_widgets,
            outputs=train_result
        )

        stop_train_btn.click(
            fn=stop_train_model
        )

        return ui
    
def train_model(*widget_values):
    global training_process, training_flag
    # export the dict to config file
    config_dict = {}
    widget_index = 0
    for config_key, config_item in widget_dict.items():
        config_dict[config_key] = {}
        for item_key, item_value in config_item.items():
            if isinstance(item_value, list):
                widget_value = []
                for _ in item_value:
                    widget_value.append(widget_values[widget_index])
                    widget_index += 1
            else:
                widget_value = widget_values[widget_index]
                widget_index += 1
                
            config_dict[config_key][item_key] = widget_value
            
    config_fd, config_path = tempfile.mkstemp()
    config_json = json.dumps(config_dict)
    with open(config_path, 'w') as f:
        f.write(config_json)
        f.close()
    
    print("config_path: ", config_path)
    
    # build the cmd
    run_cmd = f"python main.py --config {config_path}"

    try:
        if training_flag == False:
            training_flag = True
            fd, path = tempfile.mkstemp()
            print('temp_file:,', path)
            tmp_file = os.fdopen(fd, 'w')
            fileno = tmp_file.fileno()
            training_process = subprocess.Popen(run_cmd, shell=True, stdout=fileno, stderr=fileno, env=os.environ.copy()) 
    except Exception as e:
        print(e)

    # start listen
    global listen_client
    if not listen_client:
        listen_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # try to connect server
    try:
        print('connecting')
        listen_client.connect((internet_ip, server_port))
    except Exception as e:
        print('connect failed')
        try:
            listen_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_client.connect((internet_ip, server_port))
        except Exception as e:
            print(e)
    
    # establish the connection
    try:
        listen_port = int(listen_client.recv(1024).decode())
        print(f'port is {listen_port}')
        raise gr.Error('address: http://{}:{}/'.format(internet_ip, listen_port))
    except Exception as e:
        print(e)

    cost_time = 0
    last_content_len = 0
    try:
        print('Getting process status...')
        training_process_status = subprocess.Popen.poll(training_process)
    except:
        try:
            subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=training_process.pid))
        except:
            pass
        finally:
            training_flag = False
            listen_client.close()


    while subprocess.Popen.poll(training_process) == None and training_flag == True:
        time.sleep(1.0)
        cost_time += 1

        # send data to server
        with open(path, 'r') as f:
            new_content = f.readlines()
            new_content_len = len(new_content)
            if new_content_len > last_content_len:
                add_content = new_content[last_content_len:new_content_len]
                last_content_len = new_content_len
                # print('add_content:', add_content)
                if len(add_content) > 1024:
                    add_content = add_content[:1024]
                print(add_content)
                listen_client.sendall('\n'.join(add_content).encode())
            

        yield 'listening address:http://{}:{}/, Training time: {}s'.format(internet_ip, listen_port, cost_time)
    
    try:
        subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=training_process.pid))
    except:
        pass

    training_flag = False

    # close the connection
    listen_client.close()

    return 'Train finished!'

def stop_train_model():
    global training_flag
    if training_flag:
        global training_process, listen_client
        listen_client.close()

        try:
            subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=training_process.pid))
        except Exception as e:
            print(e)
        training_flag = False

ui = UI()
ui.launch(server_port=30001)
