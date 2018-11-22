import os
import torch
import time


import numpy as np 
import yagmail as yg 
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go

from scipy import ndimage
from PIL import Image
from sklearn.model_selection import train_test_split


def median_pool(img,filter_size):
    pool_img = ndimage.median_filter(np.asarray(img)[:,:,:-1],filter_size)
    return pool_img

def load_data(lr_folder,hr_folder,test_percent=0.33):
    # currently using num_train & num_test
    # in the future may modified to shuffle and select a portion of all dataset
    train_img = []
    files = os.listdir(lr_folder)
    lr_train, lr_test, hr_train, hr_test = train_test_split(files, files,test_size=test_percent, random_state=42)
    for file in lr_train:
        pair = []
        lr_img = Image.open(os.path.join(lr_folder,file))
#         lr_img = Image.fromarray(median_pool(lr_img,3))
        lr_img = lr_img.resize((128,128),Image.BICUBIC)
        hr_img = Image.open(os.path.join(hr_folder,file))
        pair.append(torch.FloatTensor(np.asarray(lr_img)[:,:,0]).view(1,128,128))
        pair.append(torch.FloatTensor(np.asarray(hr_img)[:,:,0]).view(1,128,128))
        train_img.append(pair)
        
    test_img = []
    for file in lr_test:
        pair = []
        lr_img = Image.open(os.path.join(lr_folder,file))
#         lr_img = Image.fromarray(median_pool(lr_img,3))
        lr_img = lr_img.resize((128,128),Image.BICUBIC)
        hr_img = Image.open(os.path.join(hr_folder, file))
        pair.append(torch.FloatTensor(np.asarray(lr_img)[:,:,0]).view(1,128,128))
        pair.append(torch.FloatTensor(np.asarray(hr_img)[:,:,0]).view(1,128,128))
        test_img.append(pair)

    return (train_img,test_img)

def email_res(reciever,subject="run_res",content=None,attach=None):
    try:
        yag = yg.SMTP(user='widen1226@163.com',password='1S22S22P63S23P6',
                  host='smtp.163.com')
        yag.send(reciever,subject,content,attachments=attach) 
        print("send result to %s successfully" % reciever)
    except:
        print('email did not send due to unknown error')


def plot_loss(train_loss, test_loss, pic_name):
    """
    train_loss: 1*N list
    test_loss: 1*N list
    pic_name: path to save the plot
    """
    x = np.arange(1,len(train_loss)+1)
    trace0 = go.Scatter(
        x = x,
        y = train_loss,
        name = "train_loss"
        )
    trace1 = go.Scatter(
        x = x,
        y = test_loss,
        name = "test_loss"
        )
    layout = dict(title = 'Train Loss v.s. Test Loss',
              xaxis = dict(title = 'Num of Epoches'),
              yaxis = dict(title = 'MSE Loss'),
              )

    data = [trace0, trace1]
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=pic_name, auto_open=False)

if __name__ == '__main__':
    train_loss = list(np.random.rand(10))
    test_loss = list(np.random.rand(10))
    pic_name = "test.html"
    plot_loss(train_loss, test_loss, pic_name)
