from scipy import ndimage
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import pandas as pd
import scipy 
import time
import os
from PIL import Image


class NetModel(nn.Module):
    def __init__(self,n1,n2):
        super(NetModel,self).__init__()
        
        # Convolution layer 1
        self.layer1 = nn.Conv2d(1,n1,kernel_size=9,stride=1, padding=4,bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Convolution layer 2
        self.layer2 = nn.Conv2d(n1,n2,kernel_size=5,stride=1, padding=2,bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Convolution layer 3
        self.layer3 = nn.Conv2d(n2,1,kernel_size=5,stride=1, padding=2,bias=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        
        x = self.layer3(x)
        
        return x

class SRCNN(object):
    def __init__(self,args):
        """
        used to load hyperparameters
        """
        current_dir = os.getcwd()
        self.n_epoches = args.n_epoches
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_data_dir = os.path.join(current_dir,args.lr_data_dir)
        self.hr_data_dir = os.path.join(current_dir,args.hr_data_dir)
        self.test_percent = args.test_percent
        self.save_res_dir = os.path.join(current_dir,args.save_res_dir)
        self.vali = args.validation
        self.train_data = []
        self.test_data = []

    def load_data(self):
        self.train_data, self.test_data = utils.load_data(self.lr_data_dir,self.hr_data_dir, self.test_percent)
        

    def train(self):
        try:
            assert len(self.train_data) != 0 and len(self.test_data)!=0 
        except:
            print("Must load data first! run self.load_data()")

        n1 = 64
        n2 = 32
        size_f1 = 9
        size_f2 = 5
        size_f3 = 5

        # define object
        net = NetModel(n1,n2)

        # loss function
        criterion = nn.MSELoss(reduction='elementwise_mean')

        #  optimizer
        # optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9) 
        optimizer = optim.Adam(net.parameters(), lr = self.lr)

        # lr decay
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 100], gamma=0.5)


        if not self.vali:
            # record train_loss and test_loss in each epoch
            train_loss = []
            test_loss = []      
            train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
            test_loader  = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=True)
            # start training
            for epoch in range(self.n_epoches):
                train_iter_loss = 0.0
                net.train()
                
                for i,(lr_img, hr_img) in enumerate(train_loader):
                    lr_img = Variable(lr_img)
                    hr_img = Variable(hr_img)
                    optimizer.zero_grad() # clear all gradients in last epoch
                    outputs = net(lr_img) # forward step
                    loss = torch.sqrt(criterion(outputs,hr_img))
                    train_iter_loss += loss.item()/outputs.size()[0]
                    loss.backward()
                    optimizer.step()
                train_loss.append(train_iter_loss/len(train_loader))
                
                
                test_iter_loss = 0.0
                for i, (lr_img,hr_img) in enumerate(test_loader):
                    lr_img = Variable(lr_img)
                    print("test_lr:",lr_img.size())
                    hr_img = Variable(hr_img)
                    print("test_hr:",hr_img.size())
                    outputs = net(lr_img)
                    print(outputs.size())
                    loss = torch.sqrt(criterion(outputs,hr_img))
                    test_iter_loss += loss.item()
                # test_loss.append(test_iter_loss/(len(train_loader)*self.batch_size))
                test_loss.append(test_iter_loss/(len(test_loader)))
                scheduler.step(epoch)

                if (epoch+1) % 1  == 0:
                    print('Epoch %d/%d, Train Loss: %.4f, Test Loss: %.4f'
                        %(epoch+1, self.n_epoches, train_loss[-1],test_loss[-1]))

            self.ctime = time.strftime("%Y%m%d%H%M%S")
            # save result
            loss_info_dir = os.path.join(self.save_res_dir,"result")
            if not os.path.exists(loss_info_dir):
                os.makedirs(loss_info_dir)
            pic_name = self.ctime + "_loss_curve.html"
            pic_name = os.path.join(loss_info_dir, pic_name)
            utils.plot_loss(train_loss, test_loss, pic_name)

            csv_name = self.ctime + "_epoch_loss.csv"
            csv_name = os.path.join(loss_info_dir, csv_name)
            loss_res = pd.DataFrame({"epoch":np.arange(1, len(train_loss)+1), \
                "train loss":train_loss, "test loss":test_loss})
            loss_res.to_csv(csv_name)

            # save model
            model_dir = os.path.join(self.save_res_dir, "model")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_name = self.ctime+"_model.pth"
            torch.save(net.state_dict(), os.path.join(model_dir, model_name))
            para_file = self.ctime+"_para.txt"
            with open(os.path.join(model_dir, para_file),"w") as para_file:
                para_file.write("total epoches: %s \n batch size: %s \n learning rate: %s \n test test_percent: %s" \
                    % (self.n_epoches, self.batch_size, self.lr, self.test_percent))

    def test(self):
        try:
            assert len(self.train_data) != 0 and len(self.test_data)!=0 
        except:
            print("Must load data first! run self.load_data()")
        
        # load model
        model_dir = os.path.join(self.save_res_dir, "model")
        net2 = NetModel(64,32)
        try:
            model_name = self.ctime+"_model.pth"
            net2.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
            print("Train Model Loaded!")
        except:
            print("there is no such model")

        net2.eval()
        test_loader  = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=False)
        test_res_dir = os.path.join(self.save_res_dir, "test_res")
        if not os.path.exists(test_res_dir):
                os.makedirs(test_res_dir)

        test_loss = 0.0
        with torch.no_grad():
            for i, (lr_img,hr_img) in enumerate(test_loader):
                lr_img = Variable(lr_img)
                hr_img = Variable(hr_img)
                outputs = net2(lr_img)

                im_pred = Image.fromarray(np.asarray(outputs.data[0][0]))
                im_pred = im_pred.convert("L")
                im_pred.save(test_res_dir + "\\pred_%s.png" % i)

                im_truth = Image.fromarray(np.asarray(hr_img.data[0][0]))
                im_truth = im_truth.convert("L")
                im_truth.save(test_res_dir + "\\truth_%s.png" % i)


    def test_single_img(self, img_dir):
        # load model
        model_dir = os.path.join(self.save_res_dir, "model")
        net3 = NetModel(64,32)
        try:
            model_name = self.ctime+"_model.pth"
            net3.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
            print("Train Model Loaded!")
        except:
            print("there is no such model")

        net3.eval()
        
        files = os.listdir(img_dir)
        for file in files:
            img = Image.open(os.path.join(img_dir,file))
            img = img.resize((128,128), Image.BICUBIC)
            input_img = torch.FloatTensor(np.asarray(img)[:,:,0]).view(1,1,128,128)
            input_img = Variable(input_img)
            outputs = net3(input_img)

            pred_img_dir = os.path.join(self.save_res_dir, "pred_res")
            if not os.path.exists(pred_img_dir):
                os.makedirs(pred_img_dir)

            
            im_pred = Image.fromarray(np.asarray(outputs.data[0][0]))
            im_pred = im_pred.convert("L")
            im_pred.save(os.path.join(pred_img_dir, file))
