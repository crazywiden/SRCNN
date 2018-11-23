import model 
import torch
import argparse
import time
import os
from PIL import Image
import numpy as np 
from torch.autograd import Variable

def arg_parse():
    desc = "this script is used to load trained model and run on test data set"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--model_dir", type=str, default="Result\model", help="only need to input the sub-dir under current directory")
    parser.add_argument("--model_name",type=str,required=True, help="name of the model")
    parser.add_argument("--test_dir", type=str, default="test_images_64",help="directory of all test images")
    parser.add_argument("--pred_dir",type=str, default="pred_images", help="directory of predicted result")
    return parser.parse_args()

def main():
    args = arg_parse()

    model_dir = os.path.join(os.getcwd(), args.model_dir, args.model_name)
    print(model_dir)
    if not os.path.exists(model_dir):
        print("the current model directory does not exist, please re-enter the path")
        return 
    print("start loading model...")
    net = model.NetModel(64,32)
    net.load_state_dict(torch.load(model_dir))
    print("model loaded successfully!")

    img_dir = os.path.join(os.getcwd(), args.test_dir)
    if not os.path.exists(img_dir):
        print("the current path of test images does not exist, please re-enter the path")
        return 

    files = os.listdir(img_dir)
    cnt = 0

    print("start predict the images...")
    start_pred = time.perf_counter()
    for file in files:
        img = Image.open(os.path.join(img_dir,file))
        img = img.resize((128,128), Image.BICUBIC)
        input_img = torch.FloatTensor(np.asarray(img)[:,:,0]).view(1,1,128,128)
        input_img = Variable(input_img)
        outputs = net(input_img)

        pred_img_dir = os.path.join(os.getcwd(), args.pred_dir)
        if not os.path.exists(pred_img_dir):
            os.makedirs(pred_img_dir)

        
        im_pred = Image.fromarray(np.asarray(outputs.data[0][0]))
        im_pred = im_pred.convert("L")
        im_pred.save(os.path.join(pred_img_dir, file))

        cnt += 1
        if cnt % 200 == 0:
            print("completed %s images" % cnt)

    end_pred = time.perf_counter()
    print("all images completed! time used: %.4f s" % (end_pred - start_pred))


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("******************")
    print("All completed! Total time:%.4f s" % (end-start))
    print("******************")
