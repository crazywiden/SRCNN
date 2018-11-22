import os
import argparse
import torch
import time
import shutil
import utils
from model import SRCNN


def parse_args():
	descrip = "Super resolution model"
	parser = argparse.ArgumentParser(description=descrip)
	parser.add_argument("--model", type=str, default="SRCNN", help="select which model to use")
	parser.add_argument("--lr_data_dir", type=str, default="train_images_64",\
		help="dir of Low Resolution images")
	parser.add_argument("--hr_data_dir", type=str, default = "train_images_128", \
		help="dir of High Resolution images")
	parser.add_argument("--n_epoches", type=int, default=5, help="the num of n_epoches")
	parser.add_argument("--batch_size", type=int, default=3, help="batch size")
	parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
	parser.add_argument("--save_res_dir", type=str, default="Result", help="diretory to save results and model")
	parser.add_argument("--res_reciever", type=str, default="widen1226@gmail.com", help="email address to recieve the result")
	parser.add_argument("--validation", type=bool, default=False, help="whether do 10 fold cross validation or not")
	parser.add_argument("--test_percent",type=float, default=0.33, help="determine how large is the test set")
	return parser.parse_args()


def main():
	"""
	something check args here
	"""
	args = parse_args()
	
	if args.model == "SRCNN":
		model = SRCNN(args)

	print("start loading data...")
	t1 = time.perf_counter()
	model.load_data()
	t2 = time.perf_counter()
	print("data loading completed! time used: %.2f s" % (t2-t1))

	print("start training...")
	t3 = time.perf_counter()
	model.train()
	t4 = time.perf_counter()
	print("training process completed! time used: %.2f s" % (t4-t3))

	print("start testing...")
	t5 = time.perf_counter()
	model.test()
	t6 = time.perf_counter()
	print("testing process completed! time used: %.2f s" % (t6-t5))

	print("start test single images...")
	t7 = time.perf_counter()
	model.test_single_img("test_images_64")
	t8 = time.perf_counter()
	print("testing process completed! time used: %.2f s" % (t8-t7))

	# send to result as email
	attachment = "Result"
	shutil.make_archive(attachment, 'zip', args.save_res_dir)
	utils.email_res(reciever = args.res_reciever,
		subject="run_result",content=None,attach=attachment)

if __name__ == '__main__':
	start = time.perf_counter()
	main()
	end = time.perf_counter()
	print("********************")
	print("All completed! Total time:%.2f s" % (end-start))
	print("********************")
