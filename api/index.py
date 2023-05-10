import os
import time
import torch
import einops
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from natsort import natsorted
from tensorboardX import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data import *
from utils import warp, update_train_log, write_train_log, update_eval_log, write_eval_log, print_eval_log

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

@app.route('/predict')
def predict():
	# MAIN MODEL
	model_stn = models.resnet50()
	#model_stn.fc = nn.Linear(2048, 8)
	model = models.resnet50()
	#model.fc = nn.Linear(2048, 720)
	resume_path = './models/main.pth'
	stn_resume_path = './models/stn.pth'
	#model.load_state_dict(torch.load(resume_path, map_location=torch.device('cpu')))
	#model_stn.load_state_dict(torch.load(stn_resume_path, map_location=torch.device('cpu')))
	#model_stn.to('cpu')
	#model.to('cpu')

	with torch.no_grad():
		model.eval()
		model_stn.eval()

		#MODEL
		img = cv2.imread("./test.png")
		img = cv2.resize(img, (224, 224))/255
		img = einops.rearrange(img, 'h w c -> c h w')
		img = torch.Tensor(img)
		img = img.float().to('cpu')
		img = torch.unsqueeze(img, 0)

		#STN
		pred_st = model_stn(img)
		pred_st = torch.cat([pred_st,torch.ones(1,1).to('cpu')], 1)
		Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
		img_ = warp(img, Minv_pred)
		pred = model(img_)

		#top 3 predictions
		max_pred = torch.argsort(pred, dim=1, descending=True)
		max_h = max_pred[0] // 60
		max_m = max_pred[0] % 60

		return f"{max_h[0].cpu().numpy()}:{max_m[0].cpu().numpy():0>2}"
        
if __name__ == '__main__':
	app.run(debug=True)