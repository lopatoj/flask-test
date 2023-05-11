import torch
import cv2
import numpy as np
import einops
import torch.nn as nn
import torchvision.models as models
from kornia.geometry.transform import warp_perspective

from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return 'inference at /predict endpoint'

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		bytes = request.get_data()

		# PREPROCESSING
		img = cv2.imdecode(np.frombuffer(bytes, np.uint8), -1)
		img = cv2.resize(img, (224, 224))/255
		img = einops.rearrange(img, 'h w c -> c h w')
		img = torch.Tensor(img)
		img = img.float().to('cpu')
		img = torch.unsqueeze(img, 0)
		
		# MAIN MODEL
		model = models.resnet50()
		model.fc = nn.Linear(2048, 720)
		model.load_state_dict(torch.load('main.pth', map_location=torch.device('cpu')))
		model.to('cpu')

		# STN MODEL
		model_stn = models.resnet50()
		model_stn.fc = nn.Linear(2048, 8)
		model_stn.load_state_dict(torch.load('stn.pth', map_location=torch.device('cpu')))
		model_stn.to('cpu')

		with torch.no_grad():
			model.eval()
			model_stn.eval()

			# PREDICTION
			pred_st = model_stn(img)
			pred_st = torch.cat([pred_st,torch.ones(1,1).to('cpu')], 1)
			Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
			img_ = warp(img, Minv_pred)
			pred = model(img_)

			# POSTPROCESSING
			predictions = torch.argsort(pred, dim=1, descending=True)
			output = predictions[0][0].cpu().numpy()
			hours = output // 60
			minutes = output % 60

			return f"{hours}:{minutes:0>2}"
	
def warp(img, Minv_pred, sz=224):
	device = 'cpu'
	s,t = sz/2., 1.
	Minv_pred = torch.Tensor([[s,0,t*s],[0,s,t*s],[0,0,1]]).to(device) @ Minv_pred @ torch.Tensor([[1/s,0,-t],[0,1/s,-t],[0,0,1]]).to(device)
	img_ = warp_perspective(img, Minv_pred, (sz, sz))
	return img_

if __name__ == '__main__':
	app.run(debug=False)