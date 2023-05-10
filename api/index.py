import torch
import einops
import torch.nn as nn
import torchvision.models as models
from data import *
from utils import warp

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/predict')
def predict():

	# MAIN MODEL
	model = models.resnet50()
	model.fc = nn.Linear(2048, 720)
	model.load_state_dict(torch.load('./models/main.pth', map_location=torch.device('cpu')))
	model.to('cpu')

	# STN MODEL
	model_stn = models.resnet50()
	model_stn.fc = nn.Linear(2048, 8)
	model_stn.load_state_dict(torch.load('./models/stn.pth', map_location=torch.device('cpu')))
	model_stn.to('cpu')

	with torch.no_grad():
		model.eval()
		model_stn.eval()

		# IMAGE PREPROCESSING
		img = cv2.imread("./test.png")
		img = cv2.resize(img, (224, 224))/255
		img = einops.rearrange(img, 'h w c -> c h w')
		img = torch.Tensor(img)
		img = img.float().to('cpu')
		img = torch.unsqueeze(img, 0)

		# PREDICTION
		pred_st = model_stn(img)
		pred_st = torch.cat([pred_st,torch.ones(1,1).to('cpu')], 1)
		Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
		img_ = warp(img, Minv_pred)
		pred = model(img_)

		# POSTPROCESSING
		predictions = torch.argsort(pred, dim=1, descending=True)
		hours = predictions[0][0].cpu().numpy() // 60
		minutes = predictions[0][0].cpu().numpy() % 60

		return f"{hours}:{minutes:0>2}"
        
if __name__ == '__main__':
	app.run(debug=True)