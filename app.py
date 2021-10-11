<<<<<<< HEAD
from flask import  Flask, redirect, url_for, request, render_template
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

#import pandas as pd

#import numpy as np
import cv2
from werkzeug.utils import secure_filename
import warnings
import random
import timm
import torchvision
from torchvision import transforms
#from tqdm import tqdm
import pickle
import os
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
#from gevent.pywsgi import WSGIServer

# Define a flask app


app = Flask(__name__,template_folder = 'templates')
app.config['UPLOAD_FOLDER'] = 'upload/image'

# Model saved with pytorch model.save()
model_path = 'efficientnet_model.pth'



class Model(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super(Model,self).__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
    

    
model = Model('tf_efficientnet_b4_ns',2)

# Load your trained model
model.load_state_dict(torch.load(model_path , map_location=torch.device('cpu')))

#transformation for the data
transform=transforms.Compose([
                              transforms.Resize([64,64]),
                              transforms.ToTensor()
])

img_folder = 'upload'
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    dataset = ImageFolder(img_folder,transform = transform)
    batch_size = len(dataset)
    df = DataLoader(dataset,batch_size = batch_size)
    for i in df:
        img,label = i
        pred = model(img)
        _,preds=torch.max(pred,dim=1)
    if preds[batch_size -1]==0:
        preds="Patient is covid positive"
    elif preds[batch_size - 1]==1:
        preds = "Patient is covid negative"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        if f.filename == '':
            print('No selected file')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
=======
from flask import  Flask, redirect, url_for, request, render_template
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

#import pandas as pd

#import numpy as np
import cv2
from werkzeug.utils import secure_filename
import warnings
import random
import timm
import torchvision
from torchvision import transforms
#from tqdm import tqdm
import pickle
import os
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
#from gevent.pywsgi import WSGIServer

# Define a flask app


app = Flask(__name__,template_folder = 'templates')
app.config['UPLOAD_FOLDER'] = 'upload/image'

# Model saved with pytorch model.save()
model_path = 'efficientnet_model.pth'



class Model(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super(Model,self).__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
    

    
model = Model('tf_efficientnet_b4_ns',2)

# Load your trained model
model.load_state_dict(torch.load(model_path , map_location=torch.device('cpu')))

#transformation for the data
transform=transforms.Compose([
                              transforms.Resize([64,64]),
                              transforms.ToTensor()
])

img_folder = 'upload'
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    dataset = ImageFolder(img_folder,transform = transform)
    batch_size = len(dataset)
    df = DataLoader(dataset,batch_size = batch_size)
    for i in df:
        img,label = i
        pred = model(img)
        _,preds=torch.max(pred,dim=1)
    if preds[batch_size -1]==0:
        preds="Patient is covid positive"
    elif preds[batch_size - 1]==1:
        preds = "Patient is covid negative"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        if f.filename == '':
            print('No selected file')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
>>>>>>> a6b8dc3a5418bbf80b65b24d42cbcd209df6144d
    app.run(debug=True)