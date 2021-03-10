from fastapi import FastAPI
from starlette.responses import JSONResponse
import numpy as np
#import torch
#import torch.nn as nn

app = FastAPI()

#redefine model class
#class PytorchMultiClass(nn.Module):
    #def __init__(self, num_features):
        #super(PytorchMultiClass, self).__init__()

        #self.layer_1 = nn.Linear(num_features, 32)
        #self.layer_out = nn.Linear(32, 104)  # from 4, 104 number of classes
        #self.softmax = nn.Softmax(dim=1)

    #def forward(self, x):
        #x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        #x = self.layer_out(x)
        #return self.softmax(x)

#num_features = 4
#model = PytorchMultiClass(num_features) #*args, **kwargs
#model.load_state_dict(torch.load('../models/NN_4feat.pt'))
#model.eval()

# Inside the main.py file, create a function called read_root() that will describe the project
@app.get("/")
def read_root():
    return "This project was designed to take a few key features from a beer " \
           "styles data set and then build a model that would take those inputs " \
           "and then predict the given beer style. It takes four inputs (review_aroma, " \
           "review_appearance, review_palate, review_taste) and each one must " \
           "be given an integer score from 0-5 and then it will predict the beer " \
           "style. Here is the github repo: https://github.com/freescania/advdsi_at2"


# Inside the main.py file, create a function called healthcheck() that will
# will state 'Neural Network model is all ready to go!'. Add a decorator to it in order
# to add a GET endpoint to app on /health with status code 200
@app.get('/health', status_code=200)
def healthcheck():
    return 'Neural Network model is all ready to go!'

#Inside the main.py file, create a function called format_features() with
# review_aroma, review_appearance, review_palate and review_taste as input parameters that will return a
# dictionary with the names of the features as keys and the inpot parameters as lists
def format_features(review_aroma: int,	review_appearance: int, review_palate: int, review_taste: int):
  return {
        'review aroma (0-5)': ([review_aroma]/5),
        'review appearance (0-5)': ([review_appearance]/5),
        'review palate (0-5)': ([review_palate]/5),
        'review taste (0-5)': ([review_taste]/5)
    }

#Inside the main.py file, Define a function called predict with the following logics:
#input parameters: review_aroma, review_appearance, review_palate, review_taste
#logics: format the input parameters as dict, convert it to a dataframe and make prediction with gmm_pipe
#output: prediction as json
#Add a decorator to it in order to add a GET endpoint to app on /beer/type
@app.get("/beer/type/")
def predict(review_aroma: int,	review_appearance: int, review_palate: int, review_taste: int):
    features = format_features(review_aroma, review_appearance, review_palate, review_taste)
    feats = np.array(list(features.values())).T
    feats = (feats / 5)
    zeroes = np.zeros((32, 4))
    zeroes[0] = feats
    #obs = torch.Tensor(zeroes)
    #change for NN
    output = ['American IPA']
    return JSONResponse(output.tolist())

@app.get("/model/architecture/")
def architecture():
    return 'The model architecture is four layers: layer_1,  Linear, input=4, output=32,' \
           'layer_2,  Linear, input=32, output=104, layer_3,  Linear, input=104, output=32,' \
           'layer_out, Linear, input=32, output=104,  Softmax activation'
