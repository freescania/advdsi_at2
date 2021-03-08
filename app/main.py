from fastapi import FastAPI
from starlette.responses import JSONResponse
#import pandas as pd
import torch

app = FastAPI()

#redefine model class
class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()

        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 104)  # from 4, 104 number of classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)

num_features = 4
model = TheModelClass(num_features) #*args, **kwargs
model.load_state_dict(torch.load('../models/NN_4feat.pt'))
model.eval()

# Inside the main.py file, create a function called read_root() that will return a
# dictionary with Hello as key and World as value. Add a decorator to it in order
# to add a GET endpoint to app on the root
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Inside the main.py file, create a function called healthcheck() that will
# return GMM Clustering is all ready to go!. Add a decorator to it in order
# to add a GET endpoint to app on /health with status code 200
@app.get('/health', status_code=200)
def healthcheck():
    return 'Neural Network model is all ready to go!'

#Inside the main.py file, create a function called format_features() with
# genre, age, income and spending as input parameters that will return a
# dictionary with the names of the features as keys and the inpot parameters as lists
def format_features(review_aroma: int,	review_appearance: int, review_palate: int, review_taste: int):
  return {
        'review aroma (1-5)': [review_aroma],
        'review appearance (1-5)': [review_appearance],
        'review palate (1-5)': [review_palate],
        'review taste (1-5)': [review_taste]
    }

#Inside the main.py file, Define a function called predict with the following logics:
#input parameters: review_aroma, review_appearance, review_palate, review_taste
#logics: format the input parameters as dict, convert it to a dataframe and make prediction with gmm_pipe
#output: prediction as json
#Add a decorator to it in order to add a GET endpoint to app on /mall/customers/segmentation
@app.get("/beer/type/")
def predict(review_aroma: int,	review_appearance: int, review_palate: int, review_taste: int):
    features = format_features(review_aroma, review_appearance, review_palate, review_taste)
    obs = torch.Tensor(np.array(features))
    #change for NN
    output = model(obs)
    return JSONResponse(output.tolist())

@app.get("/model/architecture/)
def architecture():
    return 'The model architecture is two linear layers, the first with a Relu activation function' \
           'and then a final softmax layer to help generate the prediction'
