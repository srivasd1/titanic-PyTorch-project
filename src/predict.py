import pandas as pd
import torch
import json
import joblib
from model import TitanicModel
from preprocess import Preprocessor

#load config
with open("artifacts/config.json") as f:
	config = json.load(f)
#Load scaler
scaler = joblib.load("artifacts/scaler.pkl")
#load model

model = TitanicModel(len(config["feature_columns"]))
model.load_state_dict(torch.load("artifacts/model.pth"))
model.eval()

#load test data
df = pd.read_csv("data/test.csv")
ids =df["PassengerId"]

# preprocess
preprocessor = Preprocessor(config)
df = preprocessor.transform(df)

#Scale
df[["Age", "Fare"]] = scaler.transform(df[["Age", "Fare"]])

#tensor
X = torch.tensor(df.values, dtype=torch.float32)

#predict
with torch.no_grad():
	preds = (torch.sigmoid(model(X)) > 0.5).int()
#
submission = pd.DataFrame({
		"PassengerId": ids,
		"Survived": preds.numpy().ravel()
})

submission.to_csv("submission.csv", index =False)
print("Done !")
