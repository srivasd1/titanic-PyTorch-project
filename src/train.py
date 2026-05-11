import pandas as pd
import torch
import joblib
import json

from sklearn.preprocessing import StandardScaler
from model import TitanicModel
from preprocess import Preprocessor
from sklearn.model_selection import train_test_split

from  torch.utils.data import DataLoader, TensorDataset

with open("./artifacts/config.json") as f:
	config = json.load(f)
print(config)

df = pd.read_csv("./data/train.csv")
y = df["Survived"]
X = df.drop("Survived", axis=1)

#
preprocessor = Preprocessor(config)
X = preprocessor.transform(X)
#scaling
scaler = StandardScaler()
X[["Age", "Fare"]] = scaler.fit_transform(X[["Age", "Fare"]])

joblib.dump(scaler, "./artifacts/scaler.pkl")

#tensor
X_tr, X_val, y_tr, y_val = train_test_split(X.values, y.values, test_size = 0.2, random_state=42, stratify = y.values)

X_tr = torch.tensor(X_tr, dtype = torch.float32)
X_val = torch.tensor(X_val, dtype = torch.float32)
y_tr = torch.tensor(y_tr, dtype = torch.float32).view(-1, 1)
y_val = torch.tensor(y_val, dtype = torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_tr, y_tr)
val_dataset = TensorDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = 64, shuffle = False)

#model
model = TitanicModel(X.shape[1])

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

#training loop
epochs = 20
for epoch in range(epochs):
	model.train()
	train_loss = 0
	for X_batch, y_batch in train_dataloader:
		optimizer.zero_grad()
		outputs = model(X_batch)
		loss = loss_fn(outputs, y_batch)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()

	train_loss_avg = train_loss / len(train_dataloader)
	model.eval()
	val_loss = 0
	correct = 0
	total = 0
	for X_batch, y_batch in val_dataloader:
		outputs = model(X_batch)
		loss = loss_fn(outputs, y_batch)
		val_loss += loss.item()
		probs = torch.sigmoid(outputs)
		preds = (probs > 0.5).float()
		correct += (preds == y_batch).sum().item()
		total += y_batch.size(0)
	val_loss_avg = val_loss / len(val_dataloader)
	accuracy = correct / total
	print(f"Epoch [{epoch+1}/{epochs}] | " f"Train Loss: {train_loss_avg:.4f} | "  f"Val Loss: {val_loss_avg:.4f} |" f"Accuracy: {accuracy:.4f}")
# save model
torch.save(model.state_dict(), "artifacts/model.pth")
