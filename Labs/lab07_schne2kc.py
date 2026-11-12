import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold


df = pd.read_csv("PIMA_diabetes_lab.csv")
predictors = ["Pregnancies", "Glucose", "BloodPressure", "BMI",
              "DiabetesPedigreeFunction", "Age"]          # 6 predictors
x = df[predictors].to_numpy(dtype=float)
y = df["Outcome"].to_numpy(dtype=float)

## Pt 2, by hand PyTorch code

K = 10
dim_input = 6
dim_output = 1

def build_model():
    """50 x 50 network: 6 -> 50 -> ReLU -> 50 -> ReLU -> 1 (a single logit)."""
    return nn.Sequential(nn.Linear(dim_input, K),
                         nn.ReLU(),
                         nn.Linear(K, K),
                         nn.ReLU(),
                         nn.Linear(K, dim_output))


def to_t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32).reshape(len(a), -1)
print(build_model())

def standardize(X_train, *others):
    """Scale using the TRAINING set's mean and standard deviation only."""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    return [(X_train - mu) / sd] + [(o - mu) / sd for o in others]


def train(X_tr_t, y_tr_t, X_val_t=None, y_val_t=None,
          epochs=5000, lr=0.001, seed=0):
    """Full-batch Adam on log loss. Returns model and per-epoch train/val loss."""
    torch.manual_seed(seed)
    model = build_model()
    loss_fn = nn.BCEWithLogitsLoss()  # log loss (on logits)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_hist, val_hist = [], []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        tr_hist.append(loss.item())
        if X_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_hist.append(loss_fn(model(X_val_t), y_val_t).item())
    return model, tr_hist, val_hist


def predict_prob(m, a):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(to_t(a))).numpy().ravel()


# ------------------------------------------------------------------
# Q4: 70/30 split, standardize, Adam (lr=0.001), 1000 epochs, log loss
# ------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(x, y, train_size=0.7, random_state=1)
X_tr_s, X_val_s = standardize(X_tr, X_val)

X_tr_t, X_val_t = to_t(X_tr_s), to_t(X_val_s)
y_tr_t, y_val_t = to_t(y_tr), to_t(y_val)

epochs, lr = 500, 0.001
model, train_err, val_err = train(X_tr_t, y_tr_t, X_val_t, y_val_t,
                                  epochs=epochs, lr=lr, seed=0)

# (a) training and validation error vs. epoch, one figure
ep = np.arange(1, epochs + 1)
plt.figure(figsize=(8, 5))
plt.plot(ep, train_err, label="Training error")
plt.plot(ep, val_err, label="Validation error")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.title("Training and validation error vs. epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("train_val_error.png", dpi=150)
plt.show()

# (b) epoch with the smallest validation error
best_epoch = int(np.argmin(val_err)) + 1  # epochs are 1-indexed
best_val = float(np.min(val_err))
print(f"(b) Smallest validation error: {best_val:.4f} at epoch {best_epoch}")


# Q5: 5-fold cross-validated AUC of the same network
kf = KFold(n_splits=5, shuffle=True, random_state=1)
fold_aucs = []
for fold, (tr_idx, te_idx) in enumerate(kf.split(x), start=1):
    Xtr_s, Xte_s = standardize(x[tr_idx], x[te_idx])  # stats from this fold's training part only
    m, _, _ = train(to_t(Xtr_s), to_t(y[tr_idx]), epochs=epochs, lr=lr, seed=0)
    auc = roc_auc_score(y[te_idx], predict_prob(m, Xte_s))
    fold_aucs.append(auc)
    print(f"Fold {fold}: AUC = {auc:.4f}")

print(f"5-fold CV-AUC (mean of folds): {np.mean(fold_aucs):.4f} "
      f"(sd {np.std(fold_aucs, ddof=1):.4f})")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

df_lr = pd.read_csv("PIMA_diabetes_lab.csv")
X_lr = df_lr.drop(columns="Outcome").to_numpy(dtype=float)
y_lr = df_lr["Outcome"].to_numpy().astype(int)

lr_aucs = []
for tr_idx, te_idx in KFold(n_splits=5, shuffle=True, random_state=1).split(X_lr):
    scaler = StandardScaler().fit(X_lr[tr_idx])  # training fold only
    clf = LogisticRegression(C=1e10, max_iter=5000)  # huge C = effectively no penalty
    clf.fit(scaler.transform(X_lr[tr_idx]), y_lr[tr_idx])
    p = clf.predict_proba(scaler.transform(X_lr[te_idx]))[:, 1]
    lr_aucs.append(roc_auc_score(y_lr[te_idx], p))

print("Logistic regression fold AUCs:", np.round(lr_aucs, 4))
print(f"Logistic regression 5-fold CV-AUC: {np.mean(lr_aucs):.4f}")

########### BELOW CODE IS TO DRAW A PICTURE OF THE NN MODEL, UNCOMMENT TO GET IMAGE ##########
## Code to draw the Neural Network
# from matplotlib import pyplot
# class Neuron():
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def draw(self, neuron_radius):
#         circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
#         pyplot.gca().add_patch(circle)
#
#
# class Layer():
#     def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
#         self.vertical_distance_between_layers = 100
#         self.horizontal_distance_between_neurons = 10
#         self.neuron_radius = 1.0
#         self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
#         self.previous_layer = self.__get_previous_layer(network)
#         self.y = self.__calculate_layer_y_position()
#         self.neurons = self.__intialise_neurons(number_of_neurons)
#
#     def __intialise_neurons(self, number_of_neurons):
#         neurons = []
#         x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
#         for iteration in range(number_of_neurons):
#             neuron = Neuron(x, self.y)
#             neurons.append(neuron)
#             x += self.horizontal_distance_between_neurons
#         return neurons
#
#     def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
#         return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2
#
#     def __calculate_layer_y_position(self):
#         if self.previous_layer:
#             return self.previous_layer.y + self.vertical_distance_between_layers
#         else:
#             return 0
#
#     def __get_previous_layer(self, network):
#         if len(network.layers) > 0:
#             return network.layers[-1]
#         else:
#             return None
#
#     def __line_between_two_neurons(self, neuron1, neuron2):
#         angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
#         x_adjustment = self.neuron_radius * sin(angle)
#         y_adjustment = self.neuron_radius * cos(angle)
#         line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
#         pyplot.gca().add_line(line)
#
#     def draw(self, layerType=0):
#         for neuron in self.neurons:
#             neuron.draw( self.neuron_radius )
#             if self.previous_layer:
#                 for previous_layer_neuron in self.previous_layer.neurons:
#                     self.__line_between_two_neurons(neuron, previous_layer_neuron)
#         # write Text
#         x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
#         if layerType == 0:
#             pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
#         elif layerType == -1:
#             pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
#         else:
#             pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)
#
# class NeuralNetwork():
#     def __init__(self, number_of_neurons_in_widest_layer):
#         self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
#         self.layers = []
#         self.layertype = 0
#
#     def add_layer(self, number_of_neurons ):
#         layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
#         self.layers.append(layer)
#
#     def draw(self):
#         pyplot.figure()
#         for i in range( len(self.layers) ):
#             layer = self.layers[i]
#             if i == len(self.layers)-1:
#                 i = -1
#             layer.draw( i )
#         pyplot.axis('scaled')
#         pyplot.axis('off')
#         pyplot.title( 'Neural Network architecture', fontsize=15 )
#         pyplot.show()
#
# class DrawNN():
#     def __init__( self, neural_network ):
#         self.neural_network = neural_network
#
#     def draw( self ):
#         widest_layer = max( self.neural_network )
#         network = NeuralNetwork( widest_layer )
#         for l in self.neural_network:
#             network.add_layer(l)
#         network.draw()
#
# network = DrawNN( [6,50,50,1] )
# network.draw()


