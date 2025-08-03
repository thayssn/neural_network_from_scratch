import pandas as pd
import numpy as np
import utils as u
import matplotlib.pyplot as plt
import main as m

train_data = pd.read_csv('./train.csv')
train_data = np.array(train_data);
data = train_data.T
pixels_size, samples_size = data.shape
labels = data[0]
pixels = data[1:pixels_size]
pixels = pixels/255

w1,b1,w2,b2 = m.gradient_descent(pixels, 500, 0.1, labels)

def save_training_data(w1, b1, w2, b2):
    np.savez("model_params.npz", w1=w1, b1=b1, w2=w2, b2=b2)

save_training_data(w1, b1, w2, b2)
