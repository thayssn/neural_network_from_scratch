import numpy as np
import pandas as pd
from main import forward_propagation, get_predictions
import matplotlib.pyplot as plt

test_data = pd.read_csv('./test.csv')
test_data = np.array(test_data);
test_data = test_data[1:].T
test_data = test_data / 255


def make_prediction(input):
    params = np.load("model_params.npz")
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]
    _,_,_,a2 = forward_propagation(input, w1, b1, w2, b2)
    predictions = get_predictions(a2)
    return predictions


def test_prediction(index):
    current = test_data[:,index, None]
    prediction = make_prediction(current)
    print(f"prediction: {prediction}")
    current_image = current.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title(f"prediction: {prediction}")

plt.show()
for i in range(100):
    test_prediction(np.random.randint(0, 1000))
    plt.pause(1)
    plt.clf()

