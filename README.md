# Simple Neural Network built from Scratch

Exploring the basics of neural networks, activation functions and python. Building intuition for the math!

### About this project

In this project, I'm using the MNIST dataset to train a digit recognition model.

<img src="https://raw.githubusercontent.com/thayssn/neural_network_from_scratch/main/assets/thumb.gif" width="360"/>

#### Resources and recommendations

What inspired me to try this was a video from [SamsonZhang](https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang).

To cover a few gaps left from this tutorial I also used this [Kaggle Notbook](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras) as reference.

And finally, to build the intuition for the math used here I absolutely recommend the amazing channel of [3blue1brown](https://www.youtube.com/@3blue1brown).

#### Dataset

I'm keeping the dataset I used under data.zip for easy access, but you can easily find in other places around the internet as well searching for MNIST dataset.

```
unzip data.zip
```

You will see two separate csv files. One is for the purpose of training our model `train.csv` and the other one `test.csv`, to test it agains numbers that it haven't been trained with yet.

### Scripts

The main logic for our neural network is located under `main.py`

To actually train our model, we use `train.py`, that loads and uses the data from `train.csv`.

The `test.py` file is used to actually test this model against our `test.csv` data and visualize it.

`utils.py` are mostly math functions, so the main file can be a little bit more readable.

The `model_params.npz` file will store the model's parameters so that it can be later loaded into our tests.

### Running

#### Train

```
python train.py
```

During the training process you're able to visualize the evolution of the performance of our model's accuracy.

After this, the `model_params.npz` will be saved.

#### Test

```
python test.py
```

During the test you can visualize the prediction and compare it to the actual image, to check if it matches the expected digit.
