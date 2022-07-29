import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 9, test_size = 2500, train_size=7500)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

cls = LogisticRegression(solver="saga", multi_class="multinomial").fit(x_train_scaled, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    im_bw = im_pil.convert("L")
    im_bw_resized = im_bw.resize((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(im_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(im_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(im_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled) / max_pixel

    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
    test_pred = cls.predict(test_sample)
    return test_pred[0]
