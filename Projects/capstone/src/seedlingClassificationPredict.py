import glob
from PIL import Image, ImageOps
from keras.models import Model, load_model
import pandas as pd
import numpy as np

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
NUM_CATEGORIES = len(CATEGORIES)


# Resize the test data
DIM =128
z = glob.glob('../data/TEST1/test/*.png')
test_imgs = []
names = []
for fn in z:                        # read all .png files and resize them to (DIM,DIM)
    if fn[-3:] != 'png':
        continue
    names.append(fn.split('/')[-1])
    new_img = Image.open(fn)
    test_img = ImageOps.fit(new_img, (DIM, DIM), Image.ANTIALIAS).convert('RGB') # Image.ANTIALIAS (a high-quality downsampling filter)
    test_imgs.append(test_img)

# Load the trained Model
#todo
#model = load_model('../model/CNN_AWS_TEST2-dim128.h5')
#model = load_model('../model/CNN_AWS_TEST3-NVIDIA-add-bn.h5')
#model = load_model('../model/CNN_AWS_TEST3-NVIDIA-bn-gap-33-batch800.h5')
model = load_model('../model/CNN_AWS_TEST3-NVIDIA-bn-gap-33-batch800-deep.h5')


#Test the Model with Kaggle Test datasets

timgs = np.array([np.array(im) for im in test_imgs])
testX = timgs.reshape(timgs.shape[0], DIM, DIM, 3) / 255

yhat = model.predict(testX)

# a submission .csv file need to be created based on the formate defined by Kaggle
# check the submission formate here: https://www.kaggle.com/c/plant-seedlings-classification/data
preds = []
for i in range(len(yhat)):
    pos = np.argmax(yhat[i])
    preds.append(CATEGORIES[pos])

df = pd.DataFrame(data={'file': names, 'species': preds})

df.head(5)

df_sort = df.sort_values(by=['file'])

#todo
#df_sort.to_csv('../submission/CNNResults-AWS-TEST2-dim128.csv', index=False) # 0.81360 score
#df_sort.to_csv('../submission/CNNResults-AWS-TEST3-NVIDIA-add-bn.csv', index=False) # 0.92821
#df_sort.to_csv('../submission/CNNResults-AWS-TEST3-NVIDIA-add-bn-gap-33-batch800.csv', index=False)
df_sort.to_csv('../submission/CNN_AWS_TEST3-NVIDIA-bn-gap-33-batch800-deep.csv', index=False)


#====================================================================================================

# Evaluate the final Model
# Dataset: ImagesFromTheWild
z = glob.glob('../data/ImagesFromTheWild/*/*.tiff')
ori_label = []
ori_imgs = []
DIM = 128
for fn in z:
    if fn[-4:] != 'tiff':
        continue
    ori_label.append(fn.split('/')[-2])
    new_img = Image.open(fn)
    ori_imgs.append(ImageOps.fit(new_img, (DIM, DIM), Image.ANTIALIAS).convert('RGB'))


from sklearn.preprocessing import LabelBinarizer
imgs = np.array([np.array(im) for im in ori_imgs])
imgs = imgs.reshape(imgs.shape[0], DIM, DIM, 3) / 255
lb = LabelBinarizer().fit(CATEGORIES)
label = lb.transform(ori_label)

score = model.evaluate(x=imgs, y=label)
print('evaluation: {}'.format(score))


# Evaluate the final Model
# Dataset: Validatation data split from the training datasets

z = glob.glob('../data/val/*/*.png')
ori_label2 = []
ori_imgs2 = []
DIM = 128
for fn in z:
    if fn[-3:] != 'png':
        continue
    ori_label2.append(fn.split('/')[-2])
    new_img2 = Image.open(fn)
    ori_imgs2.append(ImageOps.fit(new_img2, (DIM, DIM), Image.ANTIALIAS).convert('RGB'))

from sklearn.preprocessing import LabelBinarizer
imgs2 = np.array([np.array(im) for im in ori_imgs2])
imgs2 = imgs2.reshape(imgs2.shape[0], DIM, DIM, 3) / 255
lb2 = LabelBinarizer().fit(CATEGORIES)
label2 = lb2.transform(ori_label2)
score = model.evaluate(x=imgs2, y=label2)
print('validation: {}'.format(score))
