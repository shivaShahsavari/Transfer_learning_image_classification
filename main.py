from resenet50_prediction import resnet50_pred
from vgg16_prediction import vgg16_pred
from vgg19_prediction import vgg19_pred
import numpy as np


img_macaw = 'image/macaw_bird.jpg'
res_result=resnet50_pred(img_macaw)
print('Resnet50 result:', res_result)

vgg16_result=vgg16_pred(img_macaw)
print('VGG16 result:', vgg16_result)

vgg19_result=vgg19_pred(img_macaw)
print('VGG19 result:', vgg19_result)

