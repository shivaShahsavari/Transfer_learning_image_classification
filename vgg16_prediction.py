from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np

def vgg16_pred(img_path):
    img_macaw = img_path
    img = image.load_img(img_macaw, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x_macaw_vgg16 = preprocess_input(x)
    model = VGG16(weights='imagenet', include_top=True)
    pred_vgg16 = model.predict(x_macaw_vgg16)
    results=decode_predictions(pred_vgg16, top=3)[0]
    for result in results:
        print(result)

    #return results

vgg16_pred('image/macaw_bird.jpg')

'''
('n01818515', 'macaw', 0.99981827)
('n01843383', 'toucan', 0.00016916859)
('n01829413', 'hornbill', 5.2717146e-06)
'''