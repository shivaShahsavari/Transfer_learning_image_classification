from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
from tensorflow.keras.models import Model

import numpy as np

def vgg19_pred(img_path):
    img_macaw = img_path
    img = image.load_img(img_macaw, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x_macaw_vgg19 = preprocess_input(x)
    base_model = VGG19(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    pred_vgg19 = model.predict(x_macaw_vgg19)
    results=decode_predictions(pred_vgg19, top=3)[0]
    for result in results:
        print(result)

    #return results

vgg19_pred('image/macaw_bird.jpg')
