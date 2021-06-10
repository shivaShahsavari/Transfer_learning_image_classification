from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import numpy as np

def resnet50_pred(img_path):

    img_macaw = img_path
    img = image.load_img(img_macaw, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x_macaw = preprocess_input(x)
    model = ResNet50(weights='imagenet',include_top = True)
    pred_resnet = model.predict(x_macaw)
    # decode the results into a list of tuples (class, description, probability)
    results=decode_predictions(pred_resnet, top=3)[0]
    for result in results:
        print(result)


resnet50_pred('image/macaw_bird.jpg')
'''
('n01818515', 'macaw', 0.9997813)
('n01820546', 'lorikeet', 0.0001001241)
('n01580077', 'jay', 5.0068047e-05)
'''