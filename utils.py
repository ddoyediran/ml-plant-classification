# Utils for preprocessing data etc 
#import tensorflow as tf
#import googleapiclient.discovery
#from google.api_core.client_options import ClientOptions

base_classes = ['healthy',
 'multiple_diseases',
 'rust',
 'scab'
 ]


def load_and_prep_image(filename, img_shape = 150, rescale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    return (224, 224, 3).
    """
    
    #img = tf.io.decode_image(filename, channels=3)
    #img = tf.image.resize(img, [img_shape, img_shape])

    #if rescale:
    #    return img / 255
    #else:
    #    return img


def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and returns 
    logger dictionary.
    """
    logger = {
        "image": image,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }   
    return logger