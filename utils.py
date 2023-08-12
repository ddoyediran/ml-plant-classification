# Utils for preprocessing data etc 
import tensorflow as tf
import googleapiclient.discovery
#from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform
#from google.cloud.aiplatform import aiplatform


base_classes = ['healthy',
 'multiple_diseases',
 'rust',
 'scab'
 ]

classes_and_models = {
    "model_1": {
        "classes" : base_classes,
        "model_name": "plant_pathology_api",
        "model_id": "3243156880683433984",
        "endpoint_id": "3243156880683433984"
    }
}


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


def predict_json(project, region, endpoint_id, instances, version=None):
    # Send json data to a deployed model for prediction

    # Create the Vertex AI service object
    #prefix = "{}-ml".format(region) if region else "ml"
    #api_endpoint = "http://{}.googleapis.com".format(prefix)
    #client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    #model_path = "projects/{}/models/{}".format(project, model)

    #path = "projects/{}/locations/{}/models/{}".format(project, region, model_id)
    endpoint_path = "projects/{}/locations/{}/endpoints/{}".format(project, region, endpoint_id)

    if version is not None:
        model_path += "/versions/{}".format(version)
    
    # Initialize the Vertex AI SDK to store the common configurations use for the SDk
    aiplatform.init(
        project = project,

        location = region,

        staging_bucket = "plant_pathology_ml_bucket",

        # custom google.auth.credentials.Credentials
        # environment default creds used if not set
        #credentials=google.auth.credentials.Credentials,
    )

    #model = aiplatform.Model(path)
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_path)

    pred = endpoint.predict(instances=instances).predictions

    #endpoint = endpoint.predict(instances=instances)

    return pred





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