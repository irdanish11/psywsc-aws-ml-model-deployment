import json
import sagemaker
from sagemaker.predictor import Predictor
import json
import requests
import numpy as np



def get_label(result):
    object_categories = {}
    with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
        for line in f:
            key, val = line.strip().split(":")
            object_categories[key] = val.strip(" ").strip(",")
    pred_label = object_categories[str(np.argmax(result))]
    score = str(np.amax(result))[:5]
    print(
        "The label is",
        pred_label,
        "with probability",
        score
    )
    return pred_label, score


def get_payload(img_url):
    resp = requests.get(img_url)
    payload = resp.content
    return payload
    

def get_sagemaker_session():
    role = "sagemaker-execution-role"
    bucket = "eoe-sagemaker-bucket"
    sess = sagemaker.Session()
    return sess
    


def lambda_handler(event, context):
    sess = get_sagemaker_session()
    print(event)
    print(event["body"])
    body = json.loads(event["body"])
    img_url = body["img_url"]
    payload = get_payload(img_url)

    predictor = Predictor(endpoint_name="classification-model", sagemaker_session=sess)
    response = predictor.predict(payload)
    result = json.loads(response.decode())
    print("Most likely class: {}".format(np.argmax(result)))
    label, score = get_label(result)
    
    
    return {
        'statusCode': 200,
        'body': json.dumps({"class": label, "score": score})
    }
