import os
import time
import logging
import boto3
import json
import sagemaker
from transformers import BertTokenizer, BertModel
from sagemaker.pytorch import PyTorchModel


client = boto3.client("sagemaker")
endpoint_name = "YOUR-ENDPOINT-cfaq-api-" + time.strftime(
    "%Y-%m-%d-%H-%M-%S", time.gmtime()
)
sagemaker_role = os.environ.get("SAGEMAKER_ROLE", "YOUR_SAGEMAKER_ROLE")
zipped_model_path = os.environ.get("CFAQ_MODEL_S3", "YOUR_S3_URL")
model = PyTorchModel(
    entry_point="main.py",
    model_data=zipped_model_path,
    role=sagemaker_role,
    framework_version="1.10",
    py_version="py38",
)

instance_type = "ml.p3.2xlarge"
predictor = model.deploy(
    initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint_name
)

print(f"endpoint_name: {endpoint_name}")

# Test the deployed endpoint
index_id = os.environ.get("CFAQ_INDEX")
sm = sagemaker.Session().sagemaker_runtime_client
query = "I want to report something."
payload = {
    "query": query,
    "index_id": index_id,
    "index_type": "OpenSearch",  # for instance, OpenSearch or Kendra
    "is_single": True,
    "is_rerank": True,
}
payload = json.dumps(payload)
runtime = boto3.client("runtime.sagemaker")
response = sm.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="text/csv", Body=payload
)
response_string = json.loads(response["Body"].read().decode())["text"]
print(response_string)
