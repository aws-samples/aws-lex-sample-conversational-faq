## AWSLexCFAQModelDeployThroughSageMaker

This code exemplifies the process of deploying a generative model using a SageMaker Inference endpoint, which can be called to handle the Lex Fallback FAQ intent. 


## Installation

Run the following commands under the root folder of this project:

```bash
conda create --name deploy_through_sagemaker python=3.8
conda activate deploy_through_sagemaker
pip install transformers sagemaker torch
```

## To run the deployment
Before running the below script, one needs to: 
- inject proper isengard credentials 
- create or use existing sagemaker roles (make sure that this role has permission to call Kendra)
- create the model.tar.gz, upload it to your S3.
    ```bash
    cd yourworkspace/AWSLexCFAQModelDeployThroughSageMaker
    mkdir model 
    tar -czf model.tar.gz code/ model/
    aws s3 cp model.tar.gz s3://your_s3_bucket/dev/cfaq/
    ```
- run below to deploy
    ```bash
    SAGEMAKER_ROLE=${YOUR_SAGEMAKER_ROLE} CFAQ_INDEX=${YOUR_INDEX_ID} CFAQ_MODEL_S3="s3://your_s3_bucket/dev/cfaq/model.tar.gz" python package_and_deploy.py
    ```


## Demonstration with Lex V2

Upon completion, you can build a lambda function to access deployed endpoint with Lex V2. Please find the lambda zip in the shared S3 bucket. 

Please set up the follow environment variables under Configuration tab of your lambda function,
- ``CFAQ_INDEX`` - your search index id
- ``ENDPOINT_NAME`` - sagemaker endpoint for CFAQ



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

