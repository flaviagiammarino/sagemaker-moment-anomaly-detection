# Extend the AWS Deep Learning Container for PyTorch 2.5.1 inference (Python 3.11).

# Build from the repository root (see build_and_push.sh), so that the license files are included.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker
FROM ${BASE_IMAGE}

# Python dependencies of modules.py that are not in the base image
RUN pip install --no-cache-dir transformers==4.54.1 safetensors

# The inference code. The inference toolkit imports the module named by SAGEMAKER_PROGRAM
# from the Python path, and uses its model_fn, input_fn, predict_fn and output_fn
COPY amazon_sagemaker_model/inference.py amazon_sagemaker_model/modules.py /opt/ml/code/
ENV PYTHONPATH=/opt/ml/code
ENV SAGEMAKER_PROGRAM=inference.py

# License notices
COPY LICENSE THIRD_PARTY_NOTICES /licenses/
