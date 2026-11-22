#!/usr/bin/env bash
# Build the inference image and push it to Amazon ECR, in the current AWS account and region.
set -euo pipefail

image=sagemaker-moment-anomaly-detection
tag=latest

# AWS account and region
account=$(aws sts get-caller-identity --query Account --output text)
region=${AWS_REGION:-$(aws configure get region || true)}
if [ -z "${region}" ]; then
    echo "No AWS region: set AWS_REGION or run 'aws configure'." >&2
    exit 1
fi
registry="${account}.dkr.ecr.${region}.amazonaws.com"
fullname="${registry}/${image}:${tag}"

# Create the ECR repository if it does not exist
aws ecr describe-repositories --repository-names "${image}" --region "${region}" > /dev/null 2>&1 \
    || aws ecr create-repository --repository-name "${image}" --region "${region}" > /dev/null

# Log in to the account's ECR registry (the base image is pulled from the public ECR registry)
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${registry}"

# Build for x86_64, the architecture of SageMaker's CPU and GPU instances, with the
# repository root as the build context
model_dir=$(cd "$(dirname "$0")" && pwd)
docker build --platform linux/amd64 -t "${fullname}" -f "${model_dir}/Dockerfile" "${model_dir}/.."

# Push the image
docker push "${fullname}"
echo "Pushed ${fullname}"
