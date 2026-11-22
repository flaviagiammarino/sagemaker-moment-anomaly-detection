# Zero-shot time series anomaly detection with the MOMENT foundation model on Amazon SageMaker

This Amazon SageMaker model performs zero-shot time series anomaly detection with the pretrained MOMENT-1-large model.
It implements inference from CSV data and supports both GPU and multi-GPU instances.
The Docker image is built by extending the AWS Deep Learning Container for PyTorch 2.5.1 inference (Python 3.11) and 
includes software licensed under the [MIT License](https://github.com/moment-timeseries-foundation-model/moment/blob/main/LICENSE).

Full details in the [blog post](https://flaviagiammarino.com/blog/sagemaker-moment-anomaly-detection.html).

<img src=https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/architecture_diagram.png style="width:100%;margin-top:60px;margin-bottom:50px"/>

## Deployment

The model is deployed in three steps. 

1. **Build the inference image and push it to Amazon ECR**

   From the repository root, with Docker and the AWS CLI configured, run

   ```bash
   amazon_sagemaker_model/build_and_push.sh
   ```

   which pushes the image `sagemaker-moment-anomaly-detection:latest` to the ECR registry of your current AWS account and region.

2. **Upload the model weights to S3**

   Download `config.json` and `model.safetensors` from [AutonLab/MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large), 
   package them in an archive with both files at its top level, and upload the archive to an S3 bucket in the same region:

   ```bash
   tar -czf model.tar.gz config.json model.safetensors
   aws s3 cp model.tar.gz s3://<bucket>/moment-1-large/model.tar.gz
   ```

3. **Create a SageMaker model from the image and the archive**

   When the model runs, SageMaker pulls the image from ECR and extracts the archive into `/opt/ml/model` in the container, 
   from which the inference script loads the weights.
   As the container doesn't need any network access, the model can be created with network isolation enabled.
   The same model can then be used both for batch inference, with a batch transform job, and for real-time inference, 
   with an endpoint, as shown in the [notebook](amazon_sagemaker_notebook.ipynb).

## Batch inference

A batch transform job reads each input file from S3, sends its content to the model as one request, and writes the response to S3, 
in a file with the same name and the `.out` extension.
Each input file contains a JSON object, with content type `application/json`, and with the following structure:

```json
{"csv": "id,timestamp,value\n1,2026-01-01 00:00:00,26\n1,2026-01-01 00:30:00,28\n...", "stride": 128}
```

The `csv` field contains the time series as CSV text in long format, with one row per time step and the columns `id`, 
the identifier of the time series, `timestamp`, the time stamp, and `value`, the observed value.
The optional `stride` field sets the number of time steps between the starts of consecutive windows, and defaults to 128.

Each output file contains CSV text, with content type `text/csv`.
It contains the input columns, sorted by `id` and `timestamp`, and two additional columns: `reconstruction`, the reconstructed value, and `error`, the absolute reconstruction error `|value - reconstruction|`.

## Real-time inference

An endpoint keeps the model loaded on one or more instances and returns the response to each request directly.
The request body is the same JSON object used for batch inference, with content type `application/json`, and the response is the same CSV text, with content type `text/csv`.

## Notes

With a stride of 512, the windows don't overlap, and each time step is reconstructed once.
With a smaller stride, each time step is reconstructed several times, at different positions in the window, and averaging these reconstructions makes them more stable, as they depend less on where the time step falls in the window, for example near its edges, where it only has context on one side.
The cost is more computation, since a smaller stride means more windows: with a stride of 1, there is one window per time step.

The time series must be complete, without missing or infinite values, and must have at least 512 time steps, so that it fills at least one window.
MOMENT can impute missing values, but this is not implemented in this repository, as the anomaly score is the difference between the observed and the reconstructed values, and a missing value, which has no observed value, has no anomaly score.
