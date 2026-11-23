# Zero-shot time series anomaly detection with the MOMENT foundation model on Amazon SageMaker

This Amazon SageMaker model performs zero-shot time series anomaly detection with the pretrained MOMENT-1-large model [[1]](#references).
It implements inference from CSV data and supports both GPU and multi-GPU instances.
The Docker image is built by extending the AWS Deep Learning Container for PyTorch 2.5.1 GPU inference (Python 3.11, CUDA 12.4) and 
includes software licensed under the [MIT License](https://github.com/moment-timeseries-foundation-model/moment/blob/main/LICENSE).

<img src=https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/architecture_diagram.png style="width:100%;margin-top:60px;margin-bottom:50px"/>

## Deployment

The model is deployed in three steps. 

1. **Build the inference image and push it to Amazon ECR**

   From the repository root, with Docker and the AWS CLI configured, run

   ```bash
   container/build_and_push.sh
   ```

   which pushes the image `sagemaker-moment-anomaly-detection:latest` to the ECR registry of your current AWS account and region.

2. **Upload the model weights to S3**

   Download `config.json` and `model.safetensors` from [AutonLab/MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large), 
   package them in an archive with both files at its top level, and upload the archive to an S3 bucket in the same region:

   ```bash
   tar -czf model.tar.gz config.json model.safetensors
   aws s3 cp model.tar.gz s3://<s3-bucket-name>/moment-1-large/model.tar.gz
   ```

3. **Create a SageMaker model from the inference image in ECR and the model weights in S3**

   When the model runs, SageMaker pulls the image from ECR and extracts the archive into `/opt/ml/model` in the container, 
   from which the inference script loads the weights.
   As the container doesn't need any network access, the model can be created with network isolation enabled.
   The same model can then be used for batch inference, with a batch transform job, for real-time inference, with an endpoint, 
   and for asynchronous inference, with an asynchronous endpoint, as shown in the [notebook](notebook.ipynb).

## Batch inference

A batch transform job reads each input file from S3, sends its content to the model as one request, and writes the response to S3, 
in a file with the same name and the `.out` extension.

### Input

Each input file contains CSV text, with content type `text/csv`.
It contains the time series in long format, with one row per time step, and the columns `id`, 
the identifier of the time series, `timestamp`, the time stamp, and `value`, the observed value.

```csv
id,timestamp,value
A,2026-01-01 00:00:00,26
A,2026-01-01 00:30:00,28
...
B,2026-01-01 00:00:00,45
B,2026-01-01 00:30:00,45
...
C,2026-01-01 00:00:00,69
C,2026-01-01 00:30:00,61
...
```

### Output

Each output file contains CSV text, with content type `text/csv`.
It contains the `id`, `timestamp` and `value` columns from the input, and two additional columns: `reconstruction`, 
the reconstructed value, and `error`, the absolute reconstruction error `|value - reconstruction|`.

```csv
id,timestamp,value,reconstruction,error
A,2026-01-01 00:00:00,26,21.56,4.44
A,2026-01-01 00:30:00,28,22.41,5.59
...
B,2026-01-01 00:00:00,45,40.67,4.33
B,2026-01-01 00:30:00,45,41.46,3.54
...
C,2026-01-01 00:00:00,69,61.50,7.50
C,2026-01-01 00:30:00,61,62.42,1.42
...
```

## Real-time inference

An endpoint keeps the model loaded on one or more instances and returns the response to each request directly.
The request and response are the same CSV text used for batch inference.

## Asynchronous inference

An asynchronous endpoint keeps the model loaded like a real-time endpoint, but queues the requests and processes them in the background.
Each request body is uploaded to S3 as a file, and the endpoint is invoked with its S3 location: the invocation returns immediately 
with the S3 location where the endpoint will write the response, and the S3 location where it will write the error if the request fails.
The request and the response are the same CSV text used for batch and real-time inference.

## Notes

- Each time series is split into windows of 512 time steps that start every 128 time steps, so that consecutive windows overlap.
  Each time step is therefore reconstructed several times, at different positions in the window, and averaging these reconstructions makes them more stable, 
  as they depend less on where the time step falls in the window, for example near its edges, where it only has context on one side.
- The time series must be complete, without missing or infinite values, and must have at least 512 time steps, so that it fills at least one window.
  MOMENT can impute missing values, but this is not implemented in this repository, as the anomaly score is the difference between the observed and the reconstructed values, 
  and a missing value, which has no observed value, has no anomaly score.

## References

[1] Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). MOMENT: A family of open time-series foundation models. In *Proceedings of the 41st International Conference on Machine Learning*. [https://doi.org/10.48550/arXiv.2402.03885](https://doi.org/10.48550/arXiv.2402.03885).
