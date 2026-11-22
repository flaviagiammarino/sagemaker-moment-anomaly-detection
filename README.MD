# Zero-shot time series anomaly detection with the MOMENT foundation model on Amazon SageMaker

This Amazon SageMaker model performs zero-shot time series anomaly detection with the pretrained MOMENT-1-large model.
It implements inference from CSV data and supports both CPU and GPU instances, including multi-GPU instances.
The Docker image is built by extending the AWS Deep Learning Container for PyTorch 2.5.1 inference (Python 3.11) and 
includes software licensed under the [MIT License](https://github.com/moment-timeseries-foundation-model/moment/blob/main/LICENSE).

Full details in the [blog post](https://flaviagiammarino.com/blog/sagemaker-moment-anomaly-detection.html).

<img src=https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/architecture_diagram.png style="width:100%;margin-top:60px;margin-bottom:50px"/>

## Usage

1. **Build and push the inference image.** From the repository root, with Docker and the AWS CLI configured, run

   ```bash
   amazon_sagemaker_model/build_and_push.sh
   ```

   This pushes the image `sagemaker-moment-anomaly-detection:latest` to Amazon ECR, in your current AWS account and region.

2. **Upload the model weights to S3.** Download `config.json` and `model.safetensors` from [AutonLab/MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large), package them with both files at the top level of the archive, and upload the archive to an S3 bucket:

   ```bash
   tar -czf model.tar.gz config.json model.safetensors
   aws s3 cp model.tar.gz s3://<bucket>/moment-1-large/model.tar.gz
   ```

3. **Run the notebook.** [amazon_sagemaker_notebook.ipynb](amazon_sagemaker_notebook.ipynb) creates the model and runs it on the sample data, both with a batch transform job and with a real-time endpoint.

### Input

The request body is a JSON object, with content type `application/json`.
Its `csv` field contains the time series as CSV text in long format, with one row per time step and the columns `id`, the identifier of the time series, `timestamp`, the time stamp, and `value`, the observed value.
The optional `stride` field sets the number of time steps between the starts of consecutive windows, and defaults to 128.

```json
{"csv": "id,timestamp,value\n1,2026-01-01 00:00:00,26\n1,2026-01-01 00:30:00,28\n...", "stride": 128}
```

### Output

The response is CSV text, with content type `text/csv`.
It contains the input columns, sorted by `id` and `timestamp`, and two additional columns: `reconstruction`, the reconstructed value, and `error`, the absolute reconstruction error `|value - reconstruction|`.

## Notes

With a stride of 512, the windows don't overlap, and each time step is reconstructed once.
With a smaller stride, each time step is reconstructed several times, at different positions in the window, and averaging these reconstructions makes them more stable, as they depend less on where the time step falls in the window, for example near its edges, where it only has context on one side.
The cost is more computation, since a smaller stride means more windows: with a stride of 1, there is one window per time step.

The time series must be complete, without missing or infinite values, and must have at least 512 time steps, so that it fills at least one window.
MOMENT can impute missing values, by replacing the masked patches with the mask embedding learned in pretraining, but this repository only implements reconstruction, and doesn't implement imputation.

## References

[1] Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). MOMENT: A family of open time-series foundation models. In *Proceedings of the 41st International Conference on Machine Learning*. [https://doi.org/10.48550/arXiv.2402.03885](https://doi.org/10.48550/arXiv.2402.03885).

[2] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. In *International Conference on Learning Representations*. [https://iclr.cc/virtual/2022/poster/6034](https://iclr.cc/virtual/2022/poster/6034).

[3] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1-67. [https://doi.org/10.48550/arXiv.1910.10683](https://doi.org/10.48550/arXiv.1910.10683).
