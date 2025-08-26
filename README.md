# ML Challenge – Text–Image Similarity

This project provides a solution for computing similarity scores between images and text captions, as per the challenge brief.

## Setup and Execution

1.  **Install Dependencies:** Ensure you have Python 3.10+ and Poetry installed. Then, install the project dependencies:
    ```bash
    poetry install
    ```

2.  **Prepare Data:** Place the `challenge_set.csv` file in the root of the project directory.

3.  **Run the Scoring Pipeline:** Execute the main script:
    ```bash
    poetry run python execute_text_image_similarity_csv.py
    ```

This will generate a `challenge_scored.csv` file with `similarity` and `error` columns.

---

## Question 1: Code to Compute the Similarity Metric

The code is packaged as a reusable Python library in the `text_image_similarity` directory.

### Core Logic

The main logic is in the `score_pairs` function, which takes an iterable of `Record` objects and returns an iterable of `ScoreResult` objects. It performs the following steps:
1.  **Concurrent Image Downloading:** Downloads images in parallel using a `ThreadPoolExecutor`.
2.  **Preprocessing:** Prepares images and text for the `openai/clip-vit-base-patch32` model.
3.  **Embedding Generation:** Generates vector embeddings for images and text.
4.  **Similarity Calculation:** Computes the cosine similarity, normalizes it to a 0-1 range, and rounds to four decimal places.

### Execution Flow

The diagram below illustrates the flow for the provided CSV script.

![Project Data Flow](docs/images/data_flow.png)

---

## Question 2: Time/Memory Footprint and Optimisation Strategies

-   **Memory:** The CLIP model requires ~600 MB of RAM/VRAM. The application itself has a low, stable memory footprint due to chunked/streaming data processing.
-   **Time:** The process is I/O-bound by image downloading and compute-bound by model inference.
-   **Optimisations:**
    -   **Concurrent I/O (Implemented):** A `ThreadPoolExecutor` downloads images in parallel.
    -   **Batching (Implemented):** Records are processed in batches to leverage hardware parallelism.
    -   **Hardware Acceleration & FP16 (Supported):** The code will auto-detect and use GPUs (CUDA/MPS) and supports FP16 for faster inference on CUDA.

---

## Question 3: Packaging and Deployment at Scale

### Packaging

The application is packaged as a standard Python library using **Poetry**. Running `poetry build` creates a portable `.whl` file that can be installed with `pip` or `poetry` in any environment, making it highly versatile for deployment.

### Deployment Architecture for Millions of Daily Requests

A robust architecture must handle both real-time (online) and batch (offline) processing.

![External Library Usage](docs/images/external_usage.png)

#### Real-Time Inference

For on-demand scoring with low latency:

-   **FastAPI on Kubernetes (e.g., EKS, GKE):** The library would be wrapped in a **FastAPI** service. This service would be containerised with **Docker** and deployed to a **Kubernetes** cluster. A **Horizontal Pod Autoscaler (HPA)** would automatically scale the number of GPU-enabled pods based on traffic. An API Gateway would manage requests, authentication, and rate limiting.
-   **AWS Lambda:** For serverless deployments, the library can be packaged into a Lambda function. Because the model is large, it would be stored on **Amazon EFS** (Elastic File System) and mounted by the Lambda function at runtime. This pattern is cost-effective for sporadic traffic.

#### Batch Processing (Offline)

For processing millions of records efficiently:

-   **Apache Spark (Databricks, EMR):** The library's wheel file can be installed on a Spark cluster. The `score_pairs` function would be wrapped in a **User-Defined Function (UDF)**, allowing it to be applied to a Spark DataFrame in a distributed manner, processing massive datasets in parallel.
-   **Snowflake:** The library can be integrated with **Snowpark**. The model would be uploaded to the **Snowflake Model Registry**, and the scoring logic would be deployed as a secure UDF or Stored Procedure. This allows users to run the similarity scoring directly within their SQL queries on data stored in Snowflake.
-   **Airflow:** For orchestrating recurring batch jobs, an **Airflow DAG** would be created. A task within the DAG would install the library, fetch a batch of records from a database or data lake, run the scoring process on a dedicated compute instance (e.g., an EC2 instance), and write the results back.
