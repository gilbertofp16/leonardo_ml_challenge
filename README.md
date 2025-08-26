# ML Challenge – Text–Image Similarity

This project provides a solution for computing similarity scores between images and text captions, as per the challenge brief.

## Setup and Execution

To run the project, follow these steps:

1.  **Install Dependencies:** Ensure you have Python 3.10+ and Poetry installed. Then, install the project dependencies:
    ```bash
    poetry install
    ```

2.  **Prepare Data:** Place the `challenge_set.csv` file in the root of the project directory.

3.  **Run the Scoring Pipeline:** Execute the main script:
    ```bash
    poetry run python execute_text_image_similarity_csv.py
    ```

This will process the input CSV and generate a `challenge_scored.csv` file in the same directory, which includes the original data plus two new columns: `similarity` and `error`.

### Testing

The project includes a comprehensive test suite. You can run the tests with:
```bash
poetry run pytest
```
The suite includes:
- **Unit Tests:** Offline tests that mock network requests and model loading to verify the core logic of each component.
- **Integration Test:** An offline test that verifies the end-to-end CSV processing pipeline.
- **Semantic Coherence Tests:** A suite of tests that run against the live model to validate its performance on a small, curated dataset. These tests ensure that:
    1.  Correctly paired images and captions score higher on average than shuffled (incorrect) pairs.
    2.  For a set of distinctive images, each caption correctly retrieves its corresponding image with the highest score.
    3.  Higher-quality captions result in higher scores than irrelevant captions for the same image.
    4.  Scores are deterministic and stable across multiple runs.

---

## Question 1: Code to Compute the Similarity Metric

The code to compute the similarity metric is contained within the `text_image_similarity` Python package.

### Core Logic

The main logic is encapsulated in the `score_pairs` function (`text_image_similarity/scoring.py`). It is designed as a generator to process large datasets efficiently without loading everything into memory. For each batch of image-text pairs, it performs the following steps:

1.  **Concurrent Image Downloading:** Images are downloaded from their URLs in parallel using a `ThreadPoolExecutor` for a significant speed-up. The process includes a configurable timeout and retry mechanism for robustness.
2.  **Preprocessing:** Both the downloaded images and the text captions are preprocessed using the processor from the `openai/clip-vit-base-patch32` model. This converts the raw data into the numerical tensors the model requires.
3.  **Embedding Generation:** The preprocessed images and texts are passed to the pre-trained CLIP model, which generates numerical vector representations (embeddings) for each.
4.  **Similarity Calculation:** The cosine similarity between the image and text embeddings is calculated. The model's direct output (logits) is then normalized to a user-friendly **0-1 range** and rounded to four decimal places to produce the final score.

### Execution Flow

The `execute_text_image_similarity_csv.py` script orchestrates the process by calling the `score_csv` function from the pipeline module. This function handles reading the input CSV in chunks, loading the model a single time, and iterating through the data to compute and save the scores.

---

## Question 2: Time/Memory Footprint and Optimisation Strategies

### Time and Memory Footprint

-   **Memory:** The primary memory consumer is the CLIP model (`ViT-B/32`), which requires approximately **600 MB** of RAM or VRAM. The application's memory usage is otherwise minimal and stable, as it streams the input CSV in chunks rather than loading the entire file at once. Memory usage scales linearly with `batch_size`.
-   **Time:** The process is compute-bound by model inference and I/O-bound by image downloading.
    1.  **Image Downloading:** This is highly dependent on network latency and image sizes.
    2.  **Model Inference:** This is the core computation step. It is significantly faster on a GPU than on a CPU.

### Optimisation Strategies

Several strategies have been implemented or can be enabled to optimise performance:

-   **Batching (Implemented):** Processing records in batches is the most crucial optimisation for leveraging parallel computation on modern hardware (especially GPUs).
-   **Concurrent I/O (Implemented):** Image downloading is performed in parallel using a `ThreadPoolExecutor`, which hides network latency and dramatically improves throughput compared to sequential downloads.
-   **Hardware Acceleration (Supported):** The code will automatically detect and use a CUDA or MPS (Apple Silicon) GPU if available when `device` is set to `"auto"`, providing a significant speed-up (often >10x) for model inference.
-   **FP16/Mixed-Precision (Supported):** For CUDA devices, setting `precision` to `"fp16"` enables mixed-precision inference. This can halve the model's memory footprint and further accelerate computation with a negligible impact on accuracy.
-   **Caching (Strategy):** For further improvement, a local cache for downloaded images could be implemented to prevent re-downloading the same URL. If captions or images are frequently duplicated, their embeddings could also be cached.

---

## Question 3: Packaging and Deployment at Scale

Handling hundreds of millions of daily requests requires a distributed, scalable, and resilient architecture.

### Packaging

The application is packaged as a standard Python library using **Poetry**. This is the foundation for all deployment strategies, as it creates a portable and installable artifact (`.whl` file) via the `poetry build` command. This wheel can be installed with `pip` or `poetry` in any environment.

### Deployment Architecture

A robust, large-scale deployment would consist of the following:

-   **API Service:** The core logic would be wrapped in a high-performance web API using a framework like **FastAPI**. This service would accept requests containing image-text pairs and return similarity scores.
-   **Containerisation:** The application would be containerised using **Docker**, creating a lightweight, portable image that includes the Python environment and all dependencies.
-   **Container Orchestration:** The Docker containers would be deployed and managed by **Kubernetes (K8s)**. Kubernetes would handle service discovery, load balancing, and automated scaling.
-   **Autoscaling:** A **Horizontal Pod Autoscaler (HPA)** would be configured in Kubernetes to automatically scale the number of running application containers based on real-time metrics like CPU/GPU utilisation or requests per second. This ensures the system can handle traffic spikes and scale down to save costs.
-   **Large-Scale Data Integration:** For massive offline or batch processing, the packaged library can be used directly within large-scale data platforms. The `score_pairs` function is designed with a clear and robust interface, using dataclasses for both input (`Record`) and output (`ScoreResult`). This makes it straightforward to register as a **User-Defined Function (UDF)** in environments like **Apache Spark**, **Databricks**, **Google BigQuery**, or **Snowflake**, where clear data schemas are essential for reliable processing of billions of records.
-   **Asynchronous Processing via Message Queue:** To handle a high volume of real-time requests without blocking, a message queue (like **RabbitMQ** or **Kafka**) would be used. The API service would publish incoming requests as messages to the queue. A separate fleet of worker services would consume these messages, perform the scoring, and write the results to a database or another queue. This decouples the API from the heavy computation and allows each component to be scaled independently.
-   **Observability:** For a system of this scale, robust monitoring is crucial. This would include centralised logging (e.g., ELK stack), metrics (Prometheus), and distributed tracing (Jaeger) to ensure reliability and performance.
