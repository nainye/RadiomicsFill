# RadiomicsFill-Mammo: Synthetic Mammogram Mass Manipulation with Radiomics Features

:fire:  Official implementation of "RadiomicsFill-Mammo: Synthetic Mammogram Mass Manipulation with Radiomics Features" (MICCAI 2024)

![overview](images/overview.png)
![result](images/result.png)

## Setting Up Development Environment

To set up your development environment, follow the steps below:

1. **Pull the Docker image:**

    We are using the `pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel` Docker image. You can pull it from Docker Hub by running:

    ```sh
    docker pull pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
    ```

2. **Run the Docker container:**

    Start a container from the pulled image. You can mount your project directory into the container for easy development:

    ```sh
    docker run --gpus all -it -v /path/to/your/project:/workspace --name radiomicsfill-mammo pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel /bin/bash
    ```

    Replace `/path/to/your/project` with the actual path to your project directory.

3. **Install additional Python libraries:**

    Once inside the Docker container, navigate to the mounted project directory:

    ```sh
    cd /workspace
    ```

    Then, install the required Python libraries using `pip`:

    ```sh
    pip install -r requirements.txt
    ```
