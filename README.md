# RadiomicsFill-Mammo: Synthetic Mammogram Mass Manipulation with Radiomics Features

:fire:  Official implementation of "RadiomicsFill-Mammo: Synthetic Mammogram Mass Manipulation with Radiomics Features" (MICCAI 2024)

[![arXiv](https://img.shields.io/badge/arXiv-2407.05683-red)](https://arxiv.org/pdf/2407.05683.pdf)

![overview](images/overview.png)
![result](images/result.png)

## Datasets

This project utilizes the VinDr-Mammo and INbreast datasets. You can download these datasets from the following links:

- [VinDr-Mammo](https://www.physionet.org/content/vindr-mammo/1.0.0/)
- [INbreast](https://www.kaggle.com/datasets/martholi/inbreast)

## Usage

### Installation

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

    Install the required Python libraries using `pip`:

    ```sh
    pip install -r requirements.txt
    ```



## Citation
If you use this code for your research, please cite our papers.

**BibTeX:**
```
@article{na2024radiomicsfill,
  title={RadiomicsFill-Mammo: Synthetic Mammogram Mass Manipulation with Radiomics Features},
  author={Na, Inye and Kim, Jonghun and Ko, Eun Sook and Park, Hyunjin},
  journal={arXiv preprint arXiv:2407.05683},
  year={2024}
}
```
