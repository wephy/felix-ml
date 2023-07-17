# Installation

Before you can play around with Felix ML, you'll need to make sure all the package requirements are met, some of which specific versions might be needed.

> I recommend using [Anaconda][anaconda] to manage an environment for this, and once Anaconda is installed, you have two options: `pip` or `conda`. You may prefer to use conda as pip installs packages in a loop, without ensuring dependencies across all packages are fulfilled simultaneously, but conda achieves proper dependency control across all packages; furthermore, conda allows for installing packages without requiring certain compilers or libraries to be available in the system, since it installs precompiled binaries.

1. Install [Anaconda][anaconda]
2. If you machine has a [CUDA-enabled GPU][cudagpu], install [CUDA][cuda]
3. Choose one of the following methods:

    #### Pip

    ```bash
    # clone project
    git clone https://github.com/wephy/felix-ml
    cd felix-ml

    # [OPTIONAL] create conda environment
    conda create -n felix-ml python=3.9
    conda activate felix-ml

    # install pytorch according to instructions
    # https://pytorch.org/get-started/

    # install requirements
    pip install -r requirements.txt
    ```

    #### Conda

    ```bash
    # clone project
    git clone https://github.com/wephy/felix-ml
    cd felix-ml

    # create conda environment and install dependencies
    conda env create -f environment.yaml -n myenv

    # activate conda environment
    conda activate myenv
    ```


[anaconda]: https://www.anaconda.com/download
[cudagpu]: https://developer.nvidia.com/cuda-gpus
[cuda]: https://developer.nvidia.com/cuda-downloads