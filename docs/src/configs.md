# Configs

With Hydra, we can manage everything with just a set of config files, which were outlined in the [project structure](project-structure.md). These have been seperated into categories, containing two key files `eval.yaml` and `train.yaml`, and the following sets (which you may want more than one configuration for):

- `callbacks`
- `data`
- `debug`
- `experiment`
- `extras`
- `hparams_search`
- `hydra`
- `local`
- `logger`
- `model`
- `paths`
- `trainer`


## `train.yaml` and `eval.yaml`

Whenever you train or evaluate the model, e.g. via `make train` (or similarly ```python src/train.py```), you use a selection of configs. This all happens in the `train.yaml` file. Here is a part of it:
```yaml
defaults:
  - _self_
  - data: cvae.yaml
  - model: convolutional.yaml
  - callbacks: default.yaml
  - logger: csv
  - trainer: default.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml
```
The `data` parameter points to a yaml file in the `configs/data/` folder, which in this case is `cvae.yaml`, a config file for the dataset I primarily work with. The next parameter is `model`, which similarly directs to a yaml file located in `configs/model/` associated with a particular machine learning model. As you can see, most of them are set to `default.yaml`, which is located in each of their respective folders, and includes standard defaults. New configs can be made and if selected here, in `train.yaml`, they will be used when the model is next trained. `eval.yaml` works very similarly.

Beyond selecting these yaml files, there are some extra options which can be changed in the two files, which are all hopefully self-explanatory.


## Callbacks

Callbacks are objects that can customize the behavior of the training loop in the PyTorch Trainer, and can be used for things such as as model checkpointing, early stopping, etc. The setup is designed to use callbacks that are [built-in to Lightning](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks).

For example, in `model_summary.yaml`, we use the `RichModelSummary` object from Lightning.

```yaml
model_summary:
  _target_: lightning.pytorch.callbacks.RichModelSummary
  max_depth: 1 # the maximum depth of layer nesting that the summary will include
```

One can then use this callback as follows:
```bash
python train.py callbacks=model_summary
```

Or include it in a selection of callbacks, as done in `default.yaml`, which can then be used to utilise all of them. Ideally configs which contain a `default.yaml` should be chosen in `train.yaml`
## Data

Each yaml file here is used for the loading of a dataset, and includes all the imporant key words one requires for the use of a dataset.

This is the default `CVAE.yaml`
_target_: src.data.cvae_datamodule.CVAEDataModule
data_dir: ${paths.data_dir}/CVAE
batch_size: 16
train_val_test_split: [10_205, 1_000, 1_000]
num_workers: 8
pin_memory: False
