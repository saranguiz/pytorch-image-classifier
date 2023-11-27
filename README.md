A Pytorch image classifier with pretrained models that can be fine-tuned on a custom dataset.

## Pre-requisites

- Python version `3.9` or greater

- Libraries: `torch`, `torchvision`, `argparse`, `Pillow` (can be installed with `pip`)

## Model

The `train` script is currently implemented to work with a `resnet50` model imported from the Pytorch's torchvision library. 

```model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)```

Nevertheless, a different pre-trained model can be imported and used for training. Check more details in the [Pytorch website](https://pytorch.org/vision/stable/models.html#classification) in order to pick another model.

Loss function: [`CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

## Classifier training

- The number of categories needs to be adjusted in this line:

    ```num_features = N```

- While the model gets trained, its loss and accuracy gets evaluated during the training and validation phase.

- The model is saved only if the validation accuracy is higher than in previous epochs.

- Testing is done at the end with the model version that best performed in the validation phase. Testing loss and accuracy is calculated against a subset never seen before.

## Dataset

- The main data directory must be in the repository's root directory
- The images used for training, validation and testing must be in their respective sub-folders ordered by categories.

    ```
    |-- data_dir
    |   |-- test
    |   |   |-- 1
    |   |   |-- 2
    |   |   |-- 3
    |   |   |-- 4
    |   |   |-- 5
    |   |-- train
    |   |   |-- 1
    |   |   |-- 2
    |   |   |-- 3
    |   |   |-- 4
    |   |   |-- 5
    |   |-- valid
    |   |   |-- 1
    |   |   |-- 2
    |   |   |-- 3
    |   |   |-- 4
    |   |   |-- 5
    ```

- For splitting a dataset into testing, training and validation subsets, the [`split-folders`](https://github.com/jfilter/split-folders) library might be helpful.

## Training execution

Run:

<code>python3 train.py ./data_dir --epochs [N] --lr [LR] --batch_size [BS]</code>
