# Brain Hemorrhage Identification

## Installation

Install requirements:
```sh
pip install -r requirements.txt
```

Prepare dataset:
```sh
# Assuming data is downloaded and extracted to ./data
python3 util/format_dataset ./data
```

## Training

See `python3 main.py --help` for training usage:
```sh
usage: main.py [-h] [--dataset DATASET]
               [--model {resnet50,resnet101,inceptionv3}] [--dropout DROPOUT]
               [--bias_init BIAS_INIT] [--learning_rate LEARNING_RATE]
               [--class_weights CLASS_WEIGHTS CLASS_WEIGHTS]
               [--metrics [{true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy,auc} [{true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy,auc} ...]]]
               [--epochs EPOCHS] [--save_path SAVE_PATH] [--log_path LOG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to dataset (default: ./data)
  --model {resnet50,resnet101,inceptionv3}
                        Which base model to use (default: resnet50)
  --dropout DROPOUT     Dropout rate (leave blank for no dropout) (default:
                        None)
  --bias_init BIAS_INIT
                        Bias initializer to use for the last layer (default:
                        None)
  --learning_rate LEARNING_RATE
                        Learning rate to use (default: 0.001)
  --class_weights CLASS_WEIGHTS CLASS_WEIGHTS
                        Class weights to use for training (default: None)
  --metrics [{true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy,auc} [{true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy,auc} ...]]
                        Which metrics to use for evaluation (F1 is on by
                        default) (default: None)
  --epochs EPOCHS       Number of epochs to train for. (default:
                        18446744073709551616)
  --save_path SAVE_PATH
                        Path to save models to (default:
                        ./output/2020-01-17_23:37:06/model-{epoch:03d}.hdf5)
  --log_path LOG_PATH   Path to save logs to (default:
                        ./output/2020-01-17_23:37:06/log.txt)
```

Replicating best configuration:
```sh
python3 main.py --metrics recall precision accuracy auc --class\_weights 0.587013456 3.373118838 --learning\_rate 0.001 --bias\_init -1.748545323 --dropout 0.5 --model inceptionv3
```

## Predictions

Predicting labels for a given directory of images:
```sh
usage: predict.py [-h] [--data_dir DATA_DIR]
                  [--model {resnet50,resnet101,inceptionv3}]
                  [--dropout DROPOUT] [--bias_init BIAS_INIT]
                  [--learning_rate LEARNING_RATE]
                  [--class_weights CLASS_WEIGHTS CLASS_WEIGHTS]
                  [--metrics [{true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy,auc} [{true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy,auc} ...]]]
                  [--weights_path WEIGHTS_PATH] [--output_path OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to directory containing test images. (default:
                        ./data/test)
  --model {resnet50,resnet101,inceptionv3}
                        Which base model to use (default: resnet50)
  --dropout DROPOUT     Dropout rate (leave blank for no dropout) (default:
                        None)
  --bias_init BIAS_INIT
                        Bias initializer to use for the last layer (default:
                        None)
  --learning_rate LEARNING_RATE
                        Learning rate to use (default: 0.001)
  --class_weights CLASS_WEIGHTS CLASS_WEIGHTS
                        Class weights to use for training (default: None)
  --metrics [{true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy,auc} [{true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy,auc} ...]]
                        Which metrics to use for evaluation (F1 is on by
                        default) (default: None)
  --weights_path WEIGHTS_PATH
                        Path to load model weights from (default: None)
  --output_path OUTPUT_PATH
                        Path to save results to (default:
                        ./output/test-2020-01-17_23:38:14.txt)
```

Replicating best configuration:
```sh
python3 main.py --metrics recall precision accuracy auc --class\_weights 0.587013456 3.373118838 --learning\_rate 0.001 --bias\_init -1.748545323 --dropout 0.5 --model inceptionv3 --weights_path <path to trained weights>
```