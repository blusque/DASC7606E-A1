import numpy as np
import os
import time

from dataset import build_dataset, add_preprocessing
from model import initialize_model, initialize_processor
from utils import not_change_test_dataset
from pprint import pprint
from utils import set_random_seeds
from trainer import build_trainer

# Define the estimator
class Params:
    lr = [1e-5, 1e-4]
    fp16 = [False, True]
    num_epochs = 10
    weight_decay = [0, 1e-4]
    max_grad_norm = [0.01, 1]
    lr_scheduler_type = ['linear', 'cosine', 'constant', 'polynomial', 'constant_with_warmup']
    num_train_epochs=30  # Adjust number of epochs as needed
    per_device_train_batch_size=8  # Batch size for training
    dataloader_num_workers=4  # Number of worker processes for data loading
    # lr_scheduler_kwargs={"num_stable_steps": 500, "num_decay_steps": 2000, "min_lr_ratio": 0.01},  # Scheduler arguments
    metric_for_best_model="eval_map"  # Metric to determine the best model
    greater_is_better=True  # Whether a higher metric is better
    load_best_model_at_end=True  # Load the best model after training
    eval_strategy="no"  # Evaluate at the end of every epoch
    save_strategy="no"  # Save the model at the end of every epoch
    remove_unused_columns=False  # Don't remove columns like 'image' (important for data)
    eval_do_concat_batches=False  # Ensure proper evaluation when batches are not concatenated
    push_to_hub=False  # Whether to push the model to the Hub,
    
    @staticmethod
    def sample():
        return {
            'learning_rate': np.random.random() * (Params.lr[1] - Params.lr[0]) + Params.lr[0],
            'weight_decay': np.random.random() * (Params.weight_decay[1] - Params.weight_decay[0]) + Params.weight_decay[0],
            'max_grad_norm': np.random.random() * (Params.max_grad_norm[1] - Params.max_grad_norm[0]) + Params.max_grad_norm[0],
            'lr_scheduler_type': np.random.choice(Params.lr_scheduler_type),
            'num_train_epochs': Params.num_epochs,  # Adjust number of epochs as needed
            'fp16': np.random.choice(Params.fp16),  # Use mixed precision if you have a supported GPU (set to True for faster training)
            'per_device_train_batch_size': Params.per_device_train_batch_size,  # Batch size for training
            'dataloader_num_workers': Params.dataloader_num_workers,  # Number of worker processes for data loading
            # lr_scheduler_kwargs={"num_stable_steps": 500, "num_decay_steps": 2000, "min_lr_ratio": 0.01},  # Scheduler arguments
            'metric_for_best_model': Params.metric_for_best_model,  # Metric to determine the best model
            'greater_is_better': Params.greater_is_better,  # Whether a higher metric is better
            'load_best_model_at_end': Params.load_best_model_at_end,  # Load the best model after training
            'eval_strategy': Params.eval_strategy,  # Evaluate at the end of every epoch
            'save_strategy': Params.save_strategy,  # Save the model at the end of every epoch
            'remove_unused_columns': Params.remove_unused_columns,  # Don't remove columns like 'image' (important for data)
            'eval_do_concat_batches': Params.eval_do_concat_batches,  # Ensure proper evaluation when batches are not concatenated
            'push_to_hub': Params.push_to_hub,  # Whether to push the model to the Hub,
        }

    
def single_train(model, processor, datasets, params):
    with open('test_metrics.txt', 'a') as f:
        f.write(str(params) + '\n')
    
    # Build and train the model
    trainer = build_trainer(
        model=model,
        processor=processor,
        datasets=datasets,
        params=params
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=datasets["test"],
        metric_key_prefix="test",
    )
    with open('test_metrics.txt', 'a') as f:
        f.write("Time: " + str((end_time - start_time) / 60) + '\n')
        f.write("Test Metrics: " + str(test_metrics) + '\n')

def main(params):
    """
    Main function to execute model training and evaluation.
    """
    # Set seed for reproducibility
    set_random_seeds()

    # Build the dataset
    raw_datasets = build_dataset()

    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"

    # Initialize the image processor
    processor = initialize_processor()

    # Add preprocessing to the dataset
    datasets = add_preprocessing(raw_datasets, processor)

    # Define the hyperparameter search
    for param in params:
        # Build the object detection model
        model = initialize_model()
        single_train(model, processor, datasets, param)
        # single_train(None, None, None, params)

if __name__ == "__main__":
    params = []
    for i in range(50):
        params.append(Params.sample())
    main(params)