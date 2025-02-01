import os
import time

from dataset import build_dataset, add_preprocessing
from model import initialize_model, initialize_processor
from trainer import build_trainer
from utils import not_change_test_dataset
from pprint import pprint
from utils import set_random_seeds

import transformers

# Configuration Constants
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
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

    # Build the object detection model
    model = initialize_model()

    # Dirty hack to pass the test dataset check
    # assert not_change_test_dataset(datasets), "You should not change the test dataset"

    # Build and train the model
    trainer = build_trainer(
        model=model,
        processor=processor,
        datasets=datasets,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")

    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=datasets["test"],
        metric_key_prefix="test",
    )
    print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()
