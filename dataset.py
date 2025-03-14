from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

class CocoBase:
    def __init__(self):
        self.root = {}
        self.images = []
        self.annotations = []
        self.categories = []
    
    def add_image(self, image_id, file_name):
        from PIL import Image
        width, height = Image.open(file_name).size
        self.images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name
        })

    def add_annotation(self, id, image_id, category_id, bbox, area, is_crowd):
        self.annotations.append({
            "id": id,
            "image_id": image_id, 
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": is_crowd,
        })

    def add_category(self, category_id, category):
        self.categories.append({"id": category_id, "name": category})

    def build(self):
        self.root["images"] = self.images
        self.root["annotations"] = self.annotations
        self.root["categories"] = self.categories
        return self.root

    def dumps(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.root, f)

def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset for object detection.

    Returns:
        The dataset.

    Below is an example of how to load an object detection dataset.

    ```python
    from datasets import load_dataset

    raw_datasets = load_dataset("cppe-5")
    if "validation" not in dataset_base:
        split = dataset_base["train"].train_test_split(0.15, seed=1337)
        dataset_base["train"] = split["train"]
        dataset_base["validation"] = split["test"]
    ```

    Ref: https://huggingface.co/docs/datasets/v3.2.0/package_reference/main_classes.html#datasets.DatasetDict

    You can replace this with your own dataset. Make sure to include
    the `test` split and ensure that it is consistent with the dataset format expected for object detection.
    For example:
        raw_datasets["test"] = load_dataset("cppe-5", split="test")
    """
    # Write your code here.
    from datasets import load_dataset
    raw_datasets = load_dataset("cppe-5")
    if "validation" not in raw_datasets:
        split = raw_datasets["train"].train_test_split(0.05, seed=1337)
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]
    return raw_datasets


def add_preprocessing(dataset, processor) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Add preprocessing to the dataset.

    Args:
        dataset: The dataset to preprocess.
        processor: The image processor to use for preprocessing.

    Returns:
        The preprocessed dataset.

    In this function, you can add any preprocessing steps to the dataset.
    For example, you can add data augmentation, normalization or formatting to meet the model input, etc.

    Hint:
    # You can use the `with_transform` method of the dataset to apply transformations.
    # Ref: https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.with_transform

    # You can also use the `map` method of the dataset to apply transformations.
    # Ref: https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.map

    # For Augmentation, you can use the `albumentations` library.
    # Ref: https://albumentations.ai/docs/

    from functools import partial

    # Create the batch transform functions for training and validation sets
    train_transform_batch = # Callable for train set transforming with batched samples passed
    validation_transform_batch = # Callable for val/test set transforming with batched samples passed

    # Apply transformations to dataset splits
    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)
    """
    # Write your code here.
    import albumentations as A
    import numpy as np
    from functools import partial

    train_augs = A.Compose([
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25))

    val_augs = A.Compose([
        A.NoOp()
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category"]))

    def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
        annotations = []
        for category, area, bbox in zip(categories, areas, bboxes):
            annotations.append({
                "image_id": image_id,
                "category_id": category,
                "area": area,
                "bbox": list(bbox),
                "iscrowd": 0,
            })
        return {"image_id": image_id, "annotations": annotations}
    
    def augment_and_transform_batch(examples, augs, processor, return_pixel_mask=False):
        images = []
        annotations = []

        for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))

            output = augs(image=image, bboxes=objects["bbox"], category=objects["category"])
            images.append(output["image"])

            formatted_annotations = format_image_annotations_as_coco(image_id, objects["category"], objects["area"], objects["bbox"])
            annotations.append(formatted_annotations)

        result = processor(images=images, annotations=annotations, return_tensors="pt")

        if not return_pixel_mask:
            result.pop("pixel_mask", None)
        
        return result
    
    train_transform_batch = partial(augment_and_transform_batch, augs=train_augs, processor=processor)
    val_transform_batch = partial(augment_and_transform_batch, augs=val_augs, processor=processor)

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(val_transform_batch)
    dataset["test"] = dataset["test"].with_transform(val_transform_batch)
    return dataset

if __name__ == "__main__":
    build_dataset()
    # add_preprocessing(dataset, processor)