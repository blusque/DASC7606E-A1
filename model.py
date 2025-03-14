from constants import ID2LABEL, LABEL2ID, MODEL_NAME, MAX_SIZE

def initialize_model():
    """
    Initialize a model for object detection.

    Returns:
        A model for object detection.

    NOTE: Below is an example of how to initialize a model for object detection.

    from transformers import AutoModelForObjectDetection
    from constants import ID_TO_LABEL, LABEL_TO_ID, MODEL_NAME

    model = AutoModelForObjectDetection.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,  # specify the model checkpoint
        id2label=ID_TO_LABEL,  # map of label id to label name
        label2id=LABEL_TO_ID,  # map of label name to label id
        ignore_mismatched_sizes=True,  # allow replacing the classification head
    )

    You are free to change this.
    But make sure the model meets the requirements of the `transformers.Trainer` API.
    ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
    """
    # Write your code here.
    from transformers import AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection

    model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME, 
                                                            id2label=ID2LABEL,
                                                            label2id=LABEL2ID,
                                                            ignore_mismatched_sizes=True
                                                            )
    # model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_NAME, 
    #                                                         id2label=ID2LABEL,
    #                                                         label2id=LABEL2ID,
    #                                                         ignore_mismatched_sizes=True
    #                                                         )
    return model


def initialize_processor():
    """
    Initialize a processor for object detection.

    Returns:
        A processor for object detection.

    NOTE: Below is an example of how to initialize a processor for object detection.

    from transformers import AutoImageProcessor
    from constants import MODEL_NAME

    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME
    )

    You are free to change this.
    But make sure the processor meets the requirements of the `transformers.Trainer` API.
    ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
    """
    # Write your code here.
    from transformers import AutoImageProcessor
    
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        do_size=True,
        size={'max_height': MAX_SIZE, 'max_width': MAX_SIZE},
        do_pad=True,
        pad_size={'height': MAX_SIZE, 'width': MAX_SIZE},
        # use_fast=True
    )
    return processor