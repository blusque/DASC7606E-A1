# Prologue
This is a personal report for assignment 1 in DASC7606E class in HKU. I'm quite familiar with pytorch but never used huggingface libraries before, so I skipped the torch_gym and spent more time on learning how to use datasets and transformers api. I recorded my experience and training results here for further reference.

# Load dataset
`datasets` is a library provided by huggingface to load and process datasets for training. `load_dataset` is the API to load a dataset, and it seems that it's feasible to pass the dataset's name as the parameter and specify the `cache_dir` to change the directory where download result and data are cached. The data is downloaded as `.parquet` format so you cannot read it directly. However, in order to change the dataset into `COCO`'s format, you have to access the data 

The next step is to change the data into `COCO`'s format.