import ray
from ray.train import ScalingConfig
from ray.train.lightgbm import LightGBMTrainer

# Load data.
dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

# Split data into train and validation.
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

trainer = LightGBMTrainer(
    scaling_config=ScalingConfig(
        # Number of workers to use for data parallelism.
        num_workers=2,
        # Whether to use GPU acceleration. Set to True to schedule GPU workers.
        use_gpu=False,
    ),
    label_column="target",
    num_boost_round=20,
    params={
        # LightGBM specific params
        "objective": "binary",
        "metric": ["binary_logloss", "binary_error"],
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
    # If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    # run_config=ray.train.RunConfig(storage_path="s3://..."),
)
result = trainer.fit()
print(result.metrics)