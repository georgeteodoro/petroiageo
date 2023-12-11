import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from data_filter import FeatSelectionTrainDataFilter
import common


def test_new_feature(trial_data, config):
    '''
    Trains a model with trial_data, returning the metric values for the
    trained model.
    Abstracts any low-level assumptions about where data is located.
    Out-of-core training through incremental learning is supported, however
    not required. It is possible to configure n_training_chunks=1 to disable
    the incremental learning.
    '''

    test_wells_ids = config.alg['test_only_wells']
    n_training_chunks = int(config.alg['parallel']['n_training_chunks'])
    train_wells_ids = config.train_wells_ids

    # Configure trial_data for out-of-core execution, if needed
    trial_data.set_num_training_chunks(n_training_chunks)

    # Initialize metrics lists
    rmse_list = []
    mae_list = []

    # Leave-one-well-out
    for curr_well_id in train_wells_ids:
        # Reset model for incremental learning
        regressor = None

        # Validation data is not chunked, thus it only needs to be
        # retrieved once with all data
        X_val, y_val = trial_data.get_val_values(curr_well_id)

        # X_val=None if there are no validation points available. This can
        # only happen if there is only 1 well being propagated.
        if X_val is None:
            return None

        # Performs incremental learning on all chunks
        for chunk_id in range(n_training_chunks):
            # Setup training data
            X_train, y_train = trial_data.get_train_values(
                curr_well_id, chunk_id)

            lgb_train_dataset = lgb.Dataset(X_train, y_train)
            lgb_eval_dataset = lgb.Dataset(
                X_val,
                y_val,
                reference=lgb_train_dataset,
            )

            regressor = lgb.train(
                common.training_params,
                lgb_train_dataset,
                init_model=regressor,
                num_boost_round=100,
                valid_sets=lgb_eval_dataset,
                keep_training_booster=True,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False)
                ],
            )

        # Calculate error metrics
        pred = regressor.predict(X_val)
        rmse = np.sqrt(np.mean((pred - y_val)**2))
        mae = mean_absolute_error(y_val, pred)
        rmse_list.append(rmse)
        mae_list.append(mae)

    return np.mean(rmse_list), np.mean(mae_list)
