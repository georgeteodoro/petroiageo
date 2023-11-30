import numpy as np
from data_filter import FeatSelectionTrainDataFilter

RANDOM_STATE = 15

params = {
    'max_bin': 128,
    'max_depth': 10,
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 20,
    'verbose': -1,
    'min_data': 10,
    'boost_from_average': True,
    'bagging_freq': 1,
    'random_state': RANDOM_STATE,
    'num_threads': 1
    # 'tree_learner': 'data',
}


def test_new_feature(test_data, config):
    '''
    Trains a model with test_data, returning the metric values for the
    trained model.
    Abstracts any low-level assumptions about where data is located.
    Out-of-core training through incremental learning is supported, however
    not required. It is possible to configure n_training_chunks=1 to disable
    the incremental learning.
    '''

    test_wells_ids = config.alg['test_only_wells']
    n_training_chunks = config.alg['parallel']['n_training_chunks']
    train_wells_ids = config.train_wells_ids

    return (2, 1)

    # Configure test_data for out-of-core execution, if needed
    test_data.set_num_training_chunks(n_training_chunks)

    # Initialize metrics lists
    rmse_list = []
    mae_list = []

    # Allocate train data memory. This is done to save on allocation time
    # while also certifying that memory footprint remains tractable.
    train_size = test_data.get_train_expected_size()
    train_type = test_data.get_train_type()
    X_train = np.empty((train_size), dtype=train_type)
    y_train = np.empty((train_size), dtype=np.float64)

    # Leave-one-well-out
    for curr_well_id in train_wells_ids:

        # Reset model for incremental learning
        regressor = None

        # Validation data is not chunked, thus it only needs to be
        # retrieved once with all data
        X_val, y_val = test_data.get_val_values(curr_well_id)

        # X_val=None if there are no validation points available. This can
        # only happen if there is only 1 well being propagated.
        if not X_val:
            return None

        # Performs incremental learning on all chunks
        for chunk_id in range(n_training_chunks):
            # Setup training data
            test_data.get_train_values(curr_well_id, chunk_id, X_train,
                                       y_train)

            lgb_train_dataset = lgb.Dataset(X_train, y_train)
            lgb_eval_dataset = lgb.Dataset(
                X_val,
                y_val,
                reference=lgb_train_dataset,
            )

            regressor = lgb.train(
                params,
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
        pred = trained_regressor.predict(X_val)
        rmse = np.sqrt(np.mean((pred - y_val)**2))
        mae = mean_absolute_error(y_val, pred)
        rmse_list.append(rmse)
        mae_list.append(mae)

    return np.mean(rmse_list), np.mean(mae_list)
