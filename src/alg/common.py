# Should be different than 'x', 'y' and 'z'
# MAIN_DF_INDEX_NAMES = ["X_IDX", "Y_IDX", 'Z_IDX']

RANDOM_STATE = 15

training_params = {
    'max_bin': 128,
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'verbose': -1,
    'min_data': 10,
    'boost_from_average': True,
    'bagging_freq': 1,
    'random_state': RANDOM_STATE,
    'num_threads': 1,
    'max_depth': 10,
    'num_leaves': 20,
    'seed': 0,
    'num_iterations': 10,
    # 'tree_learner': 'data',
}

POROSITY_DSET_NAME = "p"
FEAT_DSET_NAME = "f"


# Convert the feature tuple (e.g., ('NEAR', -3, 1, 2)) to
# string (e.g., 'NEAR/-3,1,2')
def f2str(f_tuple):
    if type(f_tuple) is tuple:
        return f"{f_tuple[0]}/{f_tuple[1]},{f_tuple[2]},{f_tuple[3]}"
    else:
        return f_tuple


# Useful for automatic enums
# https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type("Enum", (), enums)


# expanded = to be propagated
# none = match no points (used for skipping propagation)
RealValues = enum("real", "propagated", "canal", "canal_expanded", "expanded",
                  "empty", "none")


def has_points_within_chunk(wells_list, ring, chunk_slice, return_list=False):
    '''
    Return value is defined by 'return_list': if true, return a list with all
    wells within the chunk, if false, return true if there is at least one
    well within the chunk.

    Calculates whether any points of the input 'ring' should be found
    within the given 'chunk_slice'.
    Input 'wells_list' should be a list of tuples (x,y), one for 
    each well coordinate.
    For a ring point to be within the chunk, there should be some 
    overlapping between the bounded box of the chunk and the ring.
    However, it is simpler to check if there is no overlap and return
    the negation of it. The only exception is the case on which chunk_slice
    fits within the ring. This should return false and is checked 
    explicitly.
    This is checked for each well.
    '''

    wells_to_update = []

    # Check if there are points for each well
    for (w_x, w_y) in wells_list:
        c_x_i = chunk_slice[0].start
        c_x_o = chunk_slice[0].stop - 1
        c_y_i = chunk_slice[1].start
        c_y_o = chunk_slice[1].stop - 1
        r_x_i = w_x - ring
        r_x_o = w_x + ring
        r_y_i = w_y - ring
        r_y_o = w_y + ring

        # Check if chunk_slice fits within the ring (border non-included)
        # If so, this chunk has no points for the ring
        if ((c_x_i > r_x_i) and (c_x_o < r_x_o) and (c_y_i > r_y_i) and
            (c_y_o < r_y_o)):
            continue

        # Check if there is no overlapping between the ring and the chunk
        # bounding box
        no_ovlp_x = (c_x_o < r_x_i) | (c_x_i > r_x_o)
        no_ovlp_y = (c_y_o < r_y_i) | (c_y_i > r_y_o)

        # If there is at least one no-overlapping, then there is no
        # overlapping. If both no-overlapping are false, then there
        # should be overlapping
        if not (no_ovlp_x or no_ovlp_y):
            if return_list:
                # If there is at least one overlapping,
                # then add the current well
                wells_to_update.append((w_x, w_y))
            else:
                # If there is at least one overlapping, then return true
                return True

    if return_list:
        return wells_to_update
    else:
        # No overlapping was found on any well
        return False
