# Should be different than 'x', 'y' and 'z'
# MAIN_DF_INDEX_NAMES = ["X_IDX", "Y_IDX", 'Z_IDX']


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
RealValues = enum("real", "propagated", "canal", "canal_expanded", "expanded",
                  "empty")


def has_points_within_chunk(wells_list, ring, chunk_slice):
    '''
    Calculates whether any points of the input 'ring' should be found
    within the given 'chunk_slice'.
    For a ring point to be within the chunk, there should be some 
    overlapping between the bounded box of the chunk and the ring.
    However, it is simpler to check if there is no overlap and return
    the negation of it. The only exception is the case on which chunk_slice
    fits within the ring. This should return false and is checked 
    explicitly.
    This is checked for each well.
    '''

    # Check if there are points for each well
    for (_, (w_x, w_y)) in wells_list:
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
            # If there is at least one overlapping, then return true
            return True

    # No overlapping was found on any well
    return False