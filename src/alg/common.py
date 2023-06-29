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
