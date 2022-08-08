#Should be different than 'x', 'y' and 'z'
MAIN_DF_INDEX_NAMES = ["X_IDX","Y_IDX",'Z_IDX']

# Useful for automatic enums
# https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# expanded = to be propagated
RealValues = enum('real', 'propagated', 'canal', 'expanded', 'empty')
