# import unittest
# from parameterized import parameterized
# import numpy as np
# from math import prod
from enum import Enum

from FeatureSchedFIFO import FeatureSchedFIFO
from FeatureSchedFLoc import FeatureSchedFLoc
# import common


class MockConfig:
    def __init__(self, features, window, rank_mapping):
        self.features_files_names = features
        self.alg = dict()
        self.alg['window'] = window
        self.mpi_rank_mapping = rank_mapping
        self.max_feats_for_trial = -1
        self.fsched_debug = True
        self.num_features = len(features)
        self.small_window = True

    def get_param(self, name):
        if name == 'mpi_rank_mapping':
            return self.mpi_rank_mapping
        elif name == 'max_feats_for_trial':
            return self.max_feats_for_trial
        elif name == 'fsched_debug':
            return self.fsched_debug
        elif name == 'num_features':
            return self.num_features
        elif name == 'small_window':
            return self.small_window


class Op(Enum):
    GET_JOB = 1
    DONE_JOB = 2
    COMMIT_FEATURE = 3
    BEGIN_IT = 4


def simulate_sched(scheduler, operations):
    jobs_running = dict()

    for op, arg in operations:
        if op == Op.GET_JOB:
            rank = arg
            jobs_running[rank] = scheduler.get_feature(rank)
        elif op == Op.DONE_JOB:
            rank = arg
            f = jobs_running[rank]
            scheduler.tried_feature(f, rank)
            jobs_running[rank] = None
        elif op == Op.COMMIT_FEATURE:
            feature = arg
            scheduler.commit_feature(feature)
        elif op == Op.BEGIN_IT:
            scheduler.begin_iteration()


def test1():
    mapping = dict()
    mapping[0] = [0, 2]
    mapping[1] = [1, 3]

    # config = MockConfig(['f1', 'f2', 'f3'], 1, mapping)
    config = MockConfig(['f1', 'f2'], 1, mapping)

    operations = [
        (Op.BEGIN_IT, None),
        (Op.GET_JOB, 0),
        (Op.DONE_JOB, 0),
        (Op.GET_JOB, 1),
        (Op.DONE_JOB, 1),
        (Op.GET_JOB, 0),
        (Op.DONE_JOB, 0),
        (Op.GET_JOB, 0),
        (Op.DONE_JOB, 0),
        (Op.GET_JOB, 0),
        (Op.DONE_JOB, 0),
        (Op.GET_JOB, 2),
        (Op.DONE_JOB, 2),
        (Op.GET_JOB, 0),
        (Op.GET_JOB, 2),
        # (Op.COMMIT_FEATURE, ('f1', (0, 0, 0))),
        # (Op.GET_JOB, 0),
        # (Op.DONE_JOB, 0),
        # (Op.GET_JOB, 0),
        # (Op.DONE_JOB, 0),
    ]

    scheduler = FeatureSchedFLoc(config)

    simulate_sched(scheduler, operations)


if __name__ == '__main__':
    test1()
