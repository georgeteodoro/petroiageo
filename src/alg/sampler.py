from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from collections import defaultdict
from decimal import Decimal

from config_parser import Config
from common import PointDtypeIdx

BETA_DIST_RING_START = 0


class AbstractChunkSampler(ABC):
    """
    This is a sampler.
    """

    def __init__(self, config: Config):
        self._config = config

    @abstractmethod
    def sample(self,
               data: np.ndarray,
               n_trial_points_total: int,
               curr_starting_layer: int,
               curr_ring: int,
               rng: np.random.Generator = None) -> np.ndarray:
        raise NotImplementedError(
            "This method should be overrided by a child class!")


class ChunkSamplerV1(AbstractChunkSampler):
    """
    This sampler samples proportionally from every well based on
    a beta dist distribuition.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self._alpha = config.alg['sampling']['beta_dist']['alpha']
        self._beta = config.alg['sampling']['beta_dist']['beta']
        self._rng_seed = config.alg['sampling']['seed']
        self._sampling_max_points = config.alg['sampling']['max_points']
        self._layers_window_size = config.alg['sampling']['layers_window_size']

    def sample(self,
               data: np.ndarray,
               n_trial_points_total: int,
               curr_final_layer: int,
               curr_ring: int,
               rng: np.random.Generator = None) -> np.ndarray:
        """
        This sampler may sample a little more data than the max_points.
        data: The chunk data
        n_trial_points_total: The total global number of trial points
        curr_final_layer: The first iteration layer being propagated.
            If we propagate one layer per iteration, its going to be 
            equal to the iteration. Otherwise, see Config.ring_range_to_expand()
        rng: An optional Random Number generator. If None, one will
            be created based on the config seed and used to sample
        """
        # print("[ChunkSamplerV1][sample] Starting sampling")
        if data is None or data.size == 0:
            print("[ChunkSamplerV1][sample] Data is None or empty! Returning")
            return None

        if not self._should_sample(n_trial_points_total,
                                   self._sampling_max_points):
            print("[ChunkSamplerV1][sample] Should not sample! Returning")
            return data

        assert curr_ring < curr_final_layer, f"Start iteration "\
            f"{curr_final_layer} should be after latest iteration {curr_ring}."
        # At this point, we for sure need to sample data

        # print("[ChunkSamplerV1][sample] Sampling!")
        if rng is None:
            rng = np.random.default_rng(seed=self._rng_seed)

        ordered_well_ids, well_ids_count = np.unique(data['well_id'],
                                                     return_counts=True)

        n_points_to_sample_chunk = self._get_n_points_to_sample_chunk(
            data, n_trial_points_total, self._sampling_max_points)

        n_samp_points_per_well = self._get_n_pts_to_sample_per_well(
            n_points_to_sample_chunk, well_ids_count)

        assert_msg = "Num of sampled points of some well was bellow 1!"
        assert_msg += f" {n_samp_points_per_well}"
        assert len(n_samp_points_per_well[n_samp_points_per_well <=
                                          0]) == 0, assert_msg

        sampled_training_points = None

        # Iteration with the maximun probability
        if self._layers_window_size > 0:
            beta_dist_start_it = max([
                BETA_DIST_RING_START,
                curr_final_layer - self._layers_window_size
            ])
        else:
            beta_dist_start_it = BETA_DIST_RING_START

        assert beta_dist_start_it >= 0

        for well_idx, well_id in enumerate(ordered_well_ids):
            n_samp_points_well = n_samp_points_per_well[well_idx]
            well_points = data[data['well_id'] == well_id]

            probs = stats.beta.pdf(np.full(well_points.size, curr_ring),
                                   self._alpha,
                                   self._beta,
                                   loc=beta_dist_start_it,
                                   scale=curr_final_layer)

            # Scaling so it sums to 1
            probs = probs / np.sum(probs)
            curr_sampled_points = rng.choice(well_points,
                                             n_samp_points_well,
                                             replace=False,
                                             p=probs)

            if sampled_training_points is None:
                sampled_training_points = curr_sampled_points
            else:
                sampled_training_points = np.concatenate(
                    [sampled_training_points, curr_sampled_points])

        assert sampled_training_points.size >= n_points_to_sample_chunk

        return sampled_training_points

    def _should_sample(self, n_trial_points_total, sampling_max_points):
        """
        If sampling_max_points is positive and its smaller
        than n_trial_points_total, then we should sample
        """
        if sampling_max_points > 0:
            n_sample_points_total = min(n_trial_points_total,
                                        sampling_max_points)
        else:
            # if sampling_max_points<0, then no sampling should be done
            n_sample_points_total = n_trial_points_total

        return n_trial_points_total > n_sample_points_total

    def _get_n_points_to_sample_chunk(self, data: np.ndarray,
                                      n_trial_points_total: int,
                                      sampling_max_points: int):
        """
        Calc how many points are to be sampled from this chunk proportionally
        on the the chunk size and the total of trial points
        """
        n_points_to_sample_chunk = data.size / n_trial_points_total
        n_points_to_sample_chunk *= sampling_max_points
        n_points_to_sample_chunk = np.ceil(n_points_to_sample_chunk).astype(int)
        return n_points_to_sample_chunk

    def _get_n_pts_to_sample_per_well(self, n_points_to_sample_chunk: int,
                                      n_pts_per_well: np.ndarray) -> np.ndarray:
        """
        Calc the total number of points to sample from each well.
        n_points_to_sample_chunk: Total points to sample from chunk
        n_pts_per_well: Num of points per well in this chunk
        Returns a np.ndarray with the total number of points to sample
        from each well in the same order as n_pts_per_well
        """
        # This might give more points to sample in total than
        # n_points_to_sample_chunk because of the ceil. So it must be treated
        n_samp_points_per_well: np.ndarray = np.ceil(
            (n_pts_per_well / n_pts_per_well.sum()) *
            n_points_to_sample_chunk).astype(int)

        # Treating difference to expected n_points_to_sample_chunk
        # We remove 1 point from every well with biggest curr samp size
        # until diff == 0. This next diff will never be < 0.
        diff = n_samp_points_per_well.sum() - n_points_to_sample_chunk

        while diff > 0:
            biggest_samp = n_samp_points_per_well.argmax()
            n_samp_points_per_well[biggest_samp] -= 1
            diff = n_samp_points_per_well.sum() - n_points_to_sample_chunk

        # This garantees that every well has at least one training point at the end
        n_samp_points_per_well = np.where(n_samp_points_per_well <= 0, 1,
                                          n_samp_points_per_well)

        return n_samp_points_per_well


def target_based_sampler(propagated_points: list,
                         buckets_len: dict,
                         poros_width: Decimal,
                         bucket_max_size: int,
                         alpha: float,
                         rng: np.random.Generator = None,
                         seed: int = 42):
    """
    Performs target based sampling on the propagated_points. It selects points based on 
    its respective buckets size. Updates bucket_len inplace.
    
    propagated_points: List of points with dtype ('x', 'y', 'z', 'phi', 'real', 'ring',
      'well_id')
    buckets_len: Dict of bucket id as key and its current size as value
    bucket_max_size: Int representing the max size every bucket should be
    alpha: Float on the interval [0,1] represeting the buckets update rate
    rng: Numpy random generator. If None, one is constructed based on the seed
    seed: Int representing the seed for the rng if needed

    Returns:
    points_to_add: list of points from the last ring which should be added 
    to the sampled database.

    points_to_remove: list of how many points from a given bucket should be 
    removed from the current sample of points to accommodate the new points 
    from the last ring.
    """

    points_to_add = []
    points_to_remove = defaultdict(int)

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    for point in propagated_points:
        point_por = point[PointDtypeIdx.phi]
        bucket_id = Decimal(point_por) - (Decimal(point_por) % poros_width)
        if buckets_len[bucket_id] < bucket_max_size:
            points_to_add.append(point)
            buckets_len[bucket_id] += 1
        else:
            if rng.random() < alpha:
                points_to_add.append(point)
                points_to_remove[bucket_id] += 1

    return points_to_add, points_to_remove
