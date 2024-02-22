from FeatureSchedBase import FeatureSchedBase


class FeatureSchedFLoc(FeatureSchedBase):
    '''
    Smart feature locality-aware scheduler. It favors the distribution of 
    one feature per compute node
    '''
    def __init__(self, config):
        super(FeatureSchedFLoc, self).__init__(config)

        # Attributes from config
        self._rank_mapping = self._config.get_param('mpi_rank_mapping')
        self._max_feats_for_trial = self._config.get_param(
            '_max_feats_for_trial')
        self._base_features = self._config.features_files_names

        # Ignore any other file from the base features 
        # which is not an .h5 file.
        self._base_features = [
            f for f in self._base_features if f != ".gitkeep"
        ]
        assert len(self._base_features) > 0, "No features found."

        # Limit features list to the maximum size
        num_features = self._config.get_param('num_features')
        if num_features > 0:
            self._base_features = self._base_features[:num_features]

        # Account of features per node. Each node is a key on the dict.
        # Each node have 0 or more features within it. To have a feature
        # means that at least one rank on said node is running a trial
        # with this feature. Each feature has a counter of how many ranks
        # within the node are running this feature. When a feature is
        # scheduled, this value is incremented. When self.tried_feature()
        # is called, this value is decremented.
        self._feat_loc = dict()

        # Helper mapping of rank to node
        self._node_of_rank = dict()
        for n, ranks in self._rank_mapping.items():
            for r in ranks:
                self._node_of_rank[r] = n

        # List of features (without displacements) which were not yet
        # scheduled to any node.
        self._unscheduled_features = None

        # List of features (without displacements) which were scheduled to
        # at least one node, but still have displacements to try.
        self._running_features = None

        # Mapping of which displacements for each feature were tried or not.
        self._remaining_trials = dict()

    # =========================================================================
    # === Public interface ====================================================
    # =========================================================================

    def begin_iteration(self):
        '''
        This method represents the initial moment at the beginning of an 
        iteration. Within it, internal data structures should be initialized.

        This method should be called exactly once at the beginning of every 
        iteration. This should not be called anywhere else.
        '''

        self._reset_internal_data()

    def has_features(self):
        '''
        Returns whether there are still features to be tried.
        '''

        return len(self._remaining_features) > 0

    def get_feature(self, rank):
        '''
        '''

        node = self._node_of_rank[rank]

        # Try to return a displacement of a feature present on the rank's node
        for feature in self._feat_loc[node]:
            if len(self._remaining_trials[feature]) > 0:
                # There is a displacement for the current feature
                # yet to be tried

                displacement = self._remaining_trials[feature].pop()
                return (feature, displacement)

        # Try to return the displacement of a new feature which was not yet
        # scheduled to any node/rank.

        # Return a displacement of a feature which was also scheduled to
        # other nodes.

        # TODO: should there be a limit of how many different features a
        # node should have? too many features means high memory usage.

    def tried_feature(self, feature, rank):
        pass

    def commit_feature(self, feature):
        '''
        This represents a feature which was committed, meaning that for a 
        given iteration, one feature was chosen at the end of said iteration. 
        In that case, this feature should not be available for further trials.
        Also, the internal representation of available features should be 
        reset in waiting of another iteration.

        This method should be called exactly once at the end of every 
        committed feature. This should not be called anywhere else.
        '''

        self._all_features.remove(feature)
        self._reset_internal_data()

    # =========================================================================
    # === Helper functions ====================================================
    # =========================================================================

    def _reset_internal_data(self):

        # Reset mapping of features per node
        self._feat_loc = dict()
        for n in self._rank_mapping.keys():
            self._feat_loc[n] = []

        # Reset the list of displacements per feature
        self._remaining_trials = dict()
        all_displacements = self._gen_displacements_list()
        for feature in self._base_features:
            self._remaining_trials = all_displacements.copy()

    def _gen_displacements_list(self):
        '''
        Generates the list of available features with all possible displacements.
        '''
        window_size = self._config.alg['window']

        # Expand features for all displacements
        all_displacements = []
        for i in range(-window_size, window_size + 1):
            for j in range(-window_size, window_size + 1):
                for k in range(-window_size, window_size + 1):
                    all_displacements.append((i, j, k))

        return all_displacements
