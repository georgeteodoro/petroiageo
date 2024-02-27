from FeatureSchedBase import FeatureSchedBase


class FeatureSchedFIFO(FeatureSchedBase):
    '''
    FIFO list feature scheduler implementation. Features are scheduled one 
    at a time for every displacement.
    '''

    def __init__(self, config):
        super(FeatureSchedFIFO, self).__init__(config)

        self._max_feats_for_trial = self._config.get_param(
            'max_feats_for_trial')
        self._all_features = self._gen_features_list()
        self._remaining_features = None

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

    def get_feature(self, rank):
        '''
        Returns a feature to be tried. This feature is returned based on 
        the concrete scheduling algorithm. Every feature can only be 
        returned once. Afterwards, it can only be returned after 
        self.begin_iteration(). If there are no more features, to be run
        it should return none.
        '''

        if len(self._remaining_features) > 0:
            return self._remaining_features.pop()
        else:
            return None

    def tried_feature(self, feature, rank):
        '''
        The FIFO implementation does not require the knowledge of whether a
        rank is finished with a feature.
        '''
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
        self._remaining_features = self._all_features.copy()
        if self._max_feats_for_trial > 0:
            self._remaining_features = \
                self._remaining_features[:self._max_feats_for_trial]

    def _gen_features_list(self):
        '''
        Generates the list of available features with all possible displacements.
        '''
        num_features = self._config.get_param('num_features')
        base_features = self._config.features_files_names
        window_size = self._config.alg['window']
        small_window = self._config.get_param('small_window')

        # Ignore any other file which is not an .h5 file.
        base_features = [f for f in base_features if f != ".gitkeep"]
        assert len(base_features) > 0, "No features found."

        # Limit features list to the maximum size
        if num_features > 0:
            base_features = base_features[:num_features]

        # Expand features for all displacements
        all_features = []
        for f in base_features:
            if small_window:
                print(f"[FeatureSchedFIFO] WARNING!!!!! USING SMALL WINDOW "
                      f"====================================================")
                for k in range(-window_size, window_size + 1):
                    all_features.append((f, (0, 0, k)))
            else:
                for i in range(-window_size, window_size + 1):
                    for j in range(-window_size, window_size + 1):
                        for k in range(-window_size, window_size + 1):
                            all_features.append((f, (i, j, k)))

        return all_features
