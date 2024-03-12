import numpy as np

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
            'max_feats_for_trial')
        self._base_features = self._config.features_files_names
        self._debug = self._config.get_param('fsched_debug')

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
        # with this feature.
        self._feat_loc = dict()

        # Helper mapping of rank to node
        self._node_of_rank = dict()
        for n, ranks in self._rank_mapping.items():
            for r in ranks:
                self._node_of_rank[r] = n

        # List of features (without displacements) which were not yet
        # scheduled to any node. These have priority to be scheduled
        # between nodes before splitting a feature between multiple nodes.
        self._unscheduled_features = None

        # Map of features (without displacements) which were scheduled to
        # at least one node, but still have displacements to try. Each
        # entry represents how many nodes are currently running. If a feature
        # is withing this map, then there are more displacements to try. A
        # feature is removed at self.tried_feature() when there are no more
        # available displacements for trial.
        self._running_features = dict()

        # Mapping of which displacements for each feature were tried or not.
        self._remaining_disps = dict()

        # List of committed (feature, displacement) tuples. The pairs
        # within this list are not returned by self.get_feature(). This
        # list is reset at the end of an iteration.
        self._committed_features = []

        # Number of features returned for trial. If 'max_feats_for_trial' was
        # set, then only self._max_feats_for_trial features are returned.
        self._tried_features = -1

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

        self._committed_features = []
        self._reset_internal_data()

    def get_feature(self, rank):
        '''
        Return a feature based on the following preference:
            1. The asking rank have a feature within its node: return a 
            displacement of said feature.
            2. The asking rank have no feature within its node with 
            displacements for execution: return a displacement of a feature
            with the least amount of nodes trying this feature. Ideally, this
            feature should have no nodes trying it.
            3. There are no more displacements for trial: return None
        '''

        node = self._node_of_rank[rank]

        # Check if the max number of features to be returned
        # was already reached
        if (self._max_feats_for_trial > 0
                and self._tried_features == self._max_feats_for_trial):

            if self._debug:
                print("[FeatureSchedFLoc][get_feature] "
                      "Reached max_feats_for_trial")
            return None

        # Try to return a displacement of a feature present on the rank's node
        # for feature in self._feat_loc[node].keys():
        for feature in self._feat_loc[node]:
            if len(self._remaining_disps[feature]) > 0:
                # There is a displacement for the current feature
                # yet to be tried

                # Get a displacement for this feature
                displacement = self._remaining_disps[feature].pop()
                self._tried_features += 1
                if self._debug:
                    print(f"[FeatureSchedFLoc][get_feature] Returning "
                          f"{(feature, displacement)} from node:rank "
                          f"{node}:{rank} => feature is present on node")
                return (feature, displacement)

        # Try to return the displacement of a new feature which was not yet
        # scheduled to any node/rank.
        if len(self._unscheduled_features) > 0:
            # Schedule the first unscheduled feature to the node of 'rank'
            # This is done by signaling that 1 rank is trying this feature
            new_feature = self._unscheduled_features.pop()
            self._feat_loc[node].append(new_feature)

            # This feature is now running
            self._running_features[new_feature] = 1

            # Get a displacement for this feature
            print(f"====feature {new_feature} from "
                  f"{self._remaining_disps[new_feature]}")
            displacement = self._remaining_disps[new_feature].pop()
            self._tried_features += 1
            if self._debug:
                print(f"[FeatureSchedFLoc][get_feature] Returning "
                      f"{(new_feature, displacement)} from node:rank "
                      f"{node}:{rank} => new unscheduled feature")
            return (new_feature, displacement)

        # Look for a displacement of a feature which was also scheduled
        # to other nodes. Returns the feature with the least amount of
        # nodes trying a feature.
        min_count = np.inf
        sel_feature = None
        for feature, count_nodes in self._running_features.items():
            if count_nodes < min_count:
                min_count = count_nodes
                sel_feature = feature

        # Return the feature if there is one available
        if sel_feature is not None:
            # Allocate the feature to node
            self._feat_loc[node].append(sel_feature)
            self._running_features[sel_feature] += 1

            # Get an available displacement for the chosen feature
            displacement = self._remaining_disps[sel_feature].pop()
            self._tried_features += 1
            if self._debug:
                print(f"[FeatureSchedFLoc][get_feature] Returning "
                      f"{(feature, displacement)} from node:rank "
                      f"{node}:{rank} => feature now allocated to "
                      f"{self._running_features[sel_feature]} nodes")
            return (sel_feature, displacement)

        # There are no more displacements for a feature to be tried
        if self._debug:
            print(f"[FeatureSchedFLoc][get_feature] No more features")
        return None

        # TODO: should there be a limit of how many different features a
        # node should have? too many features means high memory usage.

    def tried_feature(self, full_feature, rank):
        '''
        After a feature is tried, the node of the rank which executed it
        now has one less rank running it. 

        If no more ranks are running a feature within a node, then this 
        node loses the allocation of the given feature. This can be a problem 
        if the training time is too short and a given rank have multiple 
        requests for features replied before other ranks. This could result in
        a thrashing behavior, on which features allocations can be passed 
        between nodes. This is solved by only removing a feature allocation of
        a node when there are no more displacements available for trial.
        '''

        node = self._node_of_rank[rank]
        feature, displacement = full_feature

        # If there are no more displacements for trial remove the feature
        # allocation of a node and the feature from the running list
        if len(self._remaining_disps[feature]) == 0:
            if feature in self._feat_loc[node]:
                self._feat_loc[node].remove(feature)

            if feature in self._running_features.keys():
                self._running_features.pop(feature)

    def commit_feature(self, feature):
        '''
        This represents a feature which was committed, meaning that for a 
        given iteration, one feature was chosen at the end of said iteration. 
        In that case, this feature is not be available for further trials.

        This method should be called exactly once at the end of every 
        committed feature. This should not be called anywhere else.
        '''

        self._committed_features.append(feature)
        self._reset_internal_data()

    # =========================================================================
    # === Helper functions ====================================================
    # =========================================================================

    def _reset_internal_data(self):

        self._tried_features = 0

        # Reset mapping of features per node
        self._feat_loc = dict()
        for n in self._rank_mapping.keys():
            self._feat_loc[n] = []

        # Reset the list of displacements per feature
        self._remaining_disps = dict()
        all_displacements = self._gen_displacements_list()
        for feature in self._base_features:
            self._remaining_disps[feature] = all_displacements.copy()

        # Remove the (feature, displacements) which were already committed
        for feature, displacement in self._committed_features:
            self._remaining_disps[feature].remove(displacement)

        # Reset features as all being initially unscheduled
        self._unscheduled_features = self._base_features.copy()
        self._running_features = dict()

        # Check if there are features with no displacements available and 
        # remove them if any
        to_remove = []
        for feature, displacements in self._remaining_disps.items():
            if len(displacements) == 0:
                to_remove.append(feature)

        for feature in to_remove:
            self._remaining_disps.pop(feature)
            self._unscheduled_features.remove(feature)

    def _gen_displacements_list(self):
        '''
        Generates the list of available features with all possible displacements.
        '''
        window_size = self._config.alg['window']
        small_window = self._config.get_param('small_window')

        # Expand features for all displacements
        all_displacements = []
        if small_window:
            print(f"[FeatureSchedFLoc] WARNING!!!!! USING SMALL WINDOW "
                  f"===================================================")
            for k in range(-window_size, window_size + 1):
                all_displacements.append((0, 0, k))
        else:
            for i in range(-window_size, window_size + 1):
                for j in range(-window_size, window_size + 1):
                    for k in range(-window_size, window_size + 1):
                        all_displacements.append((i, j, k))

        return all_displacements
