from abc import ABC, abstractmethod


class FeatureSchedBase(ABC):
    '''
    Abstract class for manager feature scheduling. It should provide a global
    interface for getting new features to be sent for a worker for trials and
    acknowledge features that were already committed, i.e., should not be 
    tried again.
    '''
    def __init__(self, config):

        self._config = config

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

        raise Exception("[FeatureSchedBase][begin_iteration] "
                        "Abstract method not implemented.")

    def has_features(self):
        '''
        Returns whether there are still features to be tried.
        '''

        raise Exception("[FeatureSchedBase][has_features] "
                        "Abstract method not implemented.")

    def get_feature(self, rank):
        '''
        Returns a feature to be tried. This feature is returned based on 
        the concrete scheduling algorithm. Every feature can only be 
        returned once. Afterwards, it can only be returned after 
        self.begin_iteration().
        '''

        raise Exception("[FeatureSchedBase][get_feature] "
                        "Abstract method not implemented.")

    def tried_feature(self, feature, rank):
        '''
        Marks the end of a trial. This represents that the rank which
        performed a trial with 'feature' has finished. This means that
        if the feature data was to be evicted from memory, there 
        wouldn't be any issues.
        '''

        raise Exception("[FeatureSchedBase][get_feature] "
                        "Abstract method not implemented.")

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

        raise Exception("[FeatureSchedBase][commit_feature] "
                        "Abstract method not implemented.")
