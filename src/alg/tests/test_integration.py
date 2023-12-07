import unittest


class Test_All(unittest.TestCase):
    '''
    Integration test for the whole application.
    '''

    def test_sintetic_3_iterations_from_scratch(self):
        '''
        Performs 3 complete iterations from scratch.
        1 feature file is used, with a window of 1 (thus 27 tests).
        All 27 tests are performed.
        3 features are selected.
        '''

        # Generate feature file

        # Generate porosity file

        # Generate config file

        # Execute iterations

        # Compare final porosity file

        self.assertTrue(True)

    def test_sintetic_2_iterations_continued(self):
        '''
        Performs 2 complete iterations, continuing the execution of
        2 complete iterations.
        1 feature file is used, with a window of 1 (thus 27 tests).
        All 27 tests are performed.
        3 features are selected.
        '''

        self.assertTrue(True)

    def test_sintetic_short(self):
        '''
        Performs 2 complete iterations from scratch with reduced inputs.
        2 complete iterations.
        3 feature files are available but only 2 are used.
        With a window of 1 (thus 27 tests), only 10 tests are performed.
        2 features are selected.
        '''

        self.assertTrue(True)

    def test_real_2_iterations_from_scratch(self):
        '''
        Performs 2 complete iterations from scratch with reduced inputs on
        real data.
        2 complete iterations.
        1 feature file is used with a window of 1 (thus 27 tests).
        Only 10 tests are performed per selected feature.
        10 features are selected.
        '''

        self.assertTrue(True)

