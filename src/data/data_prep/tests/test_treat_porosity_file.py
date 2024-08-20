from unittest import TestCase, main
from treat_porosity_file import agg_porosities, AggregationStrategy, AggregationParams

import numpy as np
import pandas as pd


class TestAggData(TestCase):

    def setUp(self):
        self.df = pd.DataFrame()
        self.df['depth'] = np.arange(0, 5, 0.3)
        # This column represents every other column beeing aggregated
        self.df['pors'] = np.arange(0, 5, 0.3)
        self.df['area_x'] = 1
        self.df['area_y'] = 2

    def test_dont_aggregate(self):
        agg_params = AggregationParams(AggregationStrategy["NONE"], 1, 1)
        agg_df = agg_porosities(self.df, agg_params)
        self.assertTrue(self.df.equals(agg_df))

    def test_agg_rolling_window(self):
        rolling_w = 5
        agg_params = AggregationParams(AggregationStrategy["M_O_R"], rolling_w,
                                       1)
        agg_df = agg_porosities(self.df, agg_params)
        expected_pors = [
            0.3, 0.45, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3, 3.3, 3.6, 3.9,
            4.2, 4.35, 4.5
        ]
        self.assertTrue(np.allclose(expected_pors, agg_df['pors']))
    
    def test_agg_n_meters(self):
        n_meters=1
        agg_params = AggregationParams(AggregationStrategy["M_O_N"], 1,
                                       n_meters)
        agg_df = agg_porosities(self.df, agg_params)
        expected_pors = [0.45, 1.5, 2.4, 3.45, 4.5]
        self.assertTrue(np.allclose(expected_pors, agg_df['pors']))

        n_meters=2
        agg_params = AggregationParams(AggregationStrategy["M_O_N"], 1,
                                       n_meters)
        agg_df = agg_porosities(self.df, agg_params)
        expected_pors = [0.9, 3.0, 4.5]
        self.assertTrue(np.allclose(expected_pors, agg_df['pors']))


if __name__ == "__main__":
    main()
