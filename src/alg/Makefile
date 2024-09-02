test:
	coverage run -m unittest discover tests
	coverage html

test_config_parser:
	python3 -m unittest tests.test_config_parser

test_data_filter:
	python3 -m unittest tests.test_data_filter

test_feature_data:
	pytest tests/test_FeatureData.py

test_integration:
	pytest tests/test_integration.py

test_trial_data:
	pytest tests/test_TrialData.py

test_sampler:
	python3 -m unittest tests.test_sampler

test_propagate:
	python3 -m unittest tests.test_propagate