from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from ray.tune.experiment import Experiment, process_experiments
from ray.tune.error import TuneError


class ExperimentTest(unittest.TestCase):
    def testConvertExperimentFromExperiment(self):
        exp1 = Experiment(**{
            "name": "foo",
            "run": "f1",
            "config": {
                "script_min_iter_time_s": 0
            }
        })
        result = process_experiments(exp1)
        self.assertEqual(len(result), 1)
        self.assertEqual(type(result), list)

    def testConvertExperimentNone(self):
        result = process_experiments(None)
        self.assertEqual(len(result), 0)
        self.assertEqual(type(result), list)

    def testConvertExperimentList(self):
        exp1 = Experiment(**{
            "name": "foo",
            "run": "f1",
            "config": {
                "script_min_iter_time_s": 0
            }
        })
        result = process_experiments([exp1, exp1])
        self.assertEqual(len(result), 2)
        self.assertEqual(type(result), list)

    def testConvertExperimentJSON(self):
        experiment = {
            "name": {
                "run": "f1",
                "config": {
                    "script_min_iter_time_s": 0
                }
            },
            "named": {
                "run": "f1",
                "config": {
                    "script_min_iter_time_s": 0
                }
            }
        }
        result = process_experiments(experiment)
        self.assertEqual(len(result), 2)
        self.assertEqual(type(result), list)

    def testConvertExperimentIncorrect(self):
        self.assertRaises(TuneError, lambda: process_experiments("hi"))
