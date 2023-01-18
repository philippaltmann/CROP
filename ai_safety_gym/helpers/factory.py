# Copyright 2018 The AI Safety Gridworlds Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Module containing factory class to instantiate all pycolab environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ai_safety_gym.environments.distributional_shift import DistributionalShiftEnvironment
_environment_classes = { 'distributional_shift': DistributionalShiftEnvironment }


def get_environment_obj(name, *args, **kwargs):
  """Instantiate a pycolab environment by name.

  Args:
    name: Name of the pycolab environment.
    *args: Arguments for the environment class constructor.
    **kwargs: Keyword arguments for the environment class constructor.

  Returns:
    A new environment class instance.
  """
  environment_class = _environment_classes.get(name.lower(), None)

  if environment_class:
    return environment_class(*args, **kwargs)
  raise NotImplementedError(
      'The requested environment is not available.')
