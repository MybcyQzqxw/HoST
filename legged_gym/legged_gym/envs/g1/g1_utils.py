"""
DEPRECATED: This module has been moved to legged_gym.utils.

Please update your imports:
    - MotionLib, load_imitation_dataset, compute_residual_observations 
      -> from legged_gym.utils.motion_lib import ...
    - sigmoid, tolerance 
      -> from legged_gym.utils.math import sigmoid_reward, tolerance

This file is kept for backward compatibility only.
"""

import warnings

warnings.warn(
    "g1_utils is deprecated. Please import from:\n"
    "  - legged_gym.utils.motion_lib (MotionLib, load_imitation_dataset, compute_residual_observations)\n"
    "  - legged_gym.utils.math (sigmoid_reward as sigmoid, tolerance)",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
from legged_gym.utils.motion_lib import (
    MotionLib,
    load_imitation_dataset,
    compute_residual_observations,
)
from legged_gym.utils.math import (
    sigmoid_reward as sigmoid,
    tolerance,
)