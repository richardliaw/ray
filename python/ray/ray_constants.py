from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Ray constants used in the Python code."""

import os


def env_integer(key, default):
    if key in os.environ:
        return int(os.environ[key])
    return default


ID_SIZE = 20

# The default maximum number of bytes to allocate to the object store unless
# overridden by the user.
DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = 20 * 10**9
# The smallest cap on the memory used by the object store that we allow.
OBJECT_STORE_MINIMUM_MEMORY_BYTES = 10**7
# The default maximum number of bytes that the non-primary Redis shards are
# allowed to use unless overridden by the user.
DEFAULT_REDIS_MAX_MEMORY_BYTES = 10**10
# The smallest cap on the memory used by Redis that we allow.
REDIS_MINIMUM_MEMORY_BYTES = 10**7

# If a remote function or actor (or some other export) has serialized size
# greater than this quantity, print an warning.
PICKLE_OBJECT_WARNING_SIZE = 10**7

# The maximum resource quantity that is allowed. TODO(rkn): This could be
# relaxed, but the current implementation of the node manager will be slower
# for large resource quantities due to bookkeeping of specific resource IDs.
MAX_RESOURCE_QUANTITY = 512

# Different types of Ray errors that can be pushed to the driver.
# TODO(rkn): These should be defined in flatbuffers and must be synced with
# the existing C++ definitions.
WAIT_FOR_CLASS_PUSH_ERROR = "wait_for_class"
PICKLING_LARGE_OBJECT_PUSH_ERROR = "pickling_large_object"
WAIT_FOR_FUNCTION_PUSH_ERROR = "wait_for_function"
TASK_PUSH_ERROR = "task"
REGISTER_REMOTE_FUNCTION_PUSH_ERROR = "register_remote_function"
FUNCTION_TO_RUN_PUSH_ERROR = "function_to_run"
VERSION_MISMATCH_PUSH_ERROR = "version_mismatch"
CHECKPOINT_PUSH_ERROR = "checkpoint"
REGISTER_ACTOR_PUSH_ERROR = "register_actor"
WORKER_CRASH_PUSH_ERROR = "worker_crash"
WORKER_DIED_PUSH_ERROR = "worker_died"
WORKER_POOL_LARGE_ERROR = "worker_pool_large"
PUT_RECONSTRUCTION_PUSH_ERROR = "put_reconstruction"
INFEASIBLE_TASK_ERROR = "infeasible_task"
REMOVED_NODE_ERROR = "node_removed"
MONITOR_DIED_ERROR = "monitor_died"

# Abort autoscaling if more than this number of errors are encountered. This
# is a safety feature to prevent e.g. runaway node launches.
AUTOSCALER_MAX_NUM_FAILURES = env_integer("AUTOSCALER_MAX_NUM_FAILURES", 5)

# The maximum number of nodes to launch in a single request.
# Multiple requests may be made for this batch size, up to
# the limit of AUTOSCALER_MAX_CONCURRENT_LAUNCHES.
AUTOSCALER_MAX_LAUNCH_BATCH = env_integer("AUTOSCALER_MAX_LAUNCH_BATCH", 5)

# Max number of nodes to launch at a time.
AUTOSCALER_MAX_CONCURRENT_LAUNCHES = env_integer(
    "AUTOSCALER_MAX_CONCURRENT_LAUNCHES", 10)

# Interval at which to perform autoscaling updates.
AUTOSCALER_UPDATE_INTERVAL_S = env_integer("AUTOSCALER_UPDATE_INTERVAL_S", 5)

# The autoscaler will attempt to restart Ray on nodes it hasn't heard from
# in more than this interval.
AUTOSCALER_HEARTBEAT_TIMEOUT_S = env_integer("AUTOSCALER_HEARTBEAT_TIMEOUT_S",
                                             30)

# Max number of retries to AWS (default is 5, time increases exponentially)
BOTO_MAX_RETRIES = env_integer("BOTO_MAX_RETRIES", 12)

# Default logger format: only contains the message.
LOGGER_FORMAT = "%(message)s"
LOGGER_FORMAT_HELP = "The logging format. default='%(message)s'"
LOGGER_LEVEL = "info"
LOGGER_LEVEL_CHOICES = ['debug', 'info', 'warning', 'error', 'critical']
LOGGER_LEVEL_HELP = ("The logging level threshold, choices=['debug', 'info',"
                     " 'warning', 'error', 'critical'], default='info'")

# A constant indicating that an actor doesn't need reconstructions.
NO_RECONSTRUCTION = 0
# A constant indicating that an actor should be reconstructed infinite times.
INFINITE_RECONSTRUCTION = 2**30

# Constants used to define the different process types.
PROCESS_TYPE_MONITOR = "monitor"
PROCESS_TYPE_RAYLET_MONITOR = "raylet_monitor"
PROCESS_TYPE_LOG_MONITOR = "log_monitor"
PROCESS_TYPE_WORKER = "worker"
PROCESS_TYPE_RAYLET = "raylet"
PROCESS_TYPE_PLASMA_STORE = "plasma_store"
PROCESS_TYPE_REDIS_SERVER = "redis_server"
PROCESS_TYPE_WEB_UI = "web_ui"
