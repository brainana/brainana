# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thread defaults for torch and other in-process numerics (one implementation)."""

import os

__all__ = ["get_num_threads"]


def get_num_threads() -> int:
    """Default number of CPU threads for torch and other in-process numerics.

    OMP_NUM_THREADS wins when set: the pipeline exports it from the task's CPU
    allocation (Nextflow task.cpus), so torch matches what the scheduler reserved.
    Otherwise returns the number of available cores if <= 8, else 8, to avoid
    excessive thread overhead on systems with many cores.
    """
    omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    if omp.isdigit() and int(omp) > 0:
        return int(omp)
    try:
        num_cores = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        num_cores = os.cpu_count() or 1
    return min(num_cores, 8)
