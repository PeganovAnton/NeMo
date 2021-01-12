# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from pytorch_lightning.metrics import Metric

__all__ = ['Loss']


class Loss(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False, process_group=None, take_avg_loss=True):
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group
        )
        self.add_state("loss_sum", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state("num_measurements", torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')
        self.take_avg_loss = take_avg_loss

    def update(self, loss, num_measurements):
        if self.take_avg_loss:
            loss *= num_measurements
        self.loss_sum += loss
        self.num_measurements += num_measurements

    def compute(self):
        if self.num_measurements.eq(0):
            return torch.tensor(float('nan'))
        return self.loss_sum / self.num_measurements
