# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import logging
import time
from collections import defaultdict, deque

import dinov2.distributed as distributed
import torch

logger = logging.getLogger("dinov2")


class MetricLogger(object):
    """A class to log metrics during training.

    This class provides functionality to track and log various metrics
    during the training process, including synchronization across processes.

    Attributes:
        meters (defaultdict): A dictionary to hold smoothed values for metrics.
        delimiter (str): The delimiter used for logging output.
        output_file (str, optional): The file to which logs will be written.
    """

    def __init__(self, delimiter="\t", output_file=None):
        """Initializes the MetricLogger.

        Args:
            delimiter (str): The delimiter used for logging output. Default is tab.
            output_file (str, optional): The file to which logs will be written.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file

    def update(self, **kwargs):
        """Updates the metrics with new values.

        Args:
            **kwargs: Key-value pairs where keys are metric names and values are the corresponding values.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Gets the attribute from meters or the instance.

        Args:
            attr (str): The attribute name to retrieve.

        Returns:
            The value of the attribute.

        Raises:
            AttributeError: If the attribute is not found.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        """Returns a string representation of the logged metrics.

        Returns:
            str: A formatted string of all metrics.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronizes the metrics across distributed processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Adds a new meter to the logger.

        Args:
            name (str): The name of the meter.
            meter (SmoothedValue): The meter to be added.
        """
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        """Dumps the current metrics to the output file.

        Args:
            iteration (int): The current iteration number.
            iter_time (float): The time taken for the current iteration.
            data_time (float): The time taken for data loading.
        """
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        """Logs metrics at specified intervals during iteration.

        Args:
            iterable (iterable): The iterable to loop over.
            print_freq (int): The frequency of logging.
            header (str, optional): An optional header for the log message.
            n_iterations (int, optional): Total number of iterations. If None, it will be set to the length of iterable.
            start_iteration (int, optional): The starting iteration number. Default is 0.
        """
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {} ({:.6f} s / it)".format(header, total_time_str, total_time / n_iterations))


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window or
    the global series average.

    This class maintains a deque of values and computes the median, average, and global average.

    Attributes:
        window_size (int): The size of the window for smoothing.
        fmt (str): The format string for displaying values.
        deque (deque): A deque to store the values.
        total (float): The total sum of values.
        count (int): The number of values added.
    """

    def __init__(self, window_size=20, fmt=None):
        """Initializes the SmoothedValue.

        Args:
            window_size (int): The size of the window for smoothing. Default is 20.
            fmt (str, optional): The format string for displaying values. Default is "{median:.4f} ({global_avg:.4f})".
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        """Updates the smoothed value with a new value.

        Args:
            value (float): The new value to add.
            num (int, optional): The number of times to add the value. Default is 1.
        """
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """Distributed synchronization of the metric.

        Warning: does not synchronize the deque!
        """
        if not distributed.is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """Calculates the median of the stored values.

        Returns:
            float: The median value.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Calculates the average of the stored values.

        Returns:
            float: The average value.
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """Calculates the global average of all values.

        Returns:
            float: The global average value.
        """
        return self.total / self.count

    @property
    def max(self):
        """Gets the maximum value from the stored values.

        Returns:
            float: The maximum value.
        """
        return max(self.deque)

    @property
    def value(self):
        """Gets the most recent value added.

        Returns:
            float: The most recent value.
        """
        return self.deque[-1]

    def __str__(self):
        """Returns a formatted string representation of the smoothed values.

        Returns:
            str: A formatted string of the smoothed values.
        """
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
