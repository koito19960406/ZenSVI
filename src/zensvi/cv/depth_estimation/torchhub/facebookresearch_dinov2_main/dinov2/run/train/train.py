# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

from dinov2.logging import setup_logging
from dinov2.run.submit import get_args_parser, submit_jobs
from dinov2.train import get_args_parser as get_train_args_parser

logger = logging.getLogger("dinov2")


class Trainer(object):
    """Trainer class for managing the training process.

    This class is responsible for setting up the training arguments,
    executing the training process, and handling checkpointing.

    Attributes:
        args: The arguments for training, including configuration and output directory.
    """

    def __init__(self, args):
        """Initializes the Trainer with the given arguments.

        Args:
            args: The arguments for training.
        """
        self.args = args

    def __call__(self):
        """Executes the training process by calling the main training function."""
        from dinov2.train import main as train_main

        self._setup_args()
        train_main(self.args)

    def checkpoint(self):
        """Creates a checkpoint for the current training state.

        Returns:
            A DelayedSubmission object that can be used to requeue the training job.
        """
        import submitit

        logger.info(f"Requeuing {self.args}")
        empty = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty)

    def _setup_args(self):
        """Sets up the training arguments, including output directory and logging.

        This method modifies the output directory to include the job ID
        and logs the process group information and the arguments.
        """
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.output_dir.replace("%j", str(job_env.job_id))
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Args: {self.args}")


def main():
    """Main function to launch the DINOv2 training process.

    This function sets up argument parsing, logging, and submits the training job.

    Returns:
        int: Exit status code.
    """
    description = "Submitit launcher for DINOv2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    setup_logging()

    assert os.path.exists(args.config_file), "Configuration file does not exist!"
    submit_jobs(Trainer, args, name="dinov2:train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
