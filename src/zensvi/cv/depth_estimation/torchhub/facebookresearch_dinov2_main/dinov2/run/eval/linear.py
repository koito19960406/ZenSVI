# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

from dinov2.eval.linear import get_args_parser as get_linear_args_parser
from dinov2.logging import setup_logging
from dinov2.run.submit import get_args_parser, submit_jobs

logger = logging.getLogger("dinov2")


class Evaluator:
    """Evaluator class for managing the evaluation process.

    Args:
        args: Command line arguments for the evaluation.
    """

    def __init__(self, args):
        """Initializes the Evaluator with the given arguments.

        Args:
            args: Command line arguments for the evaluation.
        """
        self.args = args

    def __call__(self):
        """Executes the evaluation process by setting up arguments and calling the linear evaluation main function."""
        from dinov2.eval.linear import main as linear_main

        self._setup_args()
        linear_main(self.args)

    def checkpoint(self):
        """Creates a checkpoint for the evaluation process.

        Returns:
            A delayed submission for the evaluation job.
        """
        import submitit

        logger.info(f"Requeuing {self.args}")
        empty = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty)

    def _setup_args(self):
        """Sets up the arguments for the evaluation job, including job environment details."""
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.output_dir.replace("%j", str(job_env.job_id))
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Args: {self.args}")


def main():
    """Main function to launch the DINOv2 linear evaluation.

    Returns:
        int: Exit status code.
    """
    description = "Submitit launcher for DINOv2 linear evaluation"
    linear_args_parser = get_linear_args_parser(add_help=False)
    parents = [linear_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    setup_logging()

    assert os.path.exists(args.config_file), "Configuration file does not exist!"
    submit_jobs(Evaluator, args, name="dinov2:linear")
    return 0


if __name__ == "__main__":
    sys.exit(main())
