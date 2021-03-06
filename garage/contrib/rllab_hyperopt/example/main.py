'''
Main module to launch an example hyperopt search on EC2.

Launch this from outside the garage main dir. Otherwise, garage will try to
ship the logfiles being written by this process, which will fail because tar
doesn't want to tar files that are being written to. Alternatively, disable the
packaging of log files by garage, but I couldn't quickly find how to do this.

You can use Jupyter notebook visualize_hyperopt_results.ipynb to inspect
results.
'''
from hyperopt import hp

from garage.contrib.garage_hyperopt.core import launch_hyperopt_search
# the functions to run the task and process Exp_paper do not need to be in
# separate files. They do need to be separate from the main file though. Also,
# anything you import in the module that contains run_task needs to be on the
# garage AMI. Therefore, since I use pandas to process results, I have put them
# in separate files here.
from garage.contrib.garage_hyperopt.example.score import process_result
from garage.contrib.garage_hyperopt.example.task import run_task

# define a search space.
# See https://github.com/hyperopt/hyperopt/wiki/FMin, sect 2 for more detail
param_space = {
    'step_size': hp.uniform('step_size', 0.01, 0.1),
    'seed': hp.choice('seed', [0, 1, 2])
}

# just by way of example, pass a different config to run_experiment
run_experiment_kwargs = dict(
    n_parallel=16,
    aws_config=dict(instance_type="c4.4xlarge", spot_price='0.7'))

launch_hyperopt_search(
    run_task,  # the task to run
    process_result,  # the function that processes results and returns a score
    param_space,  # param search space
    # key for hyperopt DB, and also exp_prefix for run_experiment
    hyperopt_experiment_key='test12',
    # number of local workers AND EC2 instances that are started in parallel
    n_hyperopt_workers=3,
    # nr of parameter values to eval
    hyperopt_max_evals=5,
    result_timeout=600,  # wait this long for results from S3 before timing out
    run_experiment_kwargs=run_experiment_kwargs
)  # additional kwargs to pass to run_experiment
