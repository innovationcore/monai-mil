import logging

import optuna

from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformIntegerParameterRange, UniformParameterRange)

aSearchStrategy = None

if not aSearchStrategy:
    try:
        from clearml.automation.optuna import OptimizerOptuna
        aSearchStrategy = OptimizerOptuna
        logging.getLogger().info('Optimizer: OptimizerOptuna')
    except ImportError as ex:
        pass

if not aSearchStrategy:
    try:
        from clearml.automation.hpbandster import OptimizerBOHB
        aSearchStrategy = OptimizerBOHB
        logging.getLogger().info('Optimizer: OptimizerBOHB')
    except ImportError as ex:
        pass

if not aSearchStrategy:
    logging.getLogger().warning(
        'Apologies, it seems you do not have \'optuna\' or \'hpbandster\' installed, '
        'we will be using RandomSearch strategy instead')
    aSearchStrategy = RandomSearch

print('Seach aSearchStrategy: ' + str(aSearchStrategy))
def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


# Connecting TRAINS
task = Task.init(project_name='monai-mil',
                 task_name='HPO-Monai-MIL',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

# experiment template to optimize in the hyper-parameter optimization
args = {
    'template_task_id': None,
    'run_as_service': True,
}
args = task.connect(args)

# Get the template task experiment that we want to optimize
if not args['template_task_id']:
    args['template_task_id'] = Task.get_task(project_name='monai-mil', task_name='monai-mil_template').id

# Example use case:
an_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=args['template_task_id'],
    # here we define the hyper-parameters to optimize
    hyper_parameters=[
#--tile_count=33 --mil_mode=att_trans   --tile_size=768 --num_classes=1

        UniformIntegerParameterRange('Args/epochs', min_value=5, max_value=20, step_size=2),
        UniformIntegerParameterRange('Args/batch_size', min_value=1, max_value=16, step_size=2),
        UniformIntegerParameterRange('Args/tile_count', min_value=1, max_value=500, step_size=2),
        #UniformIntegerParameterRange('Args/tile_size', min_value=224, max_value=768, step_size=256),

        UniformParameterRange('Args/optim_lr', min_value=0.000001, max_value=0.0001),

        DiscreteParameterRange('Args/tile_size', values=[224,256,384,512,768]),
        DiscreteParameterRange('Args/mil_mode', values=['mean', 'max', 'att', 'att_trans']),


        #DiscreteParameterRange('Args/train_file', values=["/workspace/path_train.csv"]),
        #DiscreteParameterRange('Args/validation_file', values=["/workspace/path_val.csv"]),

    ],

    # this is the objective metric we want to maximize/minimize
    objective_metric_title='Loss',
    objective_metric_series='val_loss',

    #big label
    #objective_metric_title='accuracy',
    #actual line
    #objective_metric_series='total',

    # now we decide if we want to maximize it or minimize it (accuracy we maximize)
    objective_metric_sign='min',
    # let us limit the number of concurrent experiments,
    # this in turn will make sure we do dont bombard the scheduler with experiments.
    # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
    max_number_of_concurrent_tasks=1,
    # this is the optimizer class (actually doing the optimization)
    # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
    # more are coming soon...
    optimizer_class=aSearchStrategy,
    # Select an execution queue to schedule the experiments for execution
    execution_queue='default',
    # Optional: Limit the execution time of a single experiment, in minutes.
    # (this is optional, and if using  OptimizerBOHB, it is ignored)
    time_limit_per_job=240,
    # Check the experiments every 12 seconds is way too often, we should probably set it to 5 min,
    # assuming a single experiment is usually hours...
    pool_period_min=1,
    # set the maximum number of jobs to launch for the optimization, default (None) unlimited
    # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
    # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
    total_max_jobs=500,
    # set the minimum number of iterations for an experiment, before early stopping.
    # Does not apply for simple strategies such as RandomSearch or GridSearch
    min_iteration_per_job=10,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    max_iteration_per_job=150,
)

# if we are running as a service, just enqueue ourselves into the services queue and let it run the optimization
if args['run_as_service']:
    # if this code is executed by `trains-agent` the function call does nothing.
    # if executed locally, the local process will be terminated, and a remote copy will be executed instead
    task.execute_remotely(queue_name='services', exit_process=True)

# report every 12 seconds, this is way too often, but we are testing here J
an_optimizer.set_report_period(2.2)
# start the optimization process, callback function to be called every time an experiment is completed
# this function returns immediately
an_optimizer.start(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)
an_optimizer.set_time_limit(in_minutes=2880.0)
# wait until process is done (notice we are controlling the optimization process in the background)
an_optimizer.wait()
# optimization is completed, print the top performing experiments id
top_exp = an_optimizer.get_top_experiments(top_k=5)
print([t.id for t in top_exp])
# make sure background optimization stopped
an_optimizer.stop()

print('We are done, good bye')