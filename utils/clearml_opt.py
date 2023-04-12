from clearml import	Task
from clearml.automation.optuna import OptimizerOptuna
import optuna
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, DiscreteParameterRange

task = Task.init(
	project_name='Hyper Parameter Optimization',
	task_name='yolov8n_50_opt',
	task_type=Task.TaskTypes.optimizer,
	reuse_last_task_id=False)

my_optimizer = HyperParameterOptimizer(
	base_task_id=Task.get_task(project_name='OilTankDetecting', task_name='yolov8n_50e').id,
	hyper_parameters=[
		UniformParameterRange('Hyperparameters/lr0', min_value=1e-5, max_value=1e-1),
		UniformParameterRange('Hyperparameters/lrf', min_value=0.01, max_value=1.0),
		UniformParameterRange('Hyperparameters/momentum', min_value=0.6, max_value=0.98),
		UniformParameterRange('Hyperparameters/weight_decay', min_value=0.0, max_value=0.001),
		UniformParameterRange('Hyperparameters/warmup_epochs', min_value=0.0, max_value=5.0),
		UniformParameterRange('Hyperparameters/warmup_momentum', min_value=0.0, max_value=0.95),
		UniformParameterRange('Hyperparameters/warmup_bias_lr', min_value=0.0, max_value=0.2),
		UniformParameterRange('Hyperparameters/box', min_value=0.02, max_value=0.2),
		UniformParameterRange('Hyperparameters/cls', min_value=0.2, max_value=4.0),
		UniformParameterRange('Hyperparameters/cls_pw', min_value=0.5, max_value=2.0),
		UniformParameterRange('Hyperparameters/obj', min_value=0.2, max_value=4.0),
		UniformParameterRange('Hyperparameters/obj_pw', min_value=0.5, max_value=2.0),
		UniformParameterRange('Hyperparameters/iou_t', min_value=0.1, max_value=0.7),
		UniformParameterRange('Hyperparameters/anchor_t', min_value=2.0, max_value=8.0),
		UniformParameterRange('Hyperparameters/fl_gamma', min_value=0.0, max_value=4.0),
		DiscreteParameterRange('Hyperparameters/fl_gamma', ['SGD', 'Adam'])
	],
	objective_metric_title='metrics',
	objective_metric_series='mAP50',
	objective_metric_sign='max',
	max_number_of_concurrent_tasks=1,
	optimizer_class= OptimizerOptuna,
	save_top_k_tasks_only=5
)
my_optimizer.set_report_period(10 / 60)
# You can also use the line below instead to run all the optimizer tasks locally, without using queues or agent
# an_optimizer.start_locally(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)
my_optimizer.set_time_limit(in_minutes=120.0)
# Start the optimization process in the local environment
my_optimizer.start_locally()
# wait until process is done (notice we are controlling the optimization process in the background)
my_optimizer.wait()
# make sure background optimization stopped
my_optimizer.stop()

print('We are done, good bye')


print(Task.get_task(project_name='OilTankDetecting', task_name='yolov8n_50e').id)
