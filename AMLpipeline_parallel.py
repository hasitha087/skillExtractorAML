'''
Run script in AML service in parallel mode
and expose the pipeline as a REST end point.
'''

from azureml.core.dataset import Dataset
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter, ScheduleRecurrence, Schedule
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.core.runconfig import RunConfiguration, DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import Environment
from azureml.core import Experiment, Run, Workspace, Datastore
from azureml.pipeline.steps import PythonScriptStep, ParallelRunConfig, ParallelRunStep


ws = Workspace.from_config()

datastore = Datastore.get(ws, 'adlsgen2store') # Azure Data Lake storage
datastore_paths = [(datastore, 'file_path/*')] # Get all the files in file_path/
emp_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
registered_emp_ds = emp_ds.register(ws, 'skill_input', create_new_version=True) # Register data under AML Dataset
named_emp_ds = registered_emp_ds.as_named_input('skill_inout')

# Define output path - Output data save in AML default Blob storage
default_ds = ws.get_default_datastore()
output_dir = PipelineData(
    name='skill_recommendation',
    datastore=default_ds
)

aml_run_config = Environment(name='batch_environment')
comput_target = 'amlvmcluster' # AML compute cluster name

aml_run_config.docker.enabled = True
aml_run_config.docker._base_image = DEFAULT_CPU_IMAGE

# Add dependancies
aml_run_config.python.conda_dependencies = CondaDependencies.create(
    conda_packages=['pandas', 'scikit-learn'],
    pip_packages=['azureml-sdk', 'azureml-dataprep[fuse, pandas]', 'azureml-dataset-runtime[pandas, fuse]',
    'tensorflow', 'keras', 'textblob', 'nltk', 'fuzzywuzzy', 'azureml-defaults', 'azureml-core'],
    pin_sdk_version=False
)

parallel_run_config = ParallelRunConfig(
    source_directory='./',
    entry_script='skill_recommender_AML.py',
    mini_batch_size='5KB',
    error_threshold=-1,
    output_action='append_row',
    environment=aml_run_config,
    comput_target=comput_target,
    process_count_per_node=PipelineParameter(name='process_count_param', default_value=2),
    node_count=2,
    run_invocation_timeout=600
)

parallelrun_step = ParallelRunStep(
    name='skill-extractor-parallel',
    parallel_run_config=parallel_run_config,
    inputs=[named_emp_ds],
    output=output_dir,
    allow_reuse=True
)

pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
experiment = Experiment(ws, 'skill-extractor-parallel')
pipeline_run = experiment.submit(pipeline)

pipeline_run.wait_for_completion(show_output=True)

# Publish pipeline as REST API
publised_pipeline = pipeline_run.publish_pipeline(
    name='skill_extractor',
    description='Extract skill from employees JD text',
    version='1.0.0'
)

# Print REST end point
rest_endpoint = publised_pipeline.endpoint
print(rest_endpoint)

# Schedule pipeline weekly
daily = ScheduleRecurrence(frequency='week', interval=1)
pipeline_schedule = Schedule.create(
    ws,
    name='skill extractor weekly run',
    description='skill extractor weekly scheduler',
    pipeline_id=publised_pipeline.id,
    experiment_name='skill-extractor-parallel',
    recurrence=daily
)