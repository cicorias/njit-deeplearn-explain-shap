
$env:PATH="E:\g\bin;E:\CUDA\CUDA101Dev\bin;E:\CUDA\CUDA101Dev\extras\CUPTI\libx64;E:\CUDA\CUDA101Dev\include;$env:PATH"
$env:PYTHONPATH=$(Get-Location)

$env:PIPENV_IGNORE_VIRTUALENVS="yes"
$env:PIPENV_VENV_IN_PROJECT="yes"
$env:POETRY_VIRTUALENVS_IN_PROJECT="true"
$env:POETRY_VIRTUALENVS_CREATE="true"

$env:JUPYTER_CONFIG_DIR="${PWD}\.jupyter\config"
$env:JUPYTER_RUNTIME_DIR="${PWD}\.jupyter\runtime"
$env:JUPYTER_DATA_DIR="${PWD}\.jupyter\data"

.\py369A\Scripts\Activate.ps1