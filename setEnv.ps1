$env:PATH="C:\g\bin;C:\CUDA\CUDA101Dev\bin;C:\CUDA\CUDA101Dev\extras\CUPTI\libx64;C:\CUDA\CUDA101Dev\include;$env:PATH"
$env:PYTHONPATH=$(Get-Location)

$env:PIPENV_IGNORE_VIRTUALENVS="yes"
$env:PIPENV_VENV_IN_PROJECT="yes"
$env:POETRY_VIRTUALENVS_IN_PROJECT="true"
$env:POETRY_VIRTUALENVS_CREATE="true"

$env:JUPYTER_CONFIG_DIR="${PWD}\.jupyter\config"
$env:JUPYTER_RUNTIME_DIR="${PWD}\.jupyter\runtime"
$env:JUPYTER_DATA_DIR="${PWD}\.jupyter\data"

# this forces NO GPU - deal with the OOM issues.
$env:CUDA_VISIBLE_DEVICES="-1"

.venv\Scripts\Activate.ps1