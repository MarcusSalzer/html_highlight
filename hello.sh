
echo "Get the venv..."
. .venv/bin/activate

alias panel='panel serve dashboard/panel_main.py --dev'
alias mlflow-serve='mlflow server --backend-store-uri sqlite:///data/mlflow.db'

echo "Write some code..."
code .
