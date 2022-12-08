Create Virtual Environment (Linux)
```bash
python3 -m venv ./venv
```

Activate Virtual Environment (Linux)
```bash
source ./venv/bin/activate
```

Install Dependencies
```bash
pip install -r requirements.txt
```

Install MLflow
```bash 
pip install mlflow
```

To run MLflow UI
```bash
mlflow ui
```

To serve the model
```bash
mlflow models serve -m ./mlruns/0/{model_id}/artifacts/model/ -p 1234 --no-conda
```

Request to the model (Example 1)
```bash
curl -X POST -H 'Content-Type: application/json' --data '{"dataframe_split":{"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]]}}' http://127.0.0.1:1234/invocations
```

Request to the model (Example 2)
```bash
curl -X POST -H 'Content-Type: application/json' --data '{"dataframe_split":{"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[5.2,0.48,0.04,1.6,0.054000000000000006,19.0,106.0,0.9927,3.54,0.62,12.2]]}}' http://127.0.0.1:1234/invocations
```

Request to the model (Example 3)
```bash
curl -X POST -H 'Content-Type: application/json' --data '{"dataframe_split":{"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[12.5,0.28,0.54,2.3,0.08199999999999999,12.0,29.0,0.9997,3.11,1.36,9.8]]}}' http://127.0.0.1:1234/invocations
```
