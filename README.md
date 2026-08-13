# Spam Classifier MLOps

A deployable SMS spam-classification API built with scikit-learn, FastAPI,
Docker, GitHub Actions, and Microsoft Azure.

The first deployable version (`baseline-v1`) uses TF-IDF features and logistic
regression. The research notebook is retained under `notebooks/`; its
DistilBERT + HDBSCAN approach is planned as `advanced-v2` after the deployment
pipeline is proven.

## Architecture

```text
SMS text -> normalization -> TF-IDF -> logistic regression -> ham/spam
```

`src/train.py` trains and saves the model. `src/predict.py` loads and uses it.
`app/main.py` exposes it over HTTP.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m src.train
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> and try `POST /predict`.

Example request:

```json
{"text": "Congratulations! Claim your free prize now."}
```

## Test

```powershell
python -m src.train
pytest -q
ruff check app src tests
```

## Docker

The image trains the small baseline model during the build, so no binary model
artifact needs to be committed.

```powershell
docker build -t spam-classifier .
docker run --rm -p 8000:8000 spam-classifier
```

Then open <http://127.0.0.1:8000/docs>.

## API

- `GET /health` returns service and model status.
- `POST /predict` accepts one message of at most 5,000 characters.
- `GET /docs` provides interactive OpenAPI documentation.

## CI/CD status

`.github/workflows/ci.yml` runs linting, training, tests, and a Docker build for
every pull request and push to `main`. Azure continuous deployment is added
after the first manual Azure deployment, because the workflow will need the
actual Azure resource name and repository secrets.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the first manual Azure deployment.

## Data and privacy

The included SMS Spam Collection data is used for training. The API does not
persist submitted message text. Production logging must continue to avoid
recording message bodies because they may contain personal data.

## Roadmap

1. Deploy `baseline-v1` manually to Azure App Service.
2. Add Azure CD and a post-deployment health check.
3. Port and serialize the notebook's DistilBERT, cluster exemplars, scaler, and
   final classifier as `advanced-v2`.
4. Compare latency, F1, and cost before promoting `advanced-v2`.

The V2 training/export and fresh-process reload procedure is documented in
[ADVANCED_V2.md](ADVANCED_V2.md).
