# Advanced V2 training and artifact export

V2 ports the notebook's parallel feature branches into reproducible code:

```text
SMS -> frequent-itemset word features --------------------+
SMS -> fine-tuned DistilBERT -> HDBSCAN soft assignment --+-> classifier
```

V1 and the currently deployed API remain unchanged until V2 has been trained,
reloaded in a fresh process, benchmarked, and explicitly connected to the API.

## 1. Install the advanced environment

From the repository root in VS Code PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-advanced.txt
```

If CUDA is required, install the appropriate PyTorch build for the local NVIDIA
driver using the official PyTorch selector instead of relying on the generic
wheel above.

## 2. Train and export every inference artifact

```powershell
.\.venv\Scripts\python.exe -m src.advanced.train
```

Force CPU if GPU/CUDA setup is unavailable:

```powershell
.\.venv\Scripts\python.exe -m src.advanced.train --cpu
```

The command creates:

```text
artifacts/advanced-v2/
├── bert_model/              # fine-tuned Transformer weights/config
├── tokenizer/               # matching tokenizer files
├── pipeline_bundle.joblib   # itemsets, exemplars, scaler, classifier
├── metadata.json            # dimensions, order, labels, threshold, version
└── metrics.json             # candidate and selected model metrics
```

These files are intentionally gitignored because the BERT weights are large.

## 3. Prove reload works without retraining

Close the training terminal, open a fresh terminal, then run:

```powershell
.\.venv\Scripts\python.exe -m src.advanced.predict `
  --text "Congratulations! Call 123456 to claim £500."
```

This command must load only the saved artifacts and return a prediction. It
must not run Apriori, fine-tuning, HDBSCAN fitting, scaler fitting, or classifier
training.

## 4. Inspect results

```powershell
Get-Content artifacts\advanced-v2\metadata.json
Get-Content artifacts\advanced-v2\metrics.json
```

Record training hardware, elapsed time, peak memory, and inference latency
before deciding whether the B1 App Service plan is sufficient.

## 5. Register the trained model artifact once

V2 artifacts are gitignored and exist only on the training computer. Publish
the verified artifact set once to the private ACR model repository:

```powershell
az acr login --name spamclassifier0725
docker build `
  --file Dockerfile.model `
  --tag spamclassifier0725.azurecr.io/spam-model:advanced-v2-20260813-200127 `
  .
docker push spamclassifier0725.azurecr.io/spam-model:advanced-v2-20260813-200127
```

This does not deploy the application. It registers the immutable model artifact
that GitHub CD needs because GitHub cannot access files on the training PC.

## 6. Automatic V2 application deployment

After the model-artifact image exists, commit and push the V2 code. CI checks
the code. CD then logs into ACR, pulls the fixed model artifact, builds the V2
application image with `Dockerfile.v2`, starts it on the GitHub runner, verifies
that `/health` reports `model_backend=advanced`, pushes the verified image, and
updates Azure Web App.
