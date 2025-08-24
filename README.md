# Simple ADK Agent on Cloud Run

## Play with it live
https://www.kaybee.ai/

## Run locally
```bash
uv run uvicorn server:app
```

## Deploy Agent to Cloud Run

```bash
chmod +x build.sh
./build.sh
```

## Configure for Load Test

Deploy new revision of the agent to Cloud Run.

```bash
gcloud run deploy weather-agent \
                  --source . \
                  --port 8080 \
                  --project {YOUR_PROJECT_ID} \
                  --allow-unauthenticated \
                  --region us-central1 \
                  --concurrency 10
```

Trigger the Locust load test with the following command:

```bash
locust -f load_test.py \
-H {YOUR_CLOUD_RUN_SERVICE_URL} \
--headless \
-t 120s -u 60 -r 5 \
--csv=.results/results \
--html=.results/report.html
```
