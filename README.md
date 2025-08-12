# Deploy, Manage, and Observe ADK Agent on Cloud Run

## Deploy Agent to Cloud Run

This deployment script will deploy the agent to Cloud Run. Make sure to replace the placeholders with your actual values. This assumes you have already created a Cloud SQL instance.

```bash
gcloud run deploy weather-agent \
                  --source . \
                  --port 8080 \
                  --project {YOUR_PROJECT_ID} \
                  --allow-unauthenticated \
                  --add-cloudsql-instances {YOUR_DB_CONNECTION_NAME} \
                  --update-env-vars SESSION_SERVICE_URI="postgresql+pg8000://postgres:{YOUR_DEFAULT_USER_PASS}@postgres/?unix_sock=/cloudsql/{YOUR_DB_CONNECTION_NAME}/.s.PGSQL.5432",GOOGLE_CLOUD_PROJECT={YOUR_PROJECT_ID} \
                  --region us-central1 \
                  --min 1
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

## Deploy New Revision without Traffic

Let's deploy a new revision of the agent to Cloud Run without traffic.

```bash
gcloud run deploy weather-agent \
                  --source . \
                  --port 8080 \
                  --project {YOUR_PROJECT_ID} \
                  --allow-unauthenticated \
                  --region us-central1 \
                  --concurrency 10 \
                  --no-traffic
```
