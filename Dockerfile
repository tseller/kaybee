# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim
EXPOSE 8080

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /python_app
WORKDIR $APP_HOME

# Install production dependencies.
RUN pip install pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv sync --system

COPY . ./
WORKDIR $APP_HOME/app

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
