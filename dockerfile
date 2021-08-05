FROM python:3.8-buster
# RUN pip install fastapi uvicorn

# 

# COPY ./app /app

# CMD ["uvicorn", "unimatch:app", "--host", "0.0.0.0", "--port", "8000"]

# System deps:
RUN pip install "poetry==1.1.6"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY pyproject.toml /code/

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Creating folders, and files for a project:
COPY ./unimatch /code/unimatch
WORKDIR /code/unimatch

EXPOSE 8000

CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8000"]