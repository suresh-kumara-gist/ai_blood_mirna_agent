# Docker Compose Setup

## 1. Build and start Streamlit

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

## 2. Train the demo model

Option A: Train inside the Streamlit sidebar by clicking **Train demo model**.

Option B: Train from Docker CLI:

```bash
docker compose run --rm train-demo
```

Then restart the app if needed:

```bash
docker compose up
```

## 3. Persistent files

The compose file mounts:

```text
./artifacts -> /app/artifacts
./data      -> /app/data
```

So trained models and generated CSV files remain on your machine after the container stops.

## 4. Stop

```bash
docker compose down
```

## 5. Rebuild after code changes

```bash
docker compose up --build
```


## Fix note

Training is executed with `python -m model.train` so imports resolve from the project root inside Docker. The source `data/` package is not volume-mounted, because mounting it can accidentally hide the Python package inside the container. Put uploaded datasets in another folder such as `datasets/` and pass that path when extending the compose file.
