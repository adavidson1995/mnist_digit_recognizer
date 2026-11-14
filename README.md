# MNIST Digit Recognizer

A small end-to-end machine learning app: draw a digit (0–9) in your browser and a convolutional neural network trained on MNIST predicts which number you wrote.

Built as the foundation project for my application to the ML Institute. The model is deliberately simple; the point is the full path from training a model, to serving it behind an API, to using it from a web frontend.

## What it does

1. You draw a digit on a 280×280 canvas in a Streamlit web app.
2. When you click **Predict**, the drawing is converted to grayscale, resized to 28×28 (the MNIST input size) and sent to a FastAPI backend.
3. The backend runs the image through a PyTorch CNN and returns the predicted digit with a confidence score.
4. You can enter the true digit and click **Submit** to add the result to a history table of your recent attempts.

## How it works

```
┌──────────────────────┐   POST /predict    ┌──────────────────────┐
│  Streamlit frontend  │  28×28 pixel array │   FastAPI backend    │
│     app/app.py       │ ─────────────────▶ │    model/api.py      │
│                      │ ◀───────────────── │                      │
│  drawing canvas,     │ digit + confidence │  PyTorch CNN         │
│  results, history    │                    │  (mnist_cnn.pth)     │
└──────────────────────┘                    └──────────┬───────────┘
                                                       │ SQLAlchemy
                                            ┌──────────▼───────────┐
                                            │      PostgreSQL      │
                                            │  predictions table   │
                                            └──────────────────────┘
```

### The model

`MNISTNet` (`model/model.py`) is a small two-layer convolutional network:

| Layer | Details |
|---|---|
| Conv2d | 1 → 32 channels, 3×3 kernel, ReLU |
| Conv2d + MaxPool | 32 → 64 channels, 3×3 kernel, 2×2 max pool, ReLU |
| Dropout2d + MaxPool | 2×2 max pool, ReLU |
| Linear | 2304 → 100, ReLU, dropout |
| Linear | 100 → 10, log-softmax |

It is trained by `model/train.py` for 10 epochs with SGD (learning rate 0.01, momentum 0.5, batch size 64). Inputs are normalised with the standard MNIST mean and standard deviation (0.1307, 0.3081). The script prints accuracy on the 10,000-image MNIST test set when training finishes.

### The API

| Endpoint | Description |
|---|---|
| `POST /predict` | Body: `{"image": <28×28 nested list of floats in [0, 1]>, "true_label": <optional int>}`. Returns `{"prediction": int, "confidence": float}`. If `true_label` is given, the prediction is saved to PostgreSQL. |
| `GET /history` | Returns the 50 most recent predictions saved in the database. |

While the backend is running, interactive API docs are at http://localhost:8000/docs.

## Running it locally

### Prerequisites

- Python 3.10
- PostgreSQL running locally (the backend connects to it when it starts)

### 1. Clone the repo

```bash
git clone https://github.com/adavidson1995/ML_institute_mnist_foundation.git
cd ML_institute_mnist_foundation
```

### 2. Create the database

```bash
createdb mnist_predictions
export DATABASE_URL=postgresql://<your-postgres-user>@localhost:5432/mnist_predictions
```

The `predictions` table is created automatically the first time the backend starts.

### 3. Start the backend

```bash
cd model
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt sqlalchemy psycopg2-binary pytz

# Use the trained weights from the repo root...
cp ../mnist_cnn.pth .
# ...or train your own (downloads MNIST and writes mnist_cnn.pth here):
# python train.py

uvicorn api:app --port 8000
```

### 4. Start the frontend

In a second terminal:

```bash
cd app
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, draw a digit and click **Predict**. The sidebar holds the backend URL (default `http://localhost:8000/predict`).

### Calling the API directly

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": $(python3 -c 'print([[0.0]*28]*28)')}"
```

## Project structure

```
.
├── app/                  # Streamlit frontend
│   ├── app.py            # Drawing canvas, prediction display, history table
│   └── requirements.txt
├── model/                # Model and FastAPI backend
│   ├── model.py          # MNISTNet CNN definition
│   ├── train.py          # Training and test-set evaluation
│   ├── inference.py      # Helpers for loading the model and predicting
│   ├── api.py            # FastAPI app: /predict and /history
│   ├── database.py       # SQLAlchemy model and PostgreSQL connection
│   └── requirements.txt
├── db/init.sql           # Placeholder for database setup
├── docker-compose.yml    # Placeholder for containerised deployment
└── mnist_cnn.pth         # Trained model weights
```

## Tech stack

- **Model:** PyTorch, torchvision
- **Backend:** FastAPI, Uvicorn, SQLAlchemy, PostgreSQL
- **Frontend:** Streamlit, streamlit-drawable-canvas

## Not done yet

- **Docker.** `docker-compose.yml`, the Dockerfiles and `db/init.sql` are empty placeholders; the app has to be run locally as described above.
- **Saving history to the database.** The history table in the frontend only lasts for your browser session. The backend can save predictions to PostgreSQL, but the frontend doesn't send `true_label` yet, so nothing is written to the database.

## License

[MIT](LICENSE)
