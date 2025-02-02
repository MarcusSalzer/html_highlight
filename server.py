from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Store training progress
training_data: dict[str, list[float]] = {"epochs": [], "loss": [], "accuracy": []}


class UpdateData(BaseModel):
    epoch: int
    loss: float
    accuracy: float


@app.post("/update")
async def update_progress(data: UpdateData):
    training_data["epochs"].append(data.epoch)
    training_data["loss"].append(data.loss)
    training_data["accuracy"].append(data.accuracy)
    return {"message": "Updated"}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <canvas id="lossChart"></canvas>
        <canvas id="accuracyChart"></canvas>
        <script>
            async function fetchData() {
                const response = await fetch('/data');
                const data = await response.json();
                
                lossChart.data.labels = data.epochs;
                lossChart.data.datasets[0].data = data.loss;
                lossChart.update();

                accuracyChart.data.labels = data.epochs;
                accuracyChart.data.datasets[0].data = data.accuracy;
                accuracyChart.update();
            }

            setInterval(fetchData, 1000);

            const ctx1 = document.getElementById('lossChart').getContext('2d');
            const lossChart = new Chart(ctx1, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'Loss', borderColor: 'red', data: [] }] }
            });

            const ctx2 = document.getElementById('accuracyChart').getContext('2d');
            const accuracyChart = new Chart(ctx2, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'Accuracy', borderColor: 'green', data: [] }] }
            });

            fetchData();
        </script>
    </body>
    </html>
    """


@app.get("/data")
async def get_data():
    return training_data


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
