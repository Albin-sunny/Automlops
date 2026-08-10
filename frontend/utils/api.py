import requests

BASE_URL = "http://127.0.0.1:8000"


def upload_dataset(file):
    files = {"file": file}
    return requests.post(f"{BASE_URL}/datasets/upload", files=files)


def get_datasets():
    return requests.get(f"{BASE_URL}/datasets/")


def get_dataset(dataset_id):
    return requests.get(f"{BASE_URL}/datasets/{dataset_id}")


def delete_dataset(dataset_id):
    return requests.delete(f"{BASE_URL}/datasets/{dataset_id}")


def select_target(dataset_id, target_column):
    return requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/select-target",
        params={"target_column": target_column}
    )


def profile_dataset(dataset_id):
    return requests.post(f"{BASE_URL}/profiling/{dataset_id}")


def preprocess_dataset(dataset_id):
    return requests.post(f"{BASE_URL}/preprocessing/{dataset_id}")


def train_model(dataset_id):
    return requests.post(f"{BASE_URL}/training/{dataset_id}")


def predict(dataset_id, file):
    files = {"file": file}
    return requests.post(
        f"{BASE_URL}/prediction/{dataset_id}",
        files=files
    )




def get_latest_monitoring():
    response = requests.get(
        f"{BASE_URL}/monitoring/latest",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_monitoring_history(limit=20):
    response = requests.get(
        f"{BASE_URL}/monitoring/history",
        params={"limit": limit},
        timeout=30
    )

    response.raise_for_status()

    return response.json()