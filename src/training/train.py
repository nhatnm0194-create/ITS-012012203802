
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model():
    # Đọc cấu hình
    config = load_config("configs/train_config.yaml")

    # Lấy thông số
    training = config["training"]
    paths = config["paths"]

    # Tạo thư mục lưu kết quả
    Path(paths["checkpoint_dir"]).mkdir(
        parents=True, exist_ok=True
    )

    Path(paths["log_dir"]).mkdir(
        parents=True, exist_ok=True
    )

    # Khởi tạo mô hình
    model = YOLO(config["model_type"])

    # Huấn luyện
    results = model.train(
        data=config["dataset_config"],
        epochs=training["epochs"],
        batch=training["batch_size"],
        imgsz=training["image_size"],
        lr0=training["learning_rate"],
        optimizer=training["optimizer"],
        device=training["device"],
        project=paths["output_dir"],
        name="traffic_sign",
        exist_ok=True
    )

    print("Training completed!")
    print("Results:", results)


if __name__ == "__main__":
    train_model()