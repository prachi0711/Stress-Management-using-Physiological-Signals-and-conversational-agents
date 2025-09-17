# Stress Inference ROS2 Package

This package provides a ROS2-based stress inference pipeline with integrated uncertainty quantification (UQ) and dialogue management.

It includes:

- A ROS2 node (`stress_inference_node.py`) that performs inference using a trained CNN model with MC Dropout UQ and interacts with the dialogue manager.Similarly (`stress_inference_rf.py`) performs inference
  using a RF model with entropy UQ.

- A CSV publisher node (`csv_publisher.py`) for publishing preprocessed physiological features (EDA + IBI) as ROS messages. For the testing purpose; synthetic data is used which is generated in this way -
  [synethetic data]()

## 1. Create ROS2 Package

Inside a new workspace (`ros2_ws`):

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python stress_inference_pkg
```

This creates the package structure. Place your files inside:

```
stress_inference_pkg/
│── package.xml
│── setup.py
│── resource/stress_inference_pkg
│── stress_inference_pkg/
│   ├── __init__.py
│   ├── stress_inference_node.py
│   ├── csv_publisher.py
│   ├── dialogue_manager.py
│   ├── CNN_UQ.py
│   ├── cnn_scaler.save
│   ├── cnn_uq_model_60.pt
│   ├── stress_inference_rf.py
│   ├── rf_scaler.save
│   ├── rf_uq_model_60.pt
│   ├── synthetic_eda_features_60.csv
│   ├── synthetic_ibi_features_60.csv
```

## 2. Virtual Environment & Requirements

Create and activate a ROS2-compatible Python virtual environment:

```bash
cd ~/ros2_ws
python3 -m venv ros2env
source ros2env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Update `setup.py`

In **`setup.py`**:

```python
    entry_points={
        'console_scripts': [
            'stress_inference_node = stress_inference_pkg.stress_inference_node:main',
            'csv_publisher = stress_inference_pkg.csv_publisher:main',
            'stress_inference_rf = stress_inference_pkg.stress_inference_rf:main',
        ],
    },
```


## 4. Build & Source

From the workspace root:

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## 5. Run the Nodes

### 1. Run the Stress Inference Node

```bash
ros2 run stress_inference_pkg stress_inference_node
```

* Subscribes to:

  * `/physio_features` → feature vectors from CSVPublisher
  * `/user_feedback` → simulated user responses (`yes`, `no`, `stressed`, etc.)
* Publishes responses to:

  ```
  /stress_dialogue_response
  ```
### 2. Run the CSV Publisher 
```bash
ros2 run stress_inference_pkg csv_publisher
```

* Reads EDA + IBI CSV.
* Publishes features on topic:

  ```
  /physio_features
  ```

## 6. Example Terminal Outputs

### From `csv_publisher`:

```
[INFO] Published row 0: [0.123, 0.456, ..., 0.789]
...
```

### From `stress_inference_node`:

```
[INFO] Stress inference + dialogue manager node started.
[INFO] Predicted: 1, Entropy: 0.3721, Agent says: You seem stressed. Would you like a short breathing exercise?
```


## 7. Send User Feedback

 Publish feedback messages manually:

```bash
ros2 topic pub /user_feedback std_msgs/String "data: 'yes'"
```

Example responses:

* `yes` / `no` → accepts/rejects breathing exercise.
* `stressed` / `calm` → explicit state feedback.
* `need_help` / `no_help` → request or reject assistance.

