#!../ros2env/bin/python
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import sys
print("PYTHON:", sys.executable)
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import torch
import numpy as np
import joblib
import datetime

from dialogue_manager import DialogueManager
from CNN_UQ import Enhanced1DCNN, mc_dropout_predict, enable_dropout

USER_ID = 0  


class StressInferenceNode(Node):
    def __init__(self):
        super().__init__('stress_inference_node')

        # load saved scaler and model
        self.scaler = joblib.load('src/stress_inference_pkg/stress_inference_pkg/cnn_scaler.save')
        input_dim = 16
        self.model = Enhanced1DCNN(input_features=input_dim)
        self.model.load_state_dict(torch.load(
            'src/stress_inference_pkg/stress_inference_pkg/cnn_uq_model_60.pt',
            map_location=torch.device('cpu')))
        self.model.eval()
        enable_dropout(self.model)

        self.dialogue_manager = DialogueManager(entropy_threshold=0.45, use_time=True)

        # subscriptions
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'physio_features',
            self.listener_callback,
            10
        )

        self.feedback_sub = self.create_subscription(
            String,
            'user_feedback',
            self.feedback_callback,
            10
        )

        # publisher
        self.publisher_ = self.create_publisher(String, 'stress_dialogue_response', 10)

        self.sample_idx = 0
        self.latest_feedback = None
        self.get_logger().info('Stress inference + dialogue manager node started.')

    def feedback_callback(self, msg):
        self.latest_feedback = msg.data
        self.get_logger().info(f"Received user feedback: {msg.data}")

    def listener_callback(self, msg):
        features = np.array(msg.data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1)

        # MC dropout prediction
        mean_probs, entropies = mc_dropout_predict(self.model, input_tensor, n_samples=50)
        pred_label = mean_probs.argmax(axis=1).item()
        entropy_val = entropies[0]
        ros_time = self.get_clock().now().to_msg()
        timestamp = datetime.datetime.fromtimestamp(ros_time.sec + ros_time.nanosec * 1e-9)

        user_feedback = self.latest_feedback
        self.latest_feedback = None  # reset 

        response = self.dialogue_manager.get_response(
            user_id=USER_ID,
            entropy_val=entropy_val,
            pred_label=pred_label,
            timestamp=timestamp,
            user_feedback=user_feedback,
            sample_idx=self.sample_idx
        )

        # response
        msg_out = String()
        msg_out.data = f"Predicted: {pred_label}, Entropy: {entropy_val:.4f}, Agent says: {response}"
        self.get_logger().info(msg_out.data)
        self.publisher_.publish(msg_out)

        self.sample_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = StressInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
