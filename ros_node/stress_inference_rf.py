#!/home/prachi/new_ros2_ws/ros2env/bin/python
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import sys
print("PYTHON:", sys.executable)
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import joblib
import datetime
from scipy.stats import entropy

from dialogue_manager import DialogueManager

USER_ID = 0


class StressInferenceRFNode(Node):
    def __init__(self):
        super().__init__('stress_inference_rf_node')

        # Load saved scaler and Random Forest model
        self.scaler = joblib.load('src/stress_inference_pkg/stress_inference_pkg/rf_scaler.save')
        self.rf_model = joblib.load('src/stress_inference_pkg/stress_inference_pkg/rf_uq_model_60.pkl')

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
        self.get_logger().info('Stress inference RF + dialogue manager node started.')

    def feedback_callback(self, msg):
        self.latest_feedback = msg.data
        self.get_logger().info(f"Received user feedback: {msg.data}")

    def listener_callback(self, msg):
        features = np.array(msg.data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # Predict probabilities with RF
        probs = self.rf_model.predict_proba(features_scaled)[0]  
        pred_label = np.argmax(probs)
        entropy_val = entropy(probs)

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

        if response is None:
            response = "No response triggered."

        # response
        msg_out = String()
        msg_out.data = f"Predicted: {pred_label}, Entropy: {entropy_val:.4f}, Agent says: {response}"
        self.get_logger().info(msg_out.data)
        self.publisher_.publish(msg_out)

        self.sample_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = StressInferenceRFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
