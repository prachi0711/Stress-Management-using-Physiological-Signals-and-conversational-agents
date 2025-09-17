#!..ros2env/bin/python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import pandas as pd


class CSVPublisher(Node):
    def __init__(self, eda_csv_path, ibi_csv_path, publish_rate=1.0):
        super().__init__('csv_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'physio_features', 10)

        # load both CSV files
        eda_df = pd.read_csv(eda_csv_path)
        ibi_df = pd.read_csv(ibi_csv_path)

        if len(eda_df) != len(ibi_df):
            raise ValueError(f"CSV length mismatch: EDA={len(eda_df)}, IBI={len(ibi_df)}")

        self.data = pd.concat([eda_df, ibi_df], axis=1).values
        self.idx = 0

        self.timer = self.create_timer(publish_rate, self.timer_callback)

        self.get_logger().info(
            f"CSVPublisher started. Loaded {len(self.data)} samples "
            f"from {eda_csv_path} and {ibi_csv_path}"
        )

    def timer_callback(self):
        if self.idx < len(self.data):
            msg = Float32MultiArray()
            msg.data = self.data[self.idx].tolist()
            self.publisher_.publish(msg)

            self.get_logger().info(f"Published row {self.idx}: {msg.data}")
            self.idx += 1
        else:
            self.get_logger().info("Finished publishing all rows from CSVs.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    eda_path = "../src/stress_inference_pkg/stress_inference_pkg/synthetic_eda_features_60.csv"  
    ibi_path = "../src/stress_inference_pkg/stress_inference_pkg/synthetic_ibi_features_60.csv"  

    node = CSVPublisher(eda_csv_path=eda_path, ibi_csv_path=ibi_path, publish_rate=1.0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
