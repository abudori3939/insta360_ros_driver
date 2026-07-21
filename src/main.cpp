#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <deque>
#include <mutex>

#include <camera/camera.h>
#include <camera/photography_settings.h>
#include <camera/device_discovery.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/imu.hpp"

// Maps the camera's own clock onto ROS time.
//
// The X5 stamps video packets and gyro samples with the same clock: a millisecond
// counter running since the camera booted. SyncLocalTimeToCamera() does *not* make
// those stamps UTC (it only takes whole seconds anyway), so we have to align the two
// clocks ourselves.
//
// Arrival jitter is around 30 ms, so we track the *minimum* observed
// (ros_receive_time - camera_stamp) over a sliding window rather than the latest or
// the average. The minimum is the tightest bound on the true offset, it is immune to
// jitter, and it still follows slow clock drift. Video and gyro share one estimator
// because they share one clock - which is what lets a future LiDAR fusion line the
// two streams up against each other.
class CameraClock {
public:
    explicit CameraClock(int64_t window_ns) : window_ns_(window_ns) {}

    // Folds one observation into the estimate and returns the ROS time for cam_ms.
    // The result can never be in the future relative to received_ns, since the offset
    // used is a minimum over observations that include this one.
    int64_t toRosNanos(int64_t cam_ms, int64_t received_ns) {
        const int64_t cam_ns = cam_ms * 1000000LL;
        const int64_t offset = received_ns - cam_ns;

        std::lock_guard<std::mutex> lock(mutex_);
        // Retire observations that have aged out of the window.
        while (!samples_.empty() && received_ns - samples_.front().received_ns > window_ns_) {
            samples_.pop_front();
        }
        // Keep offsets increasing along the deque, so the front is always the minimum.
        while (!samples_.empty() && samples_.back().offset >= offset) {
            samples_.pop_back();
        }
        samples_.push_back({received_ns, offset});
        return cam_ns + samples_.front().offset;
    }

private:
    struct Sample {
        int64_t received_ns;
        int64_t offset;
    };

    int64_t window_ns_;
    std::deque<Sample> samples_;
    std::mutex mutex_;
};

class TestStreamDelegate : public ins_camera::StreamDelegate {
private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    bool use_camera_timestamp_;
    CameraClock camera_clock_;

public:
    TestStreamDelegate(const std::shared_ptr<rclcpp::Node>& node)
        : node_(node),
          use_camera_timestamp_(node->declare_parameter("use_camera_timestamp", true)),
          camera_clock_(static_cast<int64_t>(
              node->declare_parameter("camera_clock_window_sec", 10.0) * 1e9)) {
        // Publisher for the compressed H.264 video stream
        compressed_pub_ = node_->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/dual_fisheye/image/compressed",
            rclcpp::QoS(10)
        );

        // Publisher for IMU data (remains the same)
        imu_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", rclcpp::SensorDataQoS());
        RCLCPP_INFO(node_->get_logger(), "Publisher for compressed images and IMU created.");
        RCLCPP_INFO(node_->get_logger(), "Camera timestamps: %s",
                    use_camera_timestamp_ ? "enabled" : "disabled (using arrival time)");
    }

private:
    // Timestamp for a camera-clock reading, in the camera's milliseconds. A stamp of 0
    // means the SDK had none for this packet (it does that for the very first one), so
    // we fall back to arrival time.
    rclcpp::Time stampFor(int64_t cam_ms) {
        return stampFor(cam_ms, node_->get_clock()->now());
    }

    rclcpp::Time stampFor(int64_t cam_ms, const rclcpp::Time& received) {
        if (!use_camera_timestamp_ || cam_ms <= 0) {
            return received;
        }
        return rclcpp::Time(camera_clock_.toRosNanos(cam_ms, received.nanoseconds()),
                            received.get_clock_type());
    }

public:
    virtual ~TestStreamDelegate() {}

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index) override {
        // We only care about the main video stream (index 0)
        if (stream_index == 0 && size > 0 && compressed_pub_) {
            auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();

            // Set the header. The stamp is when the camera captured the frame, not when
            // we received it - the decoder and everything downstream carry it through.
            const rclcpp::Time stamp = stampFor(timestamp);
            msg->header.stamp = stamp;
            msg->header.frame_id = "camera_frame";

            RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                "camera->driver lag: %.1f ms (camera clock at %ld ms)",
                (node_->get_clock()->now() - stamp).seconds() * 1e3, (long)timestamp);

            // Set the format to H.264
            // The subscriber will need to know this to select the correct decoder.
            msg->format = "h264";

            // Copy the compressed video data directly into the message
            msg->data.assign(data, data + size);

            compressed_pub_->publish(std::move(msg));
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        // The SDK hands us a batch spanning ~100 ms of samples. They all arrive at once
        // but they were not all measured at once, so each one keeps its own camera
        // stamp - stamping the whole batch with the arrival time smears the IMU rate and
        // wrecks the orientation filter's integration.
        const rclcpp::Time received = node_->get_clock()->now();
        for (const auto& gyro : data) {
            auto msg = std::make_unique<sensor_msgs::msg::Imu>();
            msg->header.stamp = stampFor(gyro.timestamp, received);
            msg->header.frame_id = "imu_frame";
            msg->angular_velocity.x = gyro.gx;
            msg->angular_velocity.y = gyro.gy;
            msg->angular_velocity.z = gyro.gz;
            
            msg->linear_acceleration.x = gyro.ax * 9.80665;
            msg->linear_acceleration.y = gyro.ay * 9.80665;
            msg->linear_acceleration.z = gyro.az * 9.80665;

            msg->orientation.x = 0.0;
            msg->orientation.y = 0.0;
            msg->orientation.z = 0.0;
            msg->orientation.w = 1.0; // Neutral orientation
            msg->orientation_covariance[0] = -1.0; // No orientation data available

            for (int i = 0; i < 9; i++)
            {
                msg->angular_velocity_covariance[i] = 0;
                msg->linear_acceleration_covariance[i] = 0;
            }
            imu_pub_->publish(std::move(msg));
        }
    }

    void OnExposureData(const ins_camera::ExposureData& data) override {}
};

class CameraWrapper {
private:
    std::shared_ptr<ins_camera::Camera> cam;
    std::shared_ptr<rclcpp::Node> node_;

public:
    CameraWrapper(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {}

    ~CameraWrapper() {
        if (cam) {
            cam->Close();
        }
    }

    int run_camera() {
        ins_camera::DeviceDiscovery discovery;
        auto list = discovery.GetAvailableDevices();
        if (list.empty()) {
            RCLCPP_ERROR(node_->get_logger(), "No available camera devices found.");
            return -1;
        }

        cam = std::make_shared<ins_camera::Camera>(list[0].info);
        if (!cam->Open()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open camera.");
            return -1;
        }
        RCLCPP_INFO(node_->get_logger(), "Camera opened successfully.");
        discovery.FreeDeviceDescriptors(list);

        // Ensure both lenses are active. On the X5/X4 the camera remembers its last
        // active-sensor selection across USB sessions; if it was left on a single lens
        // (e.g. rear only) the live stream is a single-lens 16:9 image instead of the
        // 2:1 dual-fisheye. Forcing SENSOR_DEVICE_ALL guarantees the dual-fisheye stream.
        if (!cam->SetActiveSensor(ins_camera::SENSOR_DEVICE_ALL)) {
            RCLCPP_WARN(node_->get_logger(), "SetActiveSensor(ALL) failed; the stream may be single-lens.");
        }

        std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<TestStreamDelegate>(node_);
        cam->SetStreamDelegate(delegate);

        auto start = time(NULL);

        uint64_t utc_time = static_cast<uint64_t>(start);
        uint32_t offset_time = 0; //no offset from UTC

        cam->SyncLocalTimeToCamera(utc_time,offset_time);       
        ins_camera::LiveStreamParam param;
        param.video_resolution = ins_camera::VideoResolution::RES_1920_960P30; //Change this line to edit the resolution
        //Possible resolutions (results may vary per model) are:
        //RES_3840_1920P30
        //RES_2560_1280P30
        //RES_1152_1152P30 (this will give 2304 x 1152 at 30 FPS)
        //RES_1920_960P30  
        param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
        param.video_bitrate = 1024 * 1024 / 2;
        param.enable_audio = false;
        param.using_lrv = false;

        if (!cam->StartLiveStreaming(param)) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to start live streaming.");
            return -1;
        }
        
        RCLCPP_INFO(node_->get_logger(), "Live streaming started.");
        return 0;
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("insta_publisher");
    
    CameraWrapper camera(node);
    if (camera.run_camera() != 0) {
        rclcpp::shutdown();
        return -1;
    }
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}