#ifndef EQUIRECTANGULAR_HPP
#define EQUIRECTANGULAR_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <condition_variable>

class EquirectangularNode : public rclcpp::Node
{
public:
    explicit EquirectangularNode();
    ~EquirectangularNode();

private:
    // Callback functions
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters);

    // Worker loop; see the comment on latest_msg_ for why projection runs off the
    // subscription thread.
    void workerLoop();
    void processFrame(const sensor_msgs::msg::Image::SharedPtr& msg);

    // Initialization functions
    void loadParameters();
    void updateCameraParameters();
    void initMapping(int dual_height, int dual_width);

    // ROS2 communication
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr dual_fisheye_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr equirect_pub_;

    // Parameters
    double cx_offset_;
    double cy_offset_;
    int crop_size_;
    double tx_, ty_, tz_;
    double roll_, pitch_, yaw_;
    bool gpu_enabled_;
    int out_width_;
    int out_height_;
    double max_rate_;
    int interpolation_;

    // Camera parameters
    cv::Mat back_to_front_rotation_;
    cv::Vec3d back_to_front_translation_;

    // Combined remap tables in raw dual-fisheye coordinates. The per-frame split,
    // 90-degree rotations, center crop and front/back hemisphere selection are all
    // folded into these at init time, so projection is a single cv::remap of the
    // incoming image. Stored in the fixed-point form convertMaps() produces, which
    // remap executes noticeably faster than CV_32F maps.
    cv::Mat map1_;  // CV_16SC2 integer coordinates
    cv::Mat map2_;  // CV_16UC1 interpolation-table indices

    // State management
    std::atomic<bool> maps_initialized_;
    std::atomic<bool> params_changed_;
    int dual_height_;
    int dual_width_;

    // Scratch buffer reused across frames when the input needs a BGR->RGB swap
    // after remapping (avoids a per-frame allocation).
    cv::Mat remap_buf_;

    // Projection is far too slow to run inline in the subscription callback at the
    // camera's frame rate, and it does not need to: downstream only wants max_rate_ Hz.
    // The callback therefore just parks the newest frame here and returns, and the
    // worker picks up whatever is parked when it is ready to publish. Older frames are
    // overwritten rather than queued, so what gets projected is always the freshest
    // image available - throttling by queueing would publish stale frames instead.
    std::thread worker_thread_;
    sensor_msgs::msg::Image::SharedPtr latest_msg_;
    std::mutex slot_mutex_;
    std::condition_variable slot_cv_;
    std::atomic<bool> stop_worker_{false};
    std::chrono::steady_clock::time_point last_publish_{};
};

#endif // EQUIRECTANGULAR_HPP
