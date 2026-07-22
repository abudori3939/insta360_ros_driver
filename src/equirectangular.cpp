#include "equirectangular.hpp"
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cmath>
#include <thread>
#include <chrono>

EquirectangularNode::EquirectangularNode()
    : Node("equirectangular_node"),
      maps_initialized_(false),
      params_changed_(true),
      dual_height_(0),
      dual_width_(0)
{
    // Declare parameters
    declare_parameter("cx_offset", 0.0);
    declare_parameter("cy_offset", 0.0);
    declare_parameter("crop_size", 960);
    declare_parameter("translation", std::vector<double>{0.0, 0.0, -0.105});
    declare_parameter("rotation_deg", std::vector<double>{-0.5, 0.0, 1.1});
    declare_parameter("gpu", true);
    declare_parameter("out_width", 1920);
    declare_parameter("out_height", 960);
    // Publish rate ceiling. 0 disables throttling and projects every frame that arrives.
    declare_parameter("max_rate", 10.0);
    // nearest | linear | cubic. Cubic costs several times linear and the difference is
    // not visible on a fisheye unwarp, so linear is the default.
    declare_parameter("interpolation", "linear");

    // Load parameters
    loadParameters();

    // Log GPU settings (note: C++ version currently only supports CPU)
    RCLCPP_INFO(get_logger(), "C++ equirectangular node");


    // Add parameter callback
    auto params_callback_handle = add_on_set_parameters_callback(
        std::bind(&EquirectangularNode::parametersCallback, this, std::placeholders::_1));

    updateCameraParameters();

    // Configure QoS
    auto qos = rclcpp::QoS(1).reliable();

    // Create publishers and subscribers
    dual_fisheye_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/dual_fisheye/image", qos,
        std::bind(&EquirectangularNode::imageCallback, this, std::placeholders::_1));

    equirect_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/equirectangular/image", qos);

    worker_thread_ = std::thread(&EquirectangularNode::workerLoop, this);
}

EquirectangularNode::~EquirectangularNode()
{
    stop_worker_ = true;
    slot_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void EquirectangularNode::loadParameters()
{
    try {
        cx_offset_ = get_parameter("cx_offset").as_double();
        cy_offset_ = get_parameter("cy_offset").as_double();
        crop_size_ = get_parameter("crop_size").as_int();
        out_width_ = get_parameter("out_width").as_int();
        out_height_ = get_parameter("out_height").as_int();
        gpu_enabled_ = get_parameter("gpu").as_bool();
        max_rate_ = get_parameter("max_rate").as_double();

        const std::string interpolation = get_parameter("interpolation").as_string();
        if (interpolation == "nearest") {
            interpolation_ = cv::INTER_NEAREST;
        } else if (interpolation == "cubic") {
            interpolation_ = cv::INTER_CUBIC;
        } else {
            if (interpolation != "linear") {
                RCLCPP_WARN(get_logger(), "Unknown interpolation '%s', using linear",
                            interpolation.c_str());
            }
            interpolation_ = cv::INTER_LINEAR;
        }

        auto translation = get_parameter("translation").as_double_array();
        tx_ = translation[0];
        ty_ = translation[1];
        tz_ = translation[2];

        auto rotation_deg = get_parameter("rotation_deg").as_double_array();
        roll_ = rotation_deg[0] * M_PI / 180.0;
        pitch_ = rotation_deg[1] * M_PI / 180.0;
        yaw_ = rotation_deg[2] * M_PI / 180.0;

        RCLCPP_INFO(get_logger(), "Loaded parameters from ROS parameter server");
        RCLCPP_INFO(get_logger(), "  Crop size: %d", crop_size_);
        RCLCPP_INFO(get_logger(), "  Center offset: (%.1f, %.1f)", cx_offset_, cy_offset_);
        RCLCPP_INFO(get_logger(), "  Translation: [%.3f, %.3f, %.3f]", tx_, ty_, tz_);
        RCLCPP_INFO(get_logger(), "  Rotation (deg): [%.1f, %.1f, %.1f]",
                    rotation_deg[0], rotation_deg[1], rotation_deg[2]);
        RCLCPP_INFO(get_logger(), "  Output size: %dx%d", out_width_, out_height_);
        RCLCPP_INFO(get_logger(), "  GPU enabled: %s", gpu_enabled_ ? "true" : "false");
        RCLCPP_INFO(get_logger(), "  Interpolation: %s", interpolation.c_str());
        if (max_rate_ > 0.0) {
            RCLCPP_INFO(get_logger(), "  Max rate: %.1f Hz", max_rate_);
        } else {
            RCLCPP_INFO(get_logger(), "  Max rate: unthrottled");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error loading parameters: %s", e.what());
        gpu_enabled_ = true;
        throw;
    }
}

void EquirectangularNode::updateCameraParameters()
{
    // Build rotation matrix
    cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
        1.0, 0.0, 0.0,
        0.0, cos(roll_), -sin(roll_),
        0.0, sin(roll_), cos(roll_));

    cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
        cos(pitch_), 0.0, sin(pitch_),
        0.0, 1.0, 0.0,
        -sin(pitch_), 0.0, cos(pitch_));

    cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
        cos(yaw_), -sin(yaw_), 0.0,
        sin(yaw_), cos(yaw_), 0.0,
        0.0, 0.0, 1.0);

    back_to_front_rotation_ = Rz * Ry * Rx;
    back_to_front_translation_ = cv::Vec3d(tx_, ty_, tz_);

    if (maps_initialized_) {
        maps_initialized_ = false;
        RCLCPP_INFO(get_logger(), "Parameters updated, remapping will occur on next image");
    }
}

void EquirectangularNode::initMapping(int dual_height, int dual_width)
{
    dual_height_ = dual_height;
    dual_width_ = dual_width;

    // Geometry of the per-frame preprocessing this map replaces:
    //   front = right half of the dual image, rotated 90 deg counterclockwise
    //   back  = left half,                    rotated 90 deg clockwise
    //   both then center-cropped to crop_size x crop_size
    // Each half is dual_height rows x half_w cols, so the rotated halves are
    // half_w rows x dual_height cols.
    const int half_w = dual_width / 2;
    const int rot_h = half_w;
    const int rot_w = dual_height;

    int crop_w = crop_size_;
    int crop_h = crop_size_;
    int x_start = (rot_w - crop_w) / 2;
    int y_start = (rot_h - crop_h) / 2;
    if (x_start < 0 || y_start < 0) {
        crop_w = rot_w;
        crop_h = rot_h;
        x_start = 0;
        y_start = 0;
    }

    RCLCPP_INFO(get_logger(),
                "Initializing equirectangular projection: %dx%d dual fisheye "
                "(crop %dx%d per lens) to %dx%d",
                dual_width, dual_height, crop_w, crop_h, out_width_, out_height_);

    const double cx = crop_w / 2.0 + cx_offset_;
    const double cy = crop_h / 2.0 + cy_offset_;
    const double fisheye_radius = crop_w / 2.0;

    const cv::Mat& R = back_to_front_rotation_;
    const double r00 = R.at<double>(0, 0), r01 = R.at<double>(0, 1), r02 = R.at<double>(0, 2);
    const double r10 = R.at<double>(1, 0), r11 = R.at<double>(1, 1), r12 = R.at<double>(1, 2);
    const double r20 = R.at<double>(2, 0), r21 = R.at<double>(2, 1), r22 = R.at<double>(2, 2);

    cv::Mat map_x(out_height_, out_width_, CV_32F);
    cv::Mat map_y(out_height_, out_width_, CV_32F);

    for (int y = 0; y < out_height_; ++y) {
        const double lat = static_cast<double>(y) / out_height_ * M_PI - M_PI / 2.0;
        const double cos_lat = std::cos(lat);
        const double sin_lat = std::sin(lat);
        float* mx = map_x.ptr<float>(y);
        float* my = map_y.ptr<float>(y);

        for (int x = 0; x < out_width_; ++x) {
            const double lon = static_cast<double>(x) / out_width_ * 2.0 * M_PI - M_PI;
            double X = cos_lat * std::sin(lon);
            double Y = sin_lat;
            double Z = cos_lat * std::cos(lon);

            const bool front = Z >= 0.0;
            if (!front) {
                // Bring the ray into the back camera's frame, mirrored to match
                // the back fisheye's orientation.
                const double px = r00 * X + r01 * Y + r02 * Z + tx_;
                const double py = r10 * X + r11 * Y + r12 * Z + ty_;
                const double pz = r20 * X + r21 * Y + r22 * Z + tz_;
                X = -px;
                Y = py;
                Z = pz;
            }

            double r = std::sqrt(X * X + Y * Y);
            if (r < 1e-6) r = 1e-6;
            const double theta = std::atan2(r, std::fabs(Z));
            const double r_fisheye = 2.0 * theta / M_PI * fisheye_radius;

            // Fisheye coordinates in the rotated-and-cropped lens image
            const double u = cx + X / r * r_fisheye;
            const double v = cy + Y / r * r_fisheye;

            // Undo crop and 90-degree rotation to land in raw dual-image pixels
            if (front) {
                mx[x] = static_cast<float>(2 * half_w - 1 - (v + y_start));
                my[x] = static_cast<float>(u + x_start);
            } else {
                mx[x] = static_cast<float>(v + y_start);
                my[x] = static_cast<float>(dual_height - 1 - (u + x_start));
            }
        }
    }

    cv::convertMaps(map_x, map_y, map1_, map2_, CV_16SC2,
                    interpolation_ == cv::INTER_NEAREST);
    remap_buf_.create(out_height_, out_width_, CV_8UC3);

    maps_initialized_ = true;

    RCLCPP_INFO(get_logger(), "Mapping matrices initialization complete");
}


void EquirectangularNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr dual_fisheye_msg)
{
    {
        std::lock_guard<std::mutex> lock(slot_mutex_);
        latest_msg_ = dual_fisheye_msg;  // replaces any frame not yet picked up
    }
    slot_cv_.notify_one();
}

void EquirectangularNode::workerLoop()
{
    while (!stop_worker_) {
        sensor_msgs::msg::Image::SharedPtr frame;
        {
            std::unique_lock<std::mutex> lock(slot_mutex_);
            slot_cv_.wait(lock, [this] { return latest_msg_ != nullptr || stop_worker_; });
            if (stop_worker_) {
                break;
            }

            // Wait out the rate limit before claiming a frame, not after. Sleeping with
            // a frame in hand would publish an image that went stale while we waited;
            // sleeping first means we take whatever is newest at the moment we wake.
            if (max_rate_ > 0.0) {
                const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(1.0 / max_rate_));
                const auto ready_at = last_publish_ + interval;
                if (std::chrono::steady_clock::now() < ready_at) {
                    slot_cv_.wait_until(lock, ready_at);
                    continue;
                }
            }

            frame = latest_msg_;
            latest_msg_.reset();
            last_publish_ = std::chrono::steady_clock::now();
        }

        processFrame(frame);
    }
}

void EquirectangularNode::processFrame(const sensor_msgs::msg::Image::SharedPtr& dual_fisheye_msg)
{
    try {
        // The driver publishes bgr8. toCvShare wraps the message buffer without
        // copying; the channel order is corrected after remapping, on the smaller
        // output image, instead of converting the full input up front.
        const bool bgr_input =
            dual_fisheye_msg->encoding == sensor_msgs::image_encodings::BGR8;
        cv_bridge::CvImageConstPtr cv_ptr;
        if (bgr_input || dual_fisheye_msg->encoding == sensor_msgs::image_encodings::RGB8) {
            cv_ptr = cv_bridge::toCvShare(dual_fisheye_msg);
        } else {
            cv_ptr = cv_bridge::toCvCopy(dual_fisheye_msg, "rgb8");
        }
        const cv::Mat& dual_fisheye_img = cv_ptr->image;

        if (!maps_initialized_ || params_changed_ ||
            dual_fisheye_img.rows != dual_height_ || dual_fisheye_img.cols != dual_width_) {
            initMapping(dual_fisheye_img.rows, dual_fisheye_img.cols);
            params_changed_ = false;
        }

        auto start_time = now();

        // Remap straight into the outgoing message buffer so publishing does not
        // copy the image again. The header is passed through untouched so the
        // camera's capture time reaches subscribers.
        auto img_msg = std::make_unique<sensor_msgs::msg::Image>();
        img_msg->header = dual_fisheye_msg->header;
        img_msg->height = out_height_;
        img_msg->width = out_width_;
        img_msg->encoding = "rgb8";
        img_msg->is_bigendian = 0;
        img_msg->step = out_width_ * 3;
        img_msg->data.resize(static_cast<size_t>(img_msg->step) * out_height_);
        cv::Mat equirect_img(out_height_, out_width_, CV_8UC3,
                             img_msg->data.data(), img_msg->step);

        if (bgr_input) {
            cv::remap(dual_fisheye_img, remap_buf_, map1_, map2_,
                      interpolation_, cv::BORDER_CONSTANT);
            cv::cvtColor(remap_buf_, equirect_img, cv::COLOR_BGR2RGB);
        } else {
            cv::remap(dual_fisheye_img, equirect_img, map1_, map2_,
                      interpolation_, cv::BORDER_CONSTANT);
        }

        equirect_pub_->publish(std::move(img_msg));

        auto process_time = (now() - start_time).seconds();
        RCLCPP_DEBUG(get_logger(), "Processing time: %.3f seconds", process_time);
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
            "projection %.1f ms | camera->equirectangular lag %.1f ms",
            process_time * 1e3,
            (now() - rclcpp::Time(dual_fisheye_msg->header.stamp)).seconds() * 1e3);

    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error processing images: %s", e.what());
    }
}

rcl_interfaces::msg::SetParametersResult EquirectangularNode::parametersCallback(
    const std::vector<rclcpp::Parameter> &parameters)
{
    bool update_needed = false;

    for (const auto& param : parameters) {
        if (param.get_name() == "cx_offset" ||
            param.get_name() == "cy_offset" ||
            param.get_name() == "crop_size" ||
            param.get_name() == "translation" ||
            param.get_name() == "rotation_deg" ||
            param.get_name() == "out_width" ||
            param.get_name() == "out_height" ||
            param.get_name() == "max_rate" ||
            param.get_name() == "interpolation" ||
            param.get_name() == "gpu") {
            update_needed = true;
        }
    }

    if (update_needed) {
        loadParameters();
        updateCameraParameters();
    }

    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    return result;
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<EquirectangularNode>();

    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception during spin: %s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}
