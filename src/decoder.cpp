#include <iostream>
#include <thread>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "sensor_msgs/image_encodings.hpp"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
    #include <libavutil/pixdesc.h>
}

// A way of getting H.264 decoded. Which one works depends entirely on the machine -
// VAAPI on AMD and Intel graphics, QSV on Intel, CUDA on NVIDIA discrete cards,
// v4l2m2m on Jetson and Raspberry Pi - so the backend is chosen by parameter rather
// than compiled in.
//
// Two mechanisms are in play. A backend either names a dedicated decoder (h264_qsv and
// friends, which manage their own device), or it uses the generic h264 decoder driven
// by an ffmpeg hardware device context, which is how VAAPI works. In the latter case
// hw_pix_fmt is what get_hw_format() has to pick out of the offered formats.
struct HwBackend {
    const char* name;
    const char* decoder_name;
    AVHWDeviceType device_type;
    AVPixelFormat hw_pix_fmt;
};

static const HwBackend kHwBackends[] = {
    {"vaapi",   nullptr,        AV_HWDEVICE_TYPE_VAAPI, AV_PIX_FMT_VAAPI},
    {"qsv",     "h264_qsv",     AV_HWDEVICE_TYPE_NONE,  AV_PIX_FMT_NONE},
    {"cuda",    "h264_cuvid",   AV_HWDEVICE_TYPE_NONE,  AV_PIX_FMT_NONE},
    {"v4l2m2m", "h264_v4l2m2m", AV_HWDEVICE_TYPE_NONE,  AV_PIX_FMT_NONE},
    {"none",    nullptr,        AV_HWDEVICE_TYPE_NONE,  AV_PIX_FMT_NONE},
};

// Order tried when hw_accel is "auto". Software decoding is last, and only reached
// when every hardware path failed to open.
static const char* kAutoOrder[] = {"vaapi", "qsv", "cuda", "v4l2m2m", "none"};

static const HwBackend* FindBackend(const std::string& name) {
    for (const auto& backend : kHwBackends) {
        if (name == backend.name) {
            return &backend;
        }
    }
    return nullptr;
}

// ctx->opaque points at the backend's hw_pix_fmt; see H264DecoderNode::hw_pix_fmt_.
static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    const AVPixelFormat target = *static_cast<const AVPixelFormat*>(ctx->opaque);
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == target) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

class H264DecoderNode : public rclcpp::Node {
private:
    const AVCodec* codec_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVCodecParserContext* parser_ctx_ = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* hw_frame_ = nullptr;
    AVFrame* sw_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    cv::Mat bgr_frame_;
    AVBufferRef *hw_device_ctx_ = nullptr;
    enum AVHWDeviceType hw_type_ = AV_HWDEVICE_TYPE_NONE;
    AVPixelFormat hw_pix_fmt_ = AV_PIX_FMT_NONE;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

    // A decoded frame together with the header of the packet it came from, so the
    // camera's capture time survives the trip through the decoder.
    struct DecodedFrame {
        cv::Mat image;
        std_msgs::msg::Header header;
    };

    std::thread publisher_thread_;
    std::queue<DecodedFrame> frame_publish_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_publisher_thread_{false};
    size_t max_queue_size_ = 10;
    std_msgs::msg::Header current_header_;

    int skip_frame_ = 0;
    int frame_counter_ = 0;
    bool i_frame_only_ = false;
    std::string hw_accel_;
    std::string hw_device_;
    int sw_threads_ = 0;
    bool logged_stream_info_ = false;

    // Opens one specific backend, all the way through avcodec_open2 so that a backend
    // which is present but unusable (the usual case for h264_cuvid on a machine with no
    // NVIDIA card) is rejected here rather than silently producing nothing later.
    bool TryOpenBackend(const HwBackend& backend) {
        const char* device = hw_device_.empty() ? nullptr : hw_device_.c_str();

        codec_ = backend.decoder_name ? avcodec_find_decoder_by_name(backend.decoder_name)
                                      : avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!codec_) {
            RCLCPP_DEBUG(this->get_logger(), "Backend '%s': decoder not compiled into ffmpeg", backend.name);
            return false;
        }

        if (backend.device_type != AV_HWDEVICE_TYPE_NONE) {
            const int err = av_hwdevice_ctx_create(&hw_device_ctx_, backend.device_type, device, nullptr, 0);
            if (err < 0) {
                RCLCPP_DEBUG(this->get_logger(), "Backend '%s': no hardware device (%s)",
                             backend.name, device ? device : "default");
                CleanupFFmpegDecoder();
                return false;
            }
        }

        parser_ctx_ = av_parser_init(codec_->id);
        codec_ctx_ = avcodec_alloc_context3(codec_);
        if (!parser_ctx_ || !codec_ctx_) {
            CleanupFFmpegDecoder();
            return false;
        }

        hw_type_ = backend.device_type;
        hw_pix_fmt_ = backend.hw_pix_fmt;

        if (hw_device_ctx_) {
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            codec_ctx_->opaque = &hw_pix_fmt_;
            codec_ctx_->get_format = get_hw_format;
        }

        // Frame-level threading buffers thread_count frames before emitting the first
        // one, which is exactly the latency we are trying to avoid, so software decoding
        // uses slice threading only.
        codec_ctx_->thread_count = sw_threads_;
        codec_ctx_->thread_type = FF_THREAD_SLICE;
        codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;

        if (avcodec_open2(codec_ctx_, codec_, nullptr) < 0) {
            RCLCPP_DEBUG(this->get_logger(), "Backend '%s': codec failed to open", backend.name);
            CleanupFFmpegDecoder();
            return false;
        }

        pkt_ = av_packet_alloc();
        hw_frame_ = av_frame_alloc();
        sw_frame_ = av_frame_alloc();
        if (!pkt_ || !hw_frame_ || !sw_frame_) {
            CleanupFFmpegDecoder();
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "H.264 backend '%s' opened (decoder=%s%s)",
                    backend.name, codec_->name,
                    backend.device_type != AV_HWDEVICE_TYPE_NONE ? ", hardware accelerated" : ", software");
        return true;
    }

    void InitFFmpegDecoder() {
        if (hw_accel_ == "auto") {
            for (const char* name : kAutoOrder) {
                if (TryOpenBackend(*FindBackend(name))) {
                    return;
                }
                RCLCPP_INFO(this->get_logger(), "H.264 backend '%s' unavailable, trying next", name);
            }
            RCLCPP_ERROR(this->get_logger(), "No H.264 decoder available");
            return;
        }

        const HwBackend* backend = FindBackend(hw_accel_);
        if (!backend) {
            std::string names;
            for (const auto& b : kHwBackends) {
                names += std::string(names.empty() ? "" : ", ") + b.name;
            }
            RCLCPP_ERROR(this->get_logger(), "Unknown hw_accel '%s'. Valid values: auto, %s",
                         hw_accel_.c_str(), names.c_str());
            return;
        }

        // An explicit request is honoured or it fails - silently dropping to software
        // is what made the old NVDEC path look like it was working when it was not.
        if (!TryOpenBackend(*backend)) {
            RCLCPP_ERROR(this->get_logger(),
                         "Requested hw_accel '%s' could not be opened on this machine. "
                         "Use hw_accel:=auto to fall back automatically, or hw_accel:=none for software.",
                         hw_accel_.c_str());
        }
    }

    void PublisherThreadLoop() {
        while (!stop_publisher_thread_) {
            DecodedFrame frame;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !frame_publish_queue_.empty() || stop_publisher_thread_;
                });

                if (stop_publisher_thread_ && frame_publish_queue_.empty()) {
                    break;
                }
                if (frame_publish_queue_.empty()) {
                    continue;
                }
                frame = frame_publish_queue_.front();
                frame_publish_queue_.pop();
            }

            if (!frame.image.empty() && publisher_) {
                auto img_msg = std::make_unique<sensor_msgs::msg::Image>();
                // frame.header carries the camera's capture time from the source packet.
                cv_bridge::CvImage cv_image(frame.header, sensor_msgs::image_encodings::BGR8, frame.image);
                cv_image.toImageMsg(*img_msg);
                publisher_->publish(std::move(img_msg));
            }
        }
    }

    void DecodeAndDisplayPacket(AVPacket* packet) {
        int ret = avcodec_send_packet(codec_ctx_, packet);
        if (ret < 0) {
            return;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                return;
            } else if (ret < 0) {
                return;
            }

            AVFrame* frame_to_display = hw_frame_;

            // Any backend may hand back a frame that lives in GPU memory; ask the pixel
            // format descriptor rather than listing the formats we happen to know about.
            const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get((AVPixelFormat)hw_frame_->format);
            if (desc && (desc->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                if (av_hwframe_transfer_data(sw_frame_, hw_frame_, 0) < 0) {
                    av_frame_unref(hw_frame_);
                    continue;
                }
                frame_to_display = sw_frame_;
            }

            if (!logged_stream_info_) {
                logged_stream_info_ = true;
                RCLCPP_INFO(this->get_logger(), "Decoding %dx%d %s (threads=%d)",
                            frame_to_display->width, frame_to_display->height,
                            av_get_pix_fmt_name((AVPixelFormat)frame_to_display->format),
                            codec_ctx_->thread_count);
            }

            if (!sws_ctx_ && frame_to_display->width > 0 && frame_to_display->height > 0) {
                sws_ctx_ = sws_getContext(
                    frame_to_display->width, frame_to_display->height, (AVPixelFormat)frame_to_display->format,
                    frame_to_display->width, frame_to_display->height, AV_PIX_FMT_BGR24,
                    SWS_POINT, nullptr, nullptr, nullptr);
                
                if (!sws_ctx_) {
                    av_frame_unref(hw_frame_);
                    if (frame_to_display == sw_frame_) av_frame_unref(sw_frame_);
                    return; 
                }
                bgr_frame_.create(frame_to_display->height, frame_to_display->width, CV_8UC3);
            }

            if (sws_ctx_ && !bgr_frame_.empty()) {
                uint8_t* dst_data[4] = { bgr_frame_.data, nullptr, nullptr, nullptr };
                int dst_linesize[4] = { static_cast<int>(bgr_frame_.step[0]), 0, 0, 0 };

                sws_scale(sws_ctx_,
                            (const uint8_t* const*)frame_to_display->data, frame_to_display->linesize,
                            0, frame_to_display->height,
                            dst_data, dst_linesize);

                // Apply frame skipping after decoding
                bool should_publish = true;
                
                if (skip_frame_ > 0 && !i_frame_only_) {
                    // Skip frame logic (only when not in i_frame_only mode)
                    should_publish = (frame_counter_++ % (skip_frame_ + 1) == 0);
                }
                
                if (should_publish) {
                    DecodedFrame frame{bgr_frame_.clone(), current_header_};
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        if (frame_publish_queue_.size() < max_queue_size_) {
                            frame_publish_queue_.push(std::move(frame));
                        }
                    }
                    queue_cv_.notify_one();
                }
            }
            
            av_frame_unref(hw_frame_);
            if (frame_to_display == sw_frame_) {
                av_frame_unref(sw_frame_);
            }
        }
    }

    void CleanupFFmpegDecoder() {
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        if (sw_frame_) {
            av_frame_free(&sw_frame_);
            sw_frame_ = nullptr;
        }
        if (hw_frame_) {
            av_frame_free(&hw_frame_);
            hw_frame_ = nullptr;
        }
        if (pkt_) {
            av_packet_free(&pkt_);
            pkt_ = nullptr;
        }
        if (codec_ctx_) {
            avcodec_close(codec_ctx_); 
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        if (parser_ctx_) {
            av_parser_close(parser_ctx_);
            parser_ctx_ = nullptr;
        }
        if (hw_device_ctx_) {
            av_buffer_unref(&hw_device_ctx_);
            hw_device_ctx_ = nullptr;
        }
        codec_ = nullptr;
        hw_type_ = AV_HWDEVICE_TYPE_NONE;
        hw_pix_fmt_ = AV_PIX_FMT_NONE;
    }

    void compressed_image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        if (msg->format != "h264") {
            return;
        }

        if (!codec_ctx_ || !parser_ctx_ || !pkt_ || !hw_frame_) {
            return;
        }

        // Frames produced while decoding this packet inherit its header, which is how
        // the camera's capture time reaches the published image.
        current_header_ = msg->header;

        const uint8_t* cur_data = msg->data.data();
        size_t remaining_size = msg->data.size();

        while (remaining_size > 0) {
            int bytes_parsed = av_parser_parse2(parser_ctx_, codec_ctx_,
                                                &pkt_->data, &pkt_->size,
                                                cur_data, static_cast<int>(remaining_size),
                                                AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (bytes_parsed < 0) {
                break; 
            }
            cur_data += bytes_parsed;
            remaining_size -= bytes_parsed;

            if (pkt_->size > 0) {
                // Check if this is an I-frame when i_frame_only mode is enabled
                if (i_frame_only_) {
                    // Parse NAL unit type from H.264 stream
                    // The parser sets keyframe flag for I-frames
                    if (parser_ctx_->key_frame == 1) {
                        DecodeAndDisplayPacket(pkt_);
                    }
                } else {
                    DecodeAndDisplayPacket(pkt_);
                }
            }
        }
    }

public:
    H264DecoderNode() : Node("h264_decoder_node") {
        this->declare_parameter("compressed_topic", "/dual_fisheye/image/compressed");
        this->declare_parameter("uncompressed_topic", "/dual_fisheye/image");
        this->declare_parameter("skip_frame", 0);
        this->declare_parameter("i_frame_only", false);
        // auto | vaapi | qsv | cuda | v4l2m2m | none
        this->declare_parameter("hw_accel", "auto");
        // Device to hand the backend, e.g. /dev/dri/renderD128 for VAAPI. Empty = default.
        this->declare_parameter("hw_device", "");
        // Threads for software decoding; 0 lets ffmpeg pick.
        this->declare_parameter("sw_threads", 0);

        std::string subscribe_topic = this->get_parameter("compressed_topic").as_string();
        std::string publish_topic = this->get_parameter("uncompressed_topic").as_string();
        skip_frame_ = this->get_parameter("skip_frame").as_int();
        i_frame_only_ = this->get_parameter("i_frame_only").as_bool();
        hw_accel_ = this->get_parameter("hw_accel").as_string();
        hw_device_ = this->get_parameter("hw_device").as_string();
        sw_threads_ = this->get_parameter("sw_threads").as_int();

        subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            subscribe_topic, 10,
            std::bind(&H264DecoderNode::compressed_image_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(publish_topic, 10);

        publisher_thread_ = std::thread(&H264DecoderNode::PublisherThreadLoop, this);
        
        InitFFmpegDecoder();

        RCLCPP_INFO(this->get_logger(), "H.264 Decoder Node initialized");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", subscribe_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to: %s", publish_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Skip frame: %d, I-frame only: %s", skip_frame_, i_frame_only_ ? "true" : "false");
    }

    ~H264DecoderNode() {
        stop_publisher_thread_ = true;
        queue_cv_.notify_one();
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
        CleanupFFmpegDecoder();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<H264DecoderNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
