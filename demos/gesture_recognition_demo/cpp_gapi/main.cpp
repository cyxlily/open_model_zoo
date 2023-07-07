// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <monitors/presenter.h>
#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "custom_kernels.hpp"
#include "gesture_recognition_demo_gapi.hpp"
#include "gflags/gflags.h"
#include "stream_source.hpp"
#include "tracker.hpp"
#include "utils.hpp"
#include "visualizer.hpp"

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    /** ---------- Parsing and validating input arguments ----------**/
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_m_a.empty())
        throw std::logic_error("Parameter -m_a is not set");
    if (FLAGS_m_d.empty())
        throw std::logic_error("Parameter -m_d is not set");
    if (FLAGS_c.empty())
        throw std::logic_error("Parameter -c is not set");

    return true;
}

namespace nets {
G_API_NET(PersonDetection, <cv::GMat(cv::GMat)>, "person_detection");
G_API_NET(ActionRecognition, <cv::GMat(cv::GMat)>, "action_recognition");
}  // namespace nets

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** Get information about nets **/
        const auto pd_net_shape = getNetShape(FLAGS_m_d);
        const auto ar_net_shape = getNetShape(FLAGS_m_a);

        /** Get information about frame from cv::VideoCapture **/
        slog::info << "Yuxincui vedio 1" << slog::endl;
	std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i,
                                                               FLAGS_loop,
                                                               read_type::safe,
                                                               0,
                                                               std::numeric_limits<size_t>::max(),
                                                               stringToSize(FLAGS_res));
        slog::info << "Yuxincui vedio 2" << slog::endl;
	const auto tmp = cap->read();
	slog::info << "Yuxincui vedio 3" << slog::endl;
        cap.reset();
	slog::info << "Yuxincui vedio 4" << slog::endl;
        cv::Size frame_size = cv::Size{tmp.cols, tmp.rows};
	slog::info << "Yuxincui vedio 5" << slog::endl;
        cap = openImagesCapture(FLAGS_i,
                                FLAGS_loop,
                                read_type::safe,
                                0,
                                std::numeric_limits<size_t>::max(),
                                stringToSize(FLAGS_res));
        slog::info << "Yuxincui vedio 6" << slog::endl;
        /** Share runtime id with graph **/
        auto current_person_id_m = std::make_shared<size_t>(0);
        slog::info << "Yuxincui vedio 7" << slog::endl;
        /** ---------------- Main graph of demo ---------------- **/
        /** Graph inputs **/
        cv::GArray<cv::GMat> batch;
	slog::info << "Yuxincui vedio 8" << slog::endl;
        cv::GOpaque<std::shared_ptr<size_t>> current_person_id;
        slog::info << "Yuxincui vedio 9" << slog::endl;
        cv::GMat fast_frame = custom::GetFastFrame::on(batch, frame_size);
        slog::info << "Yuxincui vedio 10" << slog::endl;
        /** Person detection **/
        cv::GMat detections = cv::gapi::infer<nets::PersonDetection>(fast_frame);
        slog::info << "Yuxincui vedio 11" << slog::endl;
        /** Get ROIs from detections **/
        cv::GArray<TrackedObject> objects = custom::ExtractBoundingBox::on(detections, fast_frame, pd_net_shape);
        slog::info << "Yuxincui vedio 12" << slog::endl;
        /** Track detection **/
        cv::GArray<TrackedObject> tracked = custom::TrackPerson::on(fast_frame, objects);
        slog::info << "Yuxincui vedio 13" << slog::endl;
        /** Create clip for AR net **/
        cv::GArray<cv::GMat> clip =
            custom::ConstructClip::on(batch, tracked, ar_net_shape, frame_size, current_person_id);
        slog::info << "Yuxincui vedio 14" << slog::endl;
        /** Action recognition **/
        cv::GArray<cv::GMat> actions = cv::gapi::infer2<nets::ActionRecognition>(fast_frame, clip);
        slog::info << "Yuxincui vedio 15" << slog::endl;
        /** Get action label **/
        cv::GOpaque<int> label = custom::GestureRecognitionPostprocessing::on(actions, static_cast<float>(FLAGS_t));
        slog::info << "Yuxincui vedio 16" << slog::endl;
        /** Inputs and outputs of graph **/
        auto graph = cv::GComputation(cv::GIn(batch, current_person_id), cv::GOut(fast_frame, tracked, label));
        /** ---------------- End of graph ---------------- **/
        /** Configure networks **/
        // clang-format off
	slog::info << "Yuxincui vedio 17" << slog::endl;
        auto person_detection =
            cv::gapi::ie::Params<nets::PersonDetection>{
                FLAGS_m_d,  // path to model
                fileNameNoExt(FLAGS_m_d) + ".bin",  // path to weights
                FLAGS_d_d  // device to use
            }.cfgOutputLayers({"boxes"});  // This clarification here because of
                                              // GAPI take the first layer name from OutputsInfo
                                              // for one output G_API_NET API
        // clang-format on
        slog::info << "The Person Detection ASL model " << FLAGS_m_d << " is loaded to " << FLAGS_d_d << " device."
                   << slog::endl;

        // clang-format off
        auto action_recognition =
            cv::gapi::ie::Params<nets::ActionRecognition>{
                FLAGS_m_a,  // path to model
                fileNameNoExt(FLAGS_m_a) + ".bin",  // path to weights
                FLAGS_d_a  // device to use
            }.cfgOutputLayers({"output"});  // This clarification here because of
                                               // GAPI take the first layer name from OutputsInfo
                                               // for one output G_API_NET API
        // clang-format on
        slog::info << "The Action Recognition model " << FLAGS_m_a << " is loaded to " << FLAGS_d_a << " device."
                   << slog::endl;

        /** Custom kernels **/
	slog::info << "Yuxincui vedio 18" << slog::endl;
        auto kernels = custom::kernels();
	slog::info << "Yuxincui vedio 19" << slog::endl;
        auto networks = cv::gapi::networks(person_detection, action_recognition);
	slog::info << "Yuxincui vedio 20" << slog::endl;
        auto comp = cv::compile_args(kernels, networks);
	slog::info << "Yuxincui vedio 21" << slog::endl;
        auto pipeline = graph.compileStreaming(std::move(comp));
	slog::info << "Yuxincui vedio 22" << slog::endl;

        /** Output containers for results **/
        cv::Mat out_frame;
        TrackedObjects out_detections;
        int out_label_number;

        /** ---------------- The execution part ---------------- **/
        const float batch_constant_FPS = 15;
        auto drop_batch = std::make_shared<bool>(false);
        slog::info << "Yuxincui vedio 23" << slog::endl;
	pipeline.setSource(cv::gin(cv::gapi::wip::make_src<custom::GestRecCapSource>(cap,
                                                                                     frame_size,
                                                                                     static_cast<int>(ar_net_shape[1]),
                                                                                     batch_constant_FPS,
                                                                                     drop_batch),
                                   current_person_id_m));

        slog::info << "Yuxincui vedio 24" << slog::endl;
	std::string gestureWindowName = "Gesture";

        cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};
	slog::info << "Yuxincui vedio 25" << slog::endl;
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);
        slog::info << "Yuxincui vedio 26" << slog::endl;
        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        /** Fill labels container from file with classes **/
        const auto labels = fill_labels(FLAGS_c);
        size_t current_id = 0;
        size_t last_id = current_id;
        int gesture = 0;

        /** Configure drawing utilities **/
	slog::info << "Yuxincui vedio 27" << slog::endl;
        Visualizer visualizer(FLAGS_no_show, gestureWindowName, labels, FLAGS_s);

        bool isStart = true;
        const auto startTime = std::chrono::steady_clock::now();
        slog::info << "Yuxincui vedio 28" << slog::endl;
	pipeline.start();
	slog::info << "Yuxincui vedio 29" << slog::endl;
        while (pipeline.pull(cv::gout(out_frame, out_detections, out_label_number))) {
            slog::info << "Yuxincui vedio 30" << slog::endl;
            /** Put FPS to frame**/
            if (isStart) {
		slog::info << "Yuxincui vedio 31" << slog::endl;
                metrics.update(startTime,
                               out_frame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
                isStart = false;
            } else {
                metrics.update({},
                               out_frame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
            }
            slog::info << "Yuxincui vedio 31" << slog::endl;
            /** Display system parameters **/
            presenter.drawGraphs(out_frame);
	    slog::info << "Yuxincui vedio 32" << slog::endl;
            /** Display the results **/
            visualizer.show(out_frame, out_detections, out_label_number, current_id, gesture);
            gesture = 0;

            videoWriter.write(out_frame);

            /** Controls **/
            int key = cv::waitKey(1);
            if (key == 0x1B)
                break;  // (esc button) exit
            else if (key >= 48 && key <= 57)
                current_id = key - 48;  // buttons for person id
            else if (key == 0x0D)
                out_label_number = -1;  // (Enter) reset last gesture
            else if (key == 'f')
                gesture = 1;  // next gesture
            else if (key == 'b')
                gesture = -1;  // prev gesture
            else
                presenter.handleKey(key);

            /** Share id with graph **/
            if (current_id < out_detections.size()) {
                *drop_batch = last_id != current_id;
                *current_person_id_m = current_id;
                last_id = current_id;
            }
        }
        slog::info << "Metrics report:" << slog::endl;
        slog::info << "\tFPS: " << std::fixed << std::setprecision(1) << metrics.getTotal().fps << slog::endl;
        slog::info << presenter.reportMeans() << slog::endl;
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    return 0;
}
