/*
// Copyright (C) 2018-2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/gesture_detection_model.h>
#include <models/detection_model_ssd.h>
#include <models/input_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <monitors/presenter.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/default_flags.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char detection_model_message[] = 
    "Required. Path to an .xml file with a trained person detector model.";
static const char sample_dir_message[] =
    "Optional. Path to a directory with video samples of gestures.";
static const char action_threshold_message[] = 
    "Optional. Threshold for the predicted score of an action.";
static const char device_message[] =
    "Optional. Specify a device to infer on (the list of available devices is shown below). Use "
    "'-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use "
    "'-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU";
static const char no_show_message[] = "Optional. Do not visualize inference results.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char layout_message[] = "Optional. Specify inputs layouts."
                                     " Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.";
static const char input_resizable_message[] =
    "Optional. Enables resizable input with support of ROI crop & auto resize.";                                     
DEFINE_bool(h, false, help_message);
DEFINE_string(m_d, "", detection_model_message);
DEFINE_string(s, "", sample_dir_message);
DEFINE_double(t, 0.5, action_threshold_message);
DEFINE_string(d, "CPU", device_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_string(layout, "", layout_message);
DEFINE_bool(auto_resize, false, input_resizable_message);

/**
 * \brief This function shows a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "gesture_recognition_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -m_d \"<path>\"           " << detection_model_message << std::endl;
    std::cout << "    -i \"<path>\"             " << input_message << std::endl;
    std::cout << "    -o \"<path>\"             " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"          " << limit_message << std::endl;
    std::cout << "    -s \"<path>\"             " << sample_dir_message << std::endl;
    std::cout << "    -t                        " << action_threshold_message << std::endl;
    std::cout << "    -d \"<device>\"           " << device_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
        
    if (FLAGS_m_d.empty()) {
        throw std::logic_error("Parameter -m_d is not set");
    }
        
    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    return true;
}

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(DetectionResult& result, OutputTransform& outputTransform) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<ImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    outputTransform.resize(outputImg);
    // Visualizing result data over source image
    slog::debug << " -------------------- Frame # " << result.frameId << "--------------------" << slog::endl;
    slog::debug << " Class ID  | Confidence | XMIN | YMIN | XMAX | YMAX " << slog::endl;


    for (auto& obj : result.objects) {
        slog::debug << " " << std::left << std::setw(9) << obj.label << " | " << std::setw(10) << obj.confidence
                    << " | " << std::setw(4) << int(obj.x) << " | " << std::setw(4) << int(obj.y) << " | "
                    << std::setw(4) << int(obj.x + obj.width) << " | " << std::setw(4) << int(obj.y + obj.height)
                    << slog::endl;
        outputTransform.scaleRect(obj);
        std::ostringstream conf;
        conf << "Person:" << std::fixed << std::setprecision(1) << obj.confidence * 100 << '%';
        cv::Scalar color;
        color[0]=0;
        color[1]=255;//Green
        color[2]=0;
        putHighlightedText(outputImg,
                           conf.str(),
                           cv::Point2f(obj.x, obj.y - 5),
                           cv::FONT_HERSHEY_COMPLEX_SMALL,
                           1,
                           color,
                           2);
        cv::rectangle(outputImg, obj, color, 2);
    }

    try {
        for (auto& lmark : result.asRef<RetinaFaceDetectionResult>().landmarks) {
            outputTransform.scaleCoord(lmark);
            cv::circle(outputImg, lmark, 2, cv::Scalar(0, 255, 255), -1);
        }
    } catch (const std::bad_cast&) {}

    return outputImg;
}

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        const auto& strAnchors = split("", ',');
        const auto& strMasks = split("", ',');

        std::vector<float> anchors;
        std::vector<int64_t> masks;
        try {
            for (auto& str : strAnchors) {
                anchors.push_back(std::stof(str));
            }
        } catch (...) { throw std::runtime_error("Invalid anchors list is provided."); }

        try {
            for (auto& str : strMasks) {
                masks.push_back(std::stoll(str));
            }
        } catch (...) { throw std::runtime_error("Invalid masks list is provided."); }

        //------------------------------- Preparing Input ------------------------------------------------------
        auto cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe);
        cv::Mat curr_frame;

        //------------------------------ Running Detection routines ----------------------------------------------
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;
        std::vector<std::string> labels;
        //std::unique_ptr<ModelSSD> person_detector(new ModelSSD(FLAGS_m_d, static_cast<float>(FLAGS_t), FLAGS_auto_resize, labels, FLAGS_layout));
        std::unique_ptr<GestureDetectionModel> person_detector(new GestureDetectionModel(FLAGS_m_d, static_cast<float>(FLAGS_t), FLAGS_auto_resize, FLAGS_layout));
        person_detector->setInputsPreprocessing(false, "", "");

        AsyncPipeline pipeline(std::move(person_detector),
                               ConfigFactory::getUserConfig(FLAGS_d, 0, "", 0),
                               core);
        
        Presenter presenter(FLAGS_u);

        bool keepRunning = true;
        int64_t frameNum = -1;
        std::unique_ptr<ResultBase> result;
        uint32_t framesProcessed = 0;

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        PerformanceMetrics renderMetrics;

        cv::Size outputResolution;
        OutputTransform outputTransform = OutputTransform();
        while (keepRunning) {
            if (pipeline.isReadyToProcess()) {
                auto startTime = std::chrono::steady_clock::now();
                //--- Capturing frame
                curr_frame = cap->read();
                slog::info << "Yuxincui curr_frame.size: " << curr_frame.size << slog::endl;
                if (curr_frame.empty()) {
                    // Input stream is over
                    break;
                }
                frameNum = pipeline.submitData(ImageInputData(curr_frame),
                                               std::make_shared<ImageMetaData>(curr_frame, startTime));
            }

            if (frameNum == 0) {
                    outputResolution = curr_frame.size();
            }
            //--- Waiting for free input slot or output data available. Function will return immediately if any of them
            // are available.
            pipeline.waitForData();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to DetectionResult*
            //    and use your own processing instead of calling renderDetectionData().
            while (keepRunning && (result = pipeline.getResult())) {
                auto renderingStart = std::chrono::steady_clock::now();
                cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), outputTransform);

                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                renderMetrics.update(renderingStart);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);

                videoWriter.write(outFrame);
                framesProcessed++;

                if (!FLAGS_no_show) {
                    cv::imshow("Detection Results", outFrame);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    } else {
                        presenter.handleKey(key);
                    }
                }
            }
        }  // while(keepRunning)

        // ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();
        for (; framesProcessed <= frameNum; framesProcessed++) {
            result = pipeline.getResult();
            if (result != nullptr) {
                auto renderingStart = std::chrono::steady_clock::now();
                cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), outputTransform);
                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                renderMetrics.update(renderingStart);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);
                videoWriter.write(outFrame);
                if (!FLAGS_no_show) {
                    cv::imshow("Detection Results", outFrame);
                    //--- Updating output window
                    cv::waitKey(1);
                }
            }
        }
        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        logLatencyPerStage(cap->getMetrics().getTotal().latency,
                           pipeline.getPreprocessMetrics().getTotal().latency,
                           pipeline.getInferenceMetircs().getTotal().latency,
                           pipeline.getPostprocessMetrics().getTotal().latency,
                           renderMetrics.getTotal().latency);
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
