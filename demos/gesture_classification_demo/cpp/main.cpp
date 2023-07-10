// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <ratio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <openvino/openvino.hpp>

#include <models/classification_model.h>
#include <models/gesture_classification_model.h>
#include <models/input_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <monitors/presenter.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>
#include <utils/default_flags.hpp>
#include <utils/images_capture.h>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char action_model_message[] =
    "Required. Path to an .xml file with a trained gesture recognition model.";
static const char class_map_message[] = "Required. Path to a file with gesture classes.";
static const char layout_message[] = "Optional. Specify inputs layouts."
                                     " Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available "
                                            "devices is shown below). Default value is CPU. "
                                            "The demo will look for a suitable plugin for device specified.";
static const char num_threads_message[] = "Optional. Specify count of threads.";
static const char num_streams_message[] = "Optional. Specify count of streams.";
static const char num_inf_req_message[] = "Optional. Number of infer requests.";
static const char image_grid_resolution_message[] = "Optional. Set image grid resolution in format WxH. "
                                                    "Default value is 1280x720.";
static const char ntop_message[] = "Optional. Number of top results. Default value is 5. Must be >= 1.";
static const char input_resizable_message[] = "Optional. Enables resizable input.";
static const char no_show_message[] = "Optional. Disable showing of processed images.";
static const char execution_time_message[] = "Optional. Time in seconds to execute program. "
                                             "Default is -1 (infinite time).";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char reverse_input_channels_message[] = "Optional. Switch the input channels order from BGR to RGB.";
static const char mean_values_message[] =
    "Optional. Normalize input by subtracting the mean values per channel. Example: \"255.0 255.0 255.0\"";
static const char scale_values_message[] = "Optional. Divide input by scale values per channel. Division is applied "
                                           "after mean values subtraction. Example: \"255.0 255.0 255.0\"";

DEFINE_bool(h, false, help_message);
DEFINE_string(m_a, "", action_model_message);
DEFINE_string(c, "", class_map_message);
DEFINE_string(layout, "", layout_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_uint32(nireq, 0, num_inf_req_message);
DEFINE_uint32(nt, 5, ntop_message);
DEFINE_string(res, "1280x720", image_grid_resolution_message);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_uint32(time, std::numeric_limits<gflags::uint32>::max(), execution_time_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_bool(reverse_input_channels, true, reverse_input_channels_message);//BGR2RGB
DEFINE_string(mean_values, "", mean_values_message);
DEFINE_string(scale_values, "", scale_values_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "gesture_classification_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << input_message << std::endl;
    std::cout << "    -o \"<path>\"             " << output_message << std::endl;
    std::cout << "    -m_a \"<path>\"           " << action_model_message << std::endl;
    std::cout << "    -auto_resize              " << input_resizable_message << std::endl;
    std::cout << "    -c \"<path>\"             " << class_map_message << std::endl;
    std::cout << "    -limit \"<num>\"          " << limit_message << std::endl;
    std::cout << "    -layout \"<string>\"        " << layout_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams \"<integer>\"     " << num_streams_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << num_inf_req_message << std::endl;
    std::cout << "    -nt \"<integer>\"           " << ntop_message << std::endl;
    std::cout << "    -res \"<WxH>\"              " << image_grid_resolution_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
    std::cout << "    -time \"<integer>\"         " << execution_time_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
    std::cout << "    -reverse_input_channels   " << reverse_input_channels_message << std::endl;
    std::cout << "    -mean_values              " << mean_values_message << std::endl;
    std::cout << "    -scale_values             " << scale_values_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m_a.empty()) {
        throw std::logic_error("Parameter -m_a is not set");
    }

    if (FLAGS_c.empty()) {
        throw std::logic_error("Parameter -c is not set");
    }

    return true;
}

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderClassificationData(ClassificationResult& result, OutputTransform& outputTransform, std::string last_caption) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }
    auto outputImg = result.metaData->asRef<ImageMetaData>().img;
    //slog::info << "Yuxincui outputImg.dims: " << outputImg.dims << slog::endl;
    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    outputTransform.resize(outputImg);
    // Visualizing result data over source image
    cv::Scalar color;//BGR
    color[0]=255;//Blue
    color[1]=0;
    color[2]=0;
    putHighlightedText(outputImg,
                        last_caption,
                        cv::Point2f(10, outputImg.rows - 10),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2);

    return outputImg;
}

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        auto cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe);
        cv::Mat curr_frame;

        //------------------------------ Running routines ----------------------------------------------
        std::vector<std::string> labels_a = GestureClassificationModel::loadLabels(FLAGS_c);
        
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;
        std::unique_ptr<GestureClassificationModel> model(new GestureClassificationModel(FLAGS_m_a, FLAGS_nt, FLAGS_auto_resize, labels_a, FLAGS_layout));
        
        model->setInputsPreprocessing(FLAGS_reverse_input_channels, FLAGS_mean_values, FLAGS_scale_values);

        AsyncPipeline pipeline(std::move(model),
                               ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
                               core);

        Presenter presenter(FLAGS_u, 0);
        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};
        PerformanceMetrics renderMetrics;
        cv::Size outputResolution;
        OutputTransform outputTransform = OutputTransform();

        bool keepRunning = true;
        std::unique_ptr<ResultBase> result;
        std::chrono::steady_clock::duration elapsedSeconds = std::chrono::steady_clock::duration(0);
        std::chrono::seconds testDuration = std::chrono::seconds(3);
        std::chrono::seconds fpsCalculationDuration = std::chrono::seconds(1);
        int64_t frameNum = -1;
        uint32_t framesProcessed = 0;
        while (keepRunning) {
            if (pipeline.isReadyToProcess()) {
                auto startTime = std::chrono::steady_clock::now();
                //--- Capturing frame
                curr_frame = cap->read();
                if (curr_frame.empty()) {
                    // Input stream is over
                    break;
                }
                frameNum = pipeline.submitData(ImageInputData(curr_frame),
                                    std::make_shared<ImageMetaData>(curr_frame, startTime));
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them
            // are available.
            pipeline.waitForData(false);

            //--- Checking for results and rendering data if it's ready
            while ((result = pipeline.getResult(false)) && keepRunning) {
                auto renderingStart = std::chrono::steady_clock::now();
                const ClassificationResult& classificationResult = result->asRef<ClassificationResult>();
                std::string action_class_label = classificationResult.topLabels.front().label;
                slog::info << "Yuxincui main action_class_label: "<< action_class_label << slog::endl;
                cv::Mat outFrame = renderClassificationData(result->asRef<ClassificationResult>(), outputTransform, action_class_label);
                                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                renderMetrics.update(renderingStart);
                metrics.update(classificationResult.metaData->asRef<const ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);
                videoWriter.write(outFrame);
                framesProcessed++;
                if (!FLAGS_no_show) {
                    cv::imshow("classification_demo", outFrame);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    } else {
                        presenter.handleKey(key);
                    }
                }
            }
        }
        // ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();
        for (; framesProcessed <= frameNum; framesProcessed++) {
            result = pipeline.getResult();
            if (result != nullptr) {
                auto renderingStart = std::chrono::steady_clock::now();
                const ClassificationResult& classificationResult = result->asRef<ClassificationResult>();
                std::string action_class_label = classificationResult.topLabels.front().label;
                slog::info << "Yuxincui main action_class_label: "<< action_class_label << slog::endl;
                cv::Mat outFrame = renderClassificationData(result->asRef<ClassificationResult>(), outputTransform, action_class_label);
                                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                renderMetrics.update(renderingStart);
                metrics.update(classificationResult.metaData->asRef<const ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);
                videoWriter.write(outFrame);
                framesProcessed++;
                if (!FLAGS_no_show) {
                    cv::imshow("classification_demo", outFrame);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    } else {
                        presenter.handleKey(key);
                    }
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
