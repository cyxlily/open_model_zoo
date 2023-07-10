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

#include "grid_mat.hpp"

static const char help_message[] = "Print a usage message.";
static const char image_message[] = "Required. Path to a folder with images or path to an image file.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char labels_message[] = "Required. Path to .txt file with labels.";
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
DEFINE_string(i, "", image_message);
DEFINE_string(m, "", model_message);
DEFINE_string(labels, "", labels_message);
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
DEFINE_bool(reverse_input_channels, false, reverse_input_channels_message);
DEFINE_string(mean_values, "", mean_values_message);
DEFINE_string(scale_values, "", scale_values_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "classification_benchmark_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << image_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -auto_resize              " << input_resizable_message << std::endl;
    std::cout << "    -labels \"<path>\"          " << labels_message << std::endl;
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

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_labels.empty()) {
        throw std::logic_error("Parameter -labels is not set");
    }

    return true;
}

cv::Mat centerSquareCrop(const cv::Mat& image) {
    if (image.cols >= image.rows) {
        return image(cv::Rect((image.cols - image.rows) / 2, 0, image.rows, image.rows));
    }
    return image(cv::Rect(0, (image.rows - image.cols) / 2, image.cols, image.cols));
}

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics, readerMetrics, renderMetrics;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        std::vector<std::string> imageNames;
        std::vector<cv::Mat> inputImages;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty())
            throw std::runtime_error("No images provided");
        std::sort(imageNames.begin(), imageNames.end());
        for (size_t i = 0; i < imageNames.size(); i++) {
            const std::string& name = imageNames[i];
            auto readingStart = std::chrono::steady_clock::now();
            const cv::Mat& tmpImage = cv::imread(name);
            if (tmpImage.data == nullptr) {
                slog::err << "Could not read image " << name << slog::endl;
                imageNames.erase(imageNames.begin() + i);
                i--;
            } else {
                readerMetrics.update(readingStart);
                // Clone cropped image to keep memory layout dense to enable -auto_resize
                inputImages.push_back(centerSquareCrop(tmpImage).clone());
                size_t lastSlashIdx = name.find_last_of("/\\");
                if (lastSlashIdx != std::string::npos) {
                    imageNames[i] = name.substr(lastSlashIdx + 1);
                } else {
                    imageNames[i] = name;
                }
            }
        }

        //------------------------------ Running routines ----------------------------------------------
        std::vector<std::string> labels = ClassificationModel::loadLabels(FLAGS_labels);

        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        std::unique_ptr<ClassificationModel> model(new ClassificationModel(FLAGS_m, FLAGS_nt, FLAGS_auto_resize, labels, FLAGS_layout));
        model->setInputsPreprocessing(FLAGS_reverse_input_channels, FLAGS_mean_values, FLAGS_scale_values);

        AsyncPipeline pipeline(std::move(model),
                               ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
                               core);

        Presenter presenter(FLAGS_u, 0);
        int width;
        int height;
        std::vector<std::string> gridMatRowsCols = split(FLAGS_res, 'x');
        if (gridMatRowsCols.size() != 2) {
            throw std::runtime_error("The value of GridMat resolution flag is not valid.");
        } else {
            width = std::stoi(gridMatRowsCols[0]);
            height = std::stoi(gridMatRowsCols[1]);
        }
        GridMat gridMat(presenter, cv::Size(width, height));
        bool keepRunning = true;
        std::unique_ptr<ResultBase> result;
        double accuracy = 0;
        bool isTestMode = true;
        std::chrono::steady_clock::duration elapsedSeconds = std::chrono::steady_clock::duration(0);
        std::chrono::seconds testDuration = std::chrono::seconds(3);
        std::chrono::seconds fpsCalculationDuration = std::chrono::seconds(1);
        unsigned int framesNum = 0;
        int64_t frameNum = -1;
        uint32_t framesProcessed = 0;
        long long correctPredictionsCount = 0;
        unsigned int framesNumOnCalculationStart = 0;
        std::size_t nextImageIndex = 0;
        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

        while (keepRunning && elapsedSeconds < std::chrono::seconds(FLAGS_time)) {
            if (elapsedSeconds >= testDuration - fpsCalculationDuration && framesNumOnCalculationStart == 0) {
                framesNumOnCalculationStart = framesNum;
            }
            if (isTestMode && elapsedSeconds >= testDuration) {
                isTestMode = false;
                typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;
                gridMat = GridMat(presenter,
                                  cv::Size(width, height),
                                  cv::Size(16, 9),
                                  (framesNum - framesNumOnCalculationStart) /
                                      std::chrono::duration_cast<Sec>(fpsCalculationDuration).count());
                metrics = PerformanceMetrics();
                startTime = std::chrono::steady_clock::now();
                framesNum = 0;
                correctPredictionsCount = 0;
                accuracy = 0;
            }

            if (pipeline.isReadyToProcess()) {
                auto imageStartTime = std::chrono::steady_clock::now();
                if (nextImageIndex == imageNames.size()) {
                    break;
                }
                slog::info << "Yuxincui nextImageIndex: " << nextImageIndex << slog::endl;
                slog::info << "Yuxincui inputImage.size: " << inputImages[nextImageIndex].size << slog::endl;
                //frameNum = pipeline.submitData(ImageInputData(inputImages[nextImageIndex]),
                                    //std::make_shared<ImageMetaData>(inputImages[nextImageIndex],imageStartTime));
                frameNum = pipeline.submitData(ImageInputData(inputImages[nextImageIndex]),
                                    std::make_shared<ClassificationImageMetaData>(inputImages[nextImageIndex],imageStartTime,0));
                nextImageIndex++;
                
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them
            // are available.
            pipeline.waitForData(false);

            //--- Checking for results and rendering data if it's ready
            while ((result = pipeline.getResult(false)) && keepRunning) {
                auto renderingStart = std::chrono::steady_clock::now();
                const ClassificationResult& classificationResult = result->asRef<ClassificationResult>();
                slog::info << "Yuxincui main classificationResult: "<< classificationResult.metaData << slog::endl;
                if (!classificationResult.metaData) {
                    throw std::invalid_argument("Renderer: metadata is null");
                }
                const ClassificationImageMetaData& classificationImageMetaData =
                    classificationResult.metaData->asRef<const ClassificationImageMetaData>();

                auto outputImg = classificationImageMetaData.img;

                if (outputImg.empty()) {
                    throw std::invalid_argument("Renderer: image provided in metadata is empty");
                }
                std::string label = classificationResult.topLabels.front().label;
                slog::info << "Yuxincui main label: "<< label << slog::endl;
                PredictionResult predictionResult = PredictionResult::Unknown;
                //framesProcessed++;
                framesNum++;
                gridMat.updateMat(outputImg, label, predictionResult);
                renderMetrics.update(renderingStart);
                elapsedSeconds = std::chrono::steady_clock::now() - startTime;
                if (!FLAGS_no_show) {
                    cv::imshow("classification_demo", gridMat.outImg);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    } else if (32 == key || 'r' == key ||
                               'R' == key) {  // press space or r to restart testing if needed
                        isTestMode = true;
                        framesNum = 0;
                        framesNumOnCalculationStart = 0;
                        correctPredictionsCount = 0;
                        accuracy = 0;
                        elapsedSeconds = std::chrono::steady_clock::duration(0);
                        startTime = std::chrono::steady_clock::now();
                    } else {
                        presenter.handleKey(key);
                    }
                }
            }
        }
        // ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();
        slog::info << "Yuxincui main 38" << slog::endl;
        for (; framesProcessed <= frameNum; framesProcessed++) {
            slog::info << "Yuxincui main 39" << slog::endl;
            result = pipeline.getResult();
            if (result != nullptr) {
                auto renderingStart = std::chrono::steady_clock::now();
                const ClassificationResult& classificationResult = result->asRef<ClassificationResult>();
                slog::info << "Yuxincui main classificationResult: "<< classificationResult.metaData << slog::endl;
                if (!classificationResult.metaData) {
                    throw std::invalid_argument("Renderer: metadata is null");
                }
                const ClassificationImageMetaData& classificationImageMetaData =
                    classificationResult.metaData->asRef<const ClassificationImageMetaData>();

                auto outputImg = classificationImageMetaData.img;

                if (outputImg.empty()) {
                    throw std::invalid_argument("Renderer: image provided in metadata is empty");
                }
                std::string label = classificationResult.topLabels.front().label;
                slog::info << "Yuxincui main label: "<< label << slog::endl;
                PredictionResult predictionResult = PredictionResult::Unknown;
                //framesProcessed++;
                framesNum++;
                gridMat.updateMat(outputImg, label, predictionResult);
                renderMetrics.update(renderingStart);
                elapsedSeconds = std::chrono::steady_clock::now() - startTime;
                if (!FLAGS_no_show) {
                    cv::imshow("classification_demo", gridMat.outImg);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    } else if (32 == key || 'r' == key ||
                               'R' == key) {  // press space or r to restart testing if needed
                        isTestMode = true;
                        framesNum = 0;
                        framesNumOnCalculationStart = 0;
                        correctPredictionsCount = 0;
                        accuracy = 0;
                        elapsedSeconds = std::chrono::steady_clock::duration(0);
                        startTime = std::chrono::steady_clock::now();
                    } else {
                        presenter.handleKey(key);
                    }
                }
            }
        }
        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        logLatencyPerStage(readerMetrics.getTotal().latency,
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