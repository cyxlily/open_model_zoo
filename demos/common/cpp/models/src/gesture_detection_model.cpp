/*
// Copyright (C) 2020-2023 Intel Corporation
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

#include "models/gesture_detection_model.h"
#include "models/image_model.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/ocv_common.hpp>

#include "models/internal_model_data.h"
#include "models/results.h"

struct InputData;

GestureDetectionModel::GestureDetectionModel(const std::string& modelFileName,
                   float confidenceThreshold,
                   bool useAutoResize,
                   const std::string& layout)
    : ImageModel(modelFileName, useAutoResize, layout),
      confidenceThreshold(confidenceThreshold) {}

std::shared_ptr<InternalModelData> GestureDetectionModel::preprocess(const InputData& inputData, ov::InferRequest& request) {
    if (inputsNames.size() > 1) {
        const auto& imageInfoTensor = request.get_tensor(inputsNames[1]);
        const auto info = imageInfoTensor.data<float>();
        info[0] = static_cast<float>(netInputHeight);
        info[1] = static_cast<float>(netInputWidth);
        info[2] = 1;
        request.set_tensor(inputsNames[1], imageInfoTensor);
    }

    return ImageModel::preprocess(inputData, request);
}

std::unique_ptr<ResultBase> GestureDetectionModel::postprocess(InferenceResult& infResult) {
    const ov::Tensor& detectionsTensor = infResult.getFirstOutputTensor();
    size_t detectionsNum = detectionsTensor.get_shape()[detectionsNumId];
    const float* detections = detectionsTensor.data<float>();

    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();
    float widthScale = static_cast<float>(internalData.inputImgWidth) / netInputWidth;
    float heightScale = static_cast<float>(internalData.inputImgHeight) / netInputHeight;
    for (size_t i = 0; i < detectionsNum; i++) {
        float confidence = detections[i * objectSize + 4];
        /** Filtering out objects with confidence < confidence_threshold probability **/
        if (confidence > confidenceThreshold) {
            DetectedObject desc;

            desc.confidence = confidence;

            desc.x = clamp(detections[i * objectSize] * widthScale,
                           0.f,
                           static_cast<float>(internalData.inputImgWidth));
            desc.y = clamp(detections[i * objectSize + 1] * heightScale,
                           0.f,
                           static_cast<float>(internalData.inputImgHeight));
            desc.width = clamp(detections[i * objectSize + 2] * widthScale,
                               0.f,
                               static_cast<float>(internalData.inputImgWidth)) -
                         desc.x;
            desc.height = clamp(detections[i * objectSize + 3] * heightScale,
                                0.f,
                                static_cast<float>(internalData.inputImgHeight)) -
                          desc.y;
            result->objects.push_back(desc);
        }
    }

    return retVal;
}

void GestureDetectionModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input ------------------------------------------------------
    ov::preprocess::PrePostProcessor ppp(model);
    for (const auto& input : model->inputs()) {
        auto inputTensorName = input.get_any_name();
        const ov::Shape& shape = input.get_shape();
        ov::Layout inputLayout = getInputLayout(input);

        if (shape.size() == 4) {  // 1st input contains images
            if (inputsNames.empty()) {
                inputsNames.push_back(inputTensorName);
            } else {
                inputsNames[0] = inputTensorName;
            }

            inputTransform.setPrecision(ppp, inputTensorName);
            ppp.input(inputTensorName).tensor().set_layout({"NHWC"});

            if (useAutoResize) {
                ppp.input(inputTensorName).tensor().set_spatial_dynamic_shape();

                ppp.input(inputTensorName)
                    .preprocess()
                    .convert_element_type(ov::element::f32)
                    .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            }

            ppp.input(inputTensorName).model().set_layout(inputLayout);

            netInputWidth = shape[ov::layout::width_idx(inputLayout)];
            netInputHeight = shape[ov::layout::height_idx(inputLayout)];
        } else if (shape.size() == 2) {  // 2nd input contains image info
            inputsNames.resize(2);
            inputsNames[1] = inputTensorName;
            ppp.input(inputTensorName).tensor().set_element_type(ov::element::f32);
        } else {
            throw std::logic_error("Unsupported " + std::to_string(input.get_shape().size()) +
                                   "D "
                                   "input layer '" +
                                   input.get_any_name() +
                                   "'. "
                                   "Only 2D and 4D input layers are supported");
        }
    }
    model = ppp.build();

    // --------------------------- Prepare output  -----------------------------------------------------
    slog::info << "Yuxincui model outputs().size(): " << model->outputs().size() << slog::endl;
    const auto& output = model->output();
    outputsNames.push_back(output.get_any_name());

    const ov::PartialShape& shape = output.get_partial_shape();
    //const ov::Shape& shape = output.get_shape();
    slog::info << "Yuxincui get_shape 2" << shape << slog::endl;
    const ov::Layout& layout("HW");
    if (shape.size() != 2) {
        throw std::logic_error("Gesture detection output must have 2 dimensions, but had " + std::to_string(shape.size()));
    }

    if (shape[0].is_dynamic()){
        detectionsNumId = 0;
    }else{
        detectionsNumId = ov::layout::height_idx(layout);
    }
    objectSize = shape[1].get_length();
    //objectSize = shape[ov::layout::width_idx(layout)];

    if (objectSize != 5) {
        throw std::logic_error("Gesture detection output must have 5 as a last dimension, but had " +
                               std::to_string(objectSize));
    }
    ppp = ov::preprocess::PrePostProcessor(model);
    ppp.output().tensor().set_element_type(ov::element::f32).set_layout(layout);
    model = ppp.build();
}