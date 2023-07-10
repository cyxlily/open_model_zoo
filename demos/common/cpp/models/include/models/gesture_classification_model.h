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

#pragma once
#include <stddef.h>

#include <memory>
#include <string>
#include <vector>

#include "models/image_model.h"

namespace ov {
class Model;
}  // namespace ov
struct InferenceResult;
struct ResultBase;

class GestureClassificationModel : public ImageModel {
public:
    /// Constructor
    /// @param modelFileName name of model to load.
    /// @param nTop - number of top results.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by openvino.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    /// @param labels - array of labels for every class.
    /// @param layout - model input layout
    GestureClassificationModel(const std::string& modelFileName,
                        size_t nTop,
                        bool useAutoResize,
                        const std::vector<std::string>& labels,
                        const std::string& layout = "");
    std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, ov::InferRequest& request) override;
    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    static std::vector<std::string> loadLabels(const std::string& labelFilename);

protected:
    size_t nTop;
    std::vector<std::string> labels;
    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
private:
    size_t inputC = 0;
    size_t inputD = 0;
    size_t inputH = 0;
    size_t inputW = 0;
    std::deque<cv::Mat> imageQueue_;
    ov::Tensor wrapBlob2Tensor(const cv::Mat &blob);
    void gestureToTensor(const std::vector<cv::Mat> matArray, const ov::Tensor& tensor);

};
