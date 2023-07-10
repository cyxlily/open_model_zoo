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

#include "models/gesture_classification_model.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/op/softmax.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/openvino.hpp>

#include <utils/slog.hpp>

#include "models/results.h"
#include "models/input_data.h"
#include "models/internal_model_data.h"

cv::Mat centerSquareCrop(const cv::Mat& image) {
    if (image.cols >= image.rows) {
        return image(cv::Rect((image.cols - image.rows) / 2, 0, image.rows, image.rows));
    }
    return image(cv::Rect(0, (image.rows - image.cols) / 2, image.cols, image.cols));
}

void GestureClassificationModel::gestureToTensor(const std::vector<cv::Mat> matArray, const ov::Tensor& tensor) {
    if (static_cast<size_t>(matArray[0].channels()) != inputC) {
        throw std::runtime_error("The number of channels for model input and image must match");
    }
    if (inputC != 1 && inputC != 3) {
        throw std::runtime_error("Unsupported number of channels");
    }
    if (tensor.get_element_type() == ov::element::f32) {
        float_t* tensorData = tensor.data<float_t>();
        for (size_t c = 0; c < inputC; c++){
            for (size_t d = 0; d < inputD; d++){
                for (size_t h = 0; h < inputH; h++){
                    for (size_t w = 0; w < inputW; w++){
                        tensorData[c * inputD * inputW * inputH + d * inputW * inputH + h * inputW + w] =
                            getMatValue<float_t>(matArray[d], h, w, c);}}}}
    } else {
        uint8_t* tensorData = tensor.data<uint8_t>();
        if (matArray[0].depth() == CV_32F) {
            throw std::runtime_error("Conversion of cv::Mat from float_t to uint8_t is forbidden");
        }
        for (size_t c = 0; c < inputC; c++){
            for (size_t d = 0; d < inputD; d++){
                for (size_t h = 0; h < inputH; h++){
                    for (size_t w = 0; w < inputW; w++){
                        tensorData[c * inputD * inputW * inputH + d * inputW * inputH + h * inputW + w] =
                            getMatValue<uint8_t>(matArray[d], h, w, c);}}}}
    }
}

ov::Tensor GestureClassificationModel::wrapBlob2Tensor(const cv::Mat &blob) {
    auto matType = blob.type() & CV_MAT_DEPTH_MASK;
    if (matType != CV_8U && matType != CV_32F) {
        throw std::runtime_error("Unsupported mat type for wrapping");
    }
    bool isMatFloat = matType == CV_32F;
    auto precision = isMatFloat ? ov::element::f32 : ov::element::u8;
    auto allocator = std::make_shared<SharedTensorAllocator>(blob);
    return ov::Tensor(precision, ov::Shape{1, inputC, inputD, inputH, inputW}, ov::Allocator(allocator));
}

GestureClassificationModel::GestureClassificationModel(const std::string& modelFileName,
                                         size_t nTop,
                                         bool useAutoResize,
                                         const std::vector<std::string>& labels,
                                         const std::string& layout)
    : ImageModel(modelFileName, useAutoResize, layout),
      nTop(nTop),
      labels(labels) {}

std::shared_ptr<InternalModelData> GestureClassificationModel::preprocess(const InputData& inputData, ov::InferRequest& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    auto img = inputTransform(origImg);//BGR2RGB

    const ov::Tensor& frameTensor = request.get_tensor(inputsNames[0]);  // first input should be image
    ov::Shape tensorShape = frameTensor.get_shape();
    static const ov::Layout layout("NCDHW");
    inputC = tensorShape[ov::layout::channels_idx(layout)];
    inputD = tensorShape[2];
    inputH = tensorShape[ov::layout::height_idx(layout)];
    inputW = tensorShape[ov::layout::width_idx(layout)];

    if (!useAutoResize){
        /* Resize and copy data from the image to the input tensor */
        img = centerSquareCrop(img);
        img = resizeImageExt(img, inputW, inputH, RESIZE_FILL, interpolationMode);
    }
    
    this->imageQueue_.push_back(img);
    if (this->imageQueue_.size() > this->inputD) {
        this->imageQueue_.pop_front();
    }

    if (this->imageQueue_.size() == this->inputD) {
        std::vector<cv::Mat> imageList(this->imageQueue_.begin(), this->imageQueue_.end());
        gestureToTensor(imageList, frameTensor);
        request.set_input_tensor(frameTensor);

    } else if (this->imageQueue_.size() < this->inputD) {
        slog::info << "Yuxincui send dummyMat." << slog::endl;
        int shape[] = {1, 3, 8, 224, 224};
        cv::Mat dummyMat;
        if (frameTensor.get_element_type() == ov::element::f32){
            dummyMat = cv::Mat::zeros(5, shape, CV_32F);

        }else{
            slog::info << "Yuxincui CV_8U" << slog::endl;
            dummyMat = cv::Mat::zeros(5, shape, CV_8U);
        }
            
        auto input_tensor = wrapBlob2Tensor(dummyMat);
        request.set_input_tensor(input_tensor);
    }
    return std::make_shared<InternalImageModelData>(origImg.cols, origImg.rows);
}

std::unique_ptr<ResultBase> GestureClassificationModel::postprocess(InferenceResult& infResult) {
    const ov::Tensor& indicesTensor = infResult.outputsData.find(outputsNames[0])->second;
    const int* indicesPtr = indicesTensor.data<int>();
    const ov::Tensor& scoresTensor = infResult.outputsData.find(outputsNames[1])->second;
    const float* scoresPtr = scoresTensor.data<float>();

    ClassificationResult* result = new ClassificationResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    result->topLabels.reserve(scoresTensor.get_size());
    for (size_t i = 0; i < scoresTensor.get_size(); ++i) {
        int ind = indicesPtr[i];
        if (ind < 0 || ind >= static_cast<int>(labels.size())) {
            throw std::runtime_error("Invalid index for the class label is found during postprocessing");
        }
        result->topLabels.emplace_back(ind, labels[ind], scoresPtr[i]);
    }

    return retVal;
}
std::vector<std::string> GestureClassificationModel::loadLabels(const std::string& labelFilename) {
    std::vector<std::string> labelsList;

    /* Read labels (if any) */
    if (!labelFilename.empty()) {
        std::ifstream inputFile(labelFilename);
        if (!inputFile.is_open())
            throw std::runtime_error("Can't open the labels file: " + labelFilename);
        std::string label;
        while (std::getline(inputFile, label)) {
            slog::info << "Yuxincui label: " << label << slog::endl;
            labelsList.push_back(label);
        }
        if (labelsList.empty())
            throw std::logic_error("File is empty: " + labelFilename);
    }
    slog::info << "Yuxincui labels size: " << labelsList.size() << slog::endl;
    return labelsList;
}

void GestureClassificationModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    slog::info << "Yuxincui prepareInputsOutputs" << slog::endl;
    
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("Classification model wrapper supports topologies with only 1 input");
    }
    const auto& input = model->input();
    inputsNames.push_back(input.get_any_name());

    const ov::Shape& inputShape = input.get_shape();
    const ov::Layout& inputLayout = getInputLayout(input);
    slog::info << "Yuxincui inputShape size: " << inputShape.size() << slog::endl;
    if (inputShape.size() != 5 || inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 5-dimensional model's input is expected");
    }

    const auto width = inputShape[ov::layout::width_idx(inputLayout)];
    const auto height = inputShape[ov::layout::height_idx(inputLayout)];
    if (height != width) {
        throw std::logic_error("Model input has incorrect image shape. Must be NxN square."
                               " Got " +
                               std::to_string(height) + "x" + std::to_string(width) + ".");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    inputTransform.setPrecision(ppp, model->input().get_any_name());
    ppp.input().tensor().set_layout({ "BCDHW" });

    if (useAutoResize) {
        ppp.input().tensor().set_spatial_dynamic_shape();

        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }
    ppp.input().model().set_layout(inputLayout);

    // --------------------------- Prepare output  -----------------------------------------------------
    if (model->outputs().size() != 1) {
        throw std::logic_error("Classification model wrapper supports topologies with only 1 output");
    }
    
    const ov::Shape& outputShape = model->output().get_shape();
    if (outputShape.size() != 2 && outputShape.size() != 4) {
        throw std::logic_error("Classification model wrapper supports topologies only with"
                               " 2-dimensional or 4-dimensional output");
    }

    const ov::Layout outputLayout("NCHW");
    if (outputShape.size() == 4 && (outputShape[ov::layout::height_idx(outputLayout)] != 1 ||
                                    outputShape[ov::layout::width_idx(outputLayout)] != 1)) {
        throw std::logic_error("Classification model wrapper supports topologies only"
                               " with 4-dimensional output which has last two dimensions of size 1");
    }

    size_t classesNum = outputShape[ov::layout::channels_idx(outputLayout)];
    slog::info << "Yuxincui classesNum: " << classesNum << slog::endl;
    if (nTop > classesNum) {
        throw std::logic_error("The model provides " + std::to_string(classesNum) + " classes, but " +
                               std::to_string(nTop) + " labels are requested to be predicted");
    }
    if (classesNum == labels.size() + 1) {
        labels.insert(labels.begin(), "other");
        slog::warn << "Inserted 'other' label as first." << slog::endl;
    } else if (classesNum != labels.size()) {
        throw std::logic_error("Model's number of classes and parsed labels must match (" +
                               std::to_string(outputShape[1]) + " and " + std::to_string(labels.size()) + ')');
    }
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    // --------------------------- Adding softmax and topK output  ---------------------------
    auto nodes = model->get_ops();
    auto softmaxNodeIt = std::find_if(std::begin(nodes), std::end(nodes), [](const std::shared_ptr<ov::Node>& op) {
        return std::string(op->get_type_name()) == "Softmax";
    });

    std::shared_ptr<ov::Node> softmaxNode;
    if (softmaxNodeIt == nodes.end()) {
        auto logitsNode = model->get_output_op(0)->input(0).get_source_output().get_node();
        softmaxNode = std::make_shared<ov::op::v1::Softmax>(logitsNode->output(0), 1);
    } else {
        softmaxNode = *softmaxNodeIt;
    }
    const auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<size_t>{nTop});
    std::shared_ptr<ov::Node> topkNode = std::make_shared<ov::op::v3::TopK>(softmaxNode,
                                                                            k,
                                                                            1,
                                                                            ov::op::v3::TopK::Mode::MAX,
                                                                            ov::op::v3::TopK::SortType::SORT_VALUES);

    auto indices = std::make_shared<ov::op::v0::Result>(topkNode->output(0));
    auto scores = std::make_shared<ov::op::v0::Result>(topkNode->output(1));
    ov::ResultVector res({scores, indices});
    model = std::make_shared<ov::Model>(res, model->get_parameters(), "classification");

    // manually set output tensors name for created topK node
    model->outputs()[0].set_names({"indices"});
    outputsNames.push_back("indices");
    model->outputs()[1].set_names({"scores"});
    outputsNames.push_back("scores");

    // set output precisions
    ppp = ov::preprocess::PrePostProcessor(model);
    ppp.output("indices").tensor().set_element_type(ov::element::i32);
    ppp.output("scores").tensor().set_element_type(ov::element::f32);
    model = ppp.build();
}
