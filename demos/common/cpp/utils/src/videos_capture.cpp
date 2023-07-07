// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils/videos_capture.h"

#include <string.h>

#ifdef _WIN32
#    include "w_dirent.hpp"
#else
#    include <dirent.h>  // for closedir, dirent, opendir, readdir, DIR
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

class InvalidInput : public std::runtime_error {
public:
    explicit InvalidInput(const std::string& message) noexcept : std::runtime_error(message) {}
};

class OpenError : public std::runtime_error {
public:
    explicit OpenError(const std::string& message) noexcept : std::runtime_error(message) {}
};

class ImreadWrapper : public VideosCapture {
    cv::Mat img;
    bool canRead;
    size_t batch_size;

public:
    ImreadWrapper(const std::string& input, bool loop, size_t batch_size) : VideosCapture{loop}, canRead{true} , batch_size{batch_size}{
        auto startTime = std::chrono::steady_clock::now();

        std::ifstream file(input.c_str());
        if (!file.good())
            throw InvalidInput("Can't find the image by " + input);

        img = cv::imread(input);
        if (!img.data)
            throw OpenError("Can't open the image from " + input);
        else
            readerMetrics.update(startTime);
    }

    double fps() const override {
        return 1.0;
    }

    std::string getType() const override {
        return "IMAGE";
    }

    cv::Mat read() override {
        if (loop)
            return img.clone();
        if (canRead) {
            canRead = false;
            return img.clone();
        }
        return cv::Mat{};
    }

    std::vector<cv::Mat> get_batch_fake() override {
        std::vector<cv::Mat> buffer;
        if (loop){
            img = img.clone();
            for(int n = 0; n < batch_size; n++){
                    buffer.push_back(img);
            }
            return buffer;
        }
        if (canRead) {
            canRead = false;
            img = img.clone();
            for(int n = 0; n < batch_size; n++){
                    buffer.push_back(img);
            }
            return buffer;
        }
        return buffer;
    }

    std::vector<cv::Mat> get_batch() override {
        std::vector<cv::Mat> buffer;
        if (loop){
            img = img.clone();
            for(int n = 0; n < batch_size; n++){
                    buffer.push_back(img);
            }
            return buffer;
        }
        if (canRead) {
            canRead = false;
            img = img.clone();
            for(int n = 0; n < batch_size; n++){
                    buffer.push_back(img);
            }
            return buffer;
        }
        return buffer;
    }
};

class DirReader : public VideosCapture {
    std::vector<std::string> names;
    size_t fileId;
    size_t nextImgId;
    size_t batch_size;
    const size_t initialImageId;
    const size_t readLengthLimit;
    const std::string input;

public:
    DirReader(const std::string& input, bool loop, size_t batch_size, size_t initialImageId, size_t readLengthLimit)
        : VideosCapture{loop},
          fileId{0},
          batch_size{batch_size},
          nextImgId{0},
          initialImageId{initialImageId},
          readLengthLimit{readLengthLimit},
          input{input} {
        DIR* dir = opendir(input.c_str());
        if (!dir)
            throw InvalidInput("Can't find the dir by " + input);
        while (struct dirent* ent = readdir(dir))
            if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, ".."))
                names.emplace_back(ent->d_name);
        closedir(dir);
        if (names.empty())
            throw OpenError("The dir " + input + " is empty");
        sort(names.begin(), names.end());
        size_t readImgs = 0;
        while (fileId < names.size()) {
            cv::Mat img = cv::imread(input + '/' + names[fileId]);
            if (img.data) {
                ++readImgs;
                if (readImgs - 1 >= initialImageId)
                    return;
            }
            ++fileId;
        }
        throw OpenError("Can't read the first image from " + input);
    }

    double fps() const override {
        return 1.0;
    }

    std::string getType() const override {
        return "DIR";
    }

    cv::Mat read() override {
        auto startTime = std::chrono::steady_clock::now();

        while (fileId < names.size() && nextImgId < readLengthLimit) {
            cv::Mat img = cv::imread(input + '/' + names[fileId]);
            ++fileId;
            if (img.data) {
                ++nextImgId;
                readerMetrics.update(startTime);
                return img;
            }
        }

        if (loop) {
            fileId = 0;
            size_t readImgs = 0;
            while (fileId < names.size()) {
                cv::Mat img = cv::imread(input + '/' + names[fileId]);
                ++fileId;
                if (img.data) {
                    ++readImgs;
                    if (readImgs - 1 >= initialImageId) {
                        nextImgId = 1;
                        readerMetrics.update(startTime);
                        return img;
                    }
                }
            }
        }
        return cv::Mat{};
    }

    std::vector<cv::Mat> get_batch_fake() override {
        auto startTime = std::chrono::steady_clock::now();
        std::vector<cv::Mat> buffer;
        while (fileId < names.size() && nextImgId < readLengthLimit) {
            cv::Mat img = cv::imread(input + '/' + names[fileId]);
            ++fileId;
            if (img.data) {
                for(int n = 0; n < batch_size; n++){
                    buffer.push_back(img);
                }
                ++nextImgId;
                readerMetrics.update(startTime);
                return buffer;
            }
        }

        if (loop) {
            fileId = 0;
            size_t readImgs = 0;
            while (fileId < names.size()) {
                cv::Mat img = cv::imread(input + '/' + names[fileId]);
                ++fileId;
                if (img.data) {
                    ++readImgs;
                    if (readImgs - 1 >= initialImageId) {
                        for(int n = 0; n < batch_size; n++){
                            buffer.push_back(img);
                        }
                        nextImgId = 1;
                        readerMetrics.update(startTime);
                        return buffer;
                    }
                }
            }
        }
        return buffer;
    }

    std::vector<cv::Mat> get_batch() override {
        auto startTime = std::chrono::steady_clock::now();
        std::vector<cv::Mat> buffer;
        while (fileId < names.size() && nextImgId < readLengthLimit) {
            cv::Mat img = cv::imread(input + '/' + names[fileId]);
            ++fileId;
            if (img.data) {
                for(int n = 0; n < batch_size; n++){
                    buffer.push_back(img);
                }
                ++nextImgId;
                readerMetrics.update(startTime);
                return buffer;
            }
        }

        if (loop) {
            fileId = 0;
            size_t readImgs = 0;
            while (fileId < names.size()) {
                cv::Mat img = cv::imread(input + '/' + names[fileId]);
                ++fileId;
                if (img.data) {
                    ++readImgs;
                    if (readImgs - 1 >= initialImageId) {
                        for(int n = 0; n < batch_size; n++){
                            buffer.push_back(img);
                        }
                        nextImgId = 1;
                        readerMetrics.update(startTime);
                        return buffer;
                    }
                }
            }
        }
        return buffer;
    }
};

class VideoCapWrapper : public VideosCapture {
    cv::VideoCapture cap;
    bool first_read;
    bool first_batch;
    const read_type type;
    size_t batch_size;
    size_t nextImgId;
    const double initialImageId;
    size_t readLengthLimit;
    std::vector<cv::Mat> bufferQueue;

public:
    VideoCapWrapper(const std::string& input, bool loop, read_type type, size_t batch_size,size_t initialImageId, size_t readLengthLimit)
        : VideosCapture{loop},
          first_read{true},
          first_batch{true},
          type{type},
          batch_size{batch_size},
          nextImgId{0},
          initialImageId{static_cast<double>(initialImageId)} {
        if (0 == readLengthLimit) {
            throw std::runtime_error("readLengthLimit must be positive");
        }
        if (cap.open(input)) {
            this->readLengthLimit = readLengthLimit;
            if (!cap.set(cv::CAP_PROP_POS_FRAMES, this->initialImageId))
                throw OpenError("Can't set the frame to begin with");
            return;
        }
        throw InvalidInput("Can't open the video from " + input);
    }

    double fps() const override {
        return cap.get(cv::CAP_PROP_FPS);
    }

    std::string getType() const override {
        return "VIDEO";
    }

    cv::Mat read() override {
        auto startTime = std::chrono::steady_clock::now();

        if (nextImgId >= readLengthLimit) {
            if (loop && cap.set(cv::CAP_PROP_POS_FRAMES, initialImageId)) {
                nextImgId = 1;
                cv::Mat img;
                
                cap.read(img);
                if (type == read_type::safe) {
                    img = img.clone();
                }
                readerMetrics.update(startTime);
                return img;
            }
            return cv::Mat{};
        }
        cv::Mat img;
        bool success = cap.read(img);
        if (!success && first_read) {
            throw std::runtime_error("The first image can't be read");
        }
        first_read = false;
        if (!success && loop && cap.set(cv::CAP_PROP_POS_FRAMES, initialImageId)) {
            nextImgId = 1;
            cap.read(img);
        } else {
            ++nextImgId;
        }
        if (type == read_type::safe) {
            img = img.clone();
        }
        readerMetrics.update(startTime);
        return img;
    }

    std::vector<cv::Mat> get_batch_fake() override {
        std::vector<cv::Mat> buffer;
        auto startTime = std::chrono::steady_clock::now();

        if (nextImgId >= readLengthLimit) {
            if (loop && cap.set(cv::CAP_PROP_POS_FRAMES, initialImageId)) {
                nextImgId = 1;
                cv::Mat img;
                
                cap.read(img);
                if (type == read_type::safe) {
                    img = img.clone();
                }
                for(int n = 0; n < batch_size; n++){
                    buffer.push_back(img);
                }
                readerMetrics.update(startTime);
                return buffer;
            }
            return buffer;
        }
        cv::Mat img;
        bool success = cap.read(img);
        if (!success && first_read) {
            throw std::runtime_error("The first image can't be read");
        }
        first_read = false;
        if (!success && loop && cap.set(cv::CAP_PROP_POS_FRAMES, initialImageId)) {
            nextImgId = 1;
            cap.read(img);
        } else {
            ++nextImgId;
        }
        if (type == read_type::safe) {
            img = img.clone();
        }
        for(int n = 0; n < batch_size; n++){
            buffer.push_back(img);
        }
        readerMetrics.update(startTime);
        return buffer;
    }

    std::vector<cv::Mat> get_batch() override {
        slog::info << "Yuxincui VideoCapWrapper batch_size: " << batch_size << slog::endl;
        auto startTime = std::chrono::steady_clock::now();
        
        if (nextImgId >= readLengthLimit) {
            slog::info << "Yuxincui VideoCapWrapper nextImgId: " << nextImgId << slog::endl;
            slog::info << "Yuxincui VideoCapWrapper readLengthLimit: " << readLengthLimit << slog::endl;
            std::vector<cv::Mat> emptyQueue;
            return emptyQueue;
        }
        slog::info << "Yuxincui VideoCapWrapper bufferQueue.size(): " << bufferQueue.size() << slog::endl;
        if(first_batch){//first batch
            slog::info << "Yuxincui VideoCapWrapper first batch" << slog::endl;
            for(int n = 0; n < batch_size; n++){
                slog::info << "Yuxincui VideoCapWrapper n: " << n << slog::endl;
                cv::Mat img;
                double cap_pos = cap.get(cv::CAP_PROP_POS_MSEC);
                slog::info << "Yuxincui VideoCapWrapper cap_pos: " << cap_pos << slog::endl;
                slog::info << "Yuxincui VideoCapWrapper nextImgId: " << nextImgId << slog::endl;
                bool success = cap.read(img);
                if (!success) {
                    throw std::runtime_error("The image can't be read");
                }
                first_read = false;
                ++nextImgId;
                if (type == read_type::safe) {
                    img = img.clone();
                }
                bufferQueue.push_back(img);
                slog::info << "Yuxincui VideoCapWrapper bufferQueue.size(): " << bufferQueue.size() << slog::endl;
            } 
            first_batch = false;   
        }else{//not first batch
            slog::info << "Yuxincui VideoCapWrapper not first batch" << slog::endl;
            bufferQueue.erase(bufferQueue.begin());
            slog::info << "Yuxincui VideoCapWrapper erase bufferQueue.size(): " << bufferQueue.size() << slog::endl;
            cv::Mat img;
            double cap_pos = cap.get(cv::CAP_PROP_POS_MSEC);
            slog::info << "Yuxincui VideoCapWrapper cap_pos: " << cap_pos << slog::endl;
            slog::info << "Yuxincui VideoCapWrapper nextImgId: " << nextImgId << slog::endl;
            bool success = cap.read(img);
            if (!success) {
                throw std::runtime_error("The image can't be read");
            }
            ++nextImgId;
            if (type == read_type::safe) {
                img = img.clone();
            }
            bufferQueue.push_back(img);
            slog::info << "Yuxincui VideoCapWrapper push_back bufferQueue.size(): " << bufferQueue.size() << slog::endl;
        }
    readerMetrics.update(startTime);
    return bufferQueue;
    }
};

class CameraCapWrapper : public VideosCapture {
    cv::VideoCapture cap;
    const read_type type;
    size_t batch_size;
    size_t nextImgId;
    size_t readLengthLimit;

public:
    CameraCapWrapper(const std::string& input,
                     bool loop,
                     read_type type,
                     size_t batch_size,
                     size_t readLengthLimit,
                     cv::Size cameraResolution)
        : VideosCapture{loop},
          type{type},
          batch_size{batch_size},
          nextImgId{0} {
        if (0 == readLengthLimit) {
            throw std::runtime_error("readLengthLimit must be positive");
        }
        try {
            if (cap.open(std::stoi(input))) {
                this->readLengthLimit = loop ? std::numeric_limits<size_t>::max() : readLengthLimit;
                cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                cap.set(cv::CAP_PROP_FRAME_WIDTH, cameraResolution.width);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, cameraResolution.height);
                cap.set(cv::CAP_PROP_AUTOFOCUS, true);
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                return;
            }
            throw OpenError("Can't open the camera from " + input);
        } catch (const std::invalid_argument&) {
            throw InvalidInput("Can't find the camera " + input);
        } catch (const std::out_of_range&) { throw InvalidInput("Can't find the camera " + input); }
    }

    double fps() const override {
        return cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30;
    }

    std::string getType() const override {
        return "CAMERA";
    }

    cv::Mat read() override {
        auto startTime = std::chrono::steady_clock::now();

        if (nextImgId >= readLengthLimit) {
            return cv::Mat{};
        }
        cv::Mat img;
        if (!cap.read(img)) {
            throw std::runtime_error("The image can't be captured from the camera");
        }
        if (type == read_type::safe) {
            img = img.clone();
        }
        ++nextImgId;

        readerMetrics.update(startTime);
        return img;
    }

    std::vector<cv::Mat> get_batch_fake() override {
        auto startTime = std::chrono::steady_clock::now();
        std::vector<cv::Mat> buffer;
        if (nextImgId >= readLengthLimit) {
            return buffer;
        }
        cv::Mat img;
        if (!cap.read(img)) {
            throw std::runtime_error("The image can't be captured from the camera");
        }
        if (type == read_type::safe) {
            img = img.clone();
        }
        for(int n = 0; n < batch_size; n++){
            buffer.push_back(img);
        }
        ++nextImgId;

        readerMetrics.update(startTime);
        return buffer;
    }

    std::vector<cv::Mat> get_batch() override {
        auto startTime = std::chrono::steady_clock::now();
        std::vector<cv::Mat> buffer;
        if (nextImgId >= readLengthLimit) {
            return buffer;
        }
        cv::Mat img;
        if (!cap.read(img)) {
            throw std::runtime_error("The image can't be captured from the camera");
        }
        if (type == read_type::safe) {
            img = img.clone();
        }
        for(int n = 0; n < batch_size; n++){
            buffer.push_back(img);
        }
        ++nextImgId;

        readerMetrics.update(startTime);
        return buffer;
    }

};

std::unique_ptr<VideosCapture> openVideosCapture(const std::string& input,
                                                 bool loop,
                                                 read_type type,
                                                 size_t batch_size,
                                                 size_t initialImageId,
                                                 size_t readLengthLimit,
                                                 cv::Size cameraResolution) {
    if (readLengthLimit == 0)
        throw std::runtime_error{"Read length limit must be positive"};
    std::vector<std::string> invalidInputs, openErrors;
    try {
        slog::info << "Yuxincui try ImreadWrapper" << slog::endl;
        return std::unique_ptr<VideosCapture>(new ImreadWrapper{input, loop, batch_size});
    } catch (const InvalidInput& e) { invalidInputs.push_back(e.what()); } catch (const OpenError& e) {
        openErrors.push_back(e.what());
    }

    try {
        slog::info << "Yuxincui try DirReader" << slog::endl;
        return std::unique_ptr<VideosCapture>(new DirReader{input, loop, batch_size, initialImageId, readLengthLimit});
    } catch (const InvalidInput& e) { invalidInputs.push_back(e.what()); } catch (const OpenError& e) {
        openErrors.push_back(e.what());
    }

    try {
        slog::info << "Yuxincui try Video" << slog::endl;
        return std::unique_ptr<VideosCapture>(new VideoCapWrapper{input, loop, type, batch_size, initialImageId, readLengthLimit});
    } catch (const InvalidInput& e) { invalidInputs.push_back(e.what()); } catch (const OpenError& e) {
        openErrors.push_back(e.what());
    }

    try {
        return std::unique_ptr<VideosCapture>(
            new CameraCapWrapper{input, loop, type, batch_size, readLengthLimit, cameraResolution});
    } catch (const InvalidInput& e) { invalidInputs.push_back(e.what()); } catch (const OpenError& e) {
        openErrors.push_back(e.what());
    }

    std::vector<std::string> errorMessages = openErrors.empty() ? invalidInputs : openErrors;
    std::string errorsInfo;
    for (const auto& message : errorMessages) {
        errorsInfo.append(message + "\n");
    }
    throw std::runtime_error(errorsInfo);
}
