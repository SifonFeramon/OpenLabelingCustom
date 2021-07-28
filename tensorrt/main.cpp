#include <iostream>
#include <fstream>
#include "yolo-tensorrt/modules/class_detector.h"
#include "cxxopts/include/cxxopts.hpp"
#include <iomanip>

using namespace std;

ModelType parseModelType(string const &str) {
    if (str == "v4") {
        return ModelType::YOLOV4;
    } else {
        throw runtime_error("undefined model type");
    }
}

cxxopts::ParseResult parseArgs(int argc, char *argv[])
{
    cxxopts::Options options("TensorRTTool", "Helps to build model");
    options.add_options()
            ("model", "Detector model type", cxxopts::value<string>()->default_value("v4"))
            ("config", "YOLO config path", cxxopts::value<string>(),
                    "YOLO V5 config will automatically converted")
            ("weights", "YOLO weights path", cxxopts::value<string>(),
                    "YOLO V5 weights will automatically converted")
            ("input", "input images", cxxopts::value<string>(),
                    "txt file with image paths")
            ("output", "output image folder", cxxopts::value<string>())
            ("calibration", "Calibration images path", cxxopts::value<string>());

    return options.parse(argc, argv);
}

vector<cv::Mat> loadImages(string dir)
{
    vector<cv::Mat> inputs;

    ifstream infile(dir);
    for(string line; getline(infile, line);)
    {
        inputs.push_back(cv::imread(line));
    }

    return inputs;
}

int main(int argc, char *argv[]) {
    try {
        auto const parser = parseArgs(argc, argv);

        Config config_v4;
        config_v4.file_model_cfg = parser["config"].as<string>();
        config_v4.file_model_weights = parser["weights"].as<string>();
        config_v4.calibration_image_list_file_txt = parser["calibration"].as<string>();
        config_v4.net_type = parseModelType(parser["model"].as<string>());

        auto detector = make_unique<Detector>();
        detector->init(config_v4);

        vector<BatchResult> outputs;
        vector<cv::Mat> inputs = loadImages(parser["input"]);
        detector->detect(inputs, outputs);

        for (size_t i = 0; i != 0; ++i) {
            auto const &batch = outputs[i];
            for (auto const &r: batch) {
                cv::rectangle(batch, r.rect, cv::Scalar(255, 0, 0), 2);
                stringstream stream;
                stream << std::fixed << setprecision(2) << "id:" << r.id << "  score:" << r.prob;
                cv::putText(batch, stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5,
                        cv::Scalar(0, 0, 255), 2);
            }
            cv::namedWindow("image" + std::to_string(i), cv::WINDOW_NORMAL);
            cv::imshow("image" + std::to_string(i), batch);
        }
        cv::waitKey(10);
    }
    catch (exception const& e)
    {
        cerr << e.what() << endl;
    }
}