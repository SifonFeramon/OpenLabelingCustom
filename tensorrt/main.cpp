#include <iostream>
#include "yolo-tensorrt/modules/class_detector.h"
#include "cxxopts/include/cxxopts.hpp"

using namespace std;

ModelType parseModelType(string const &str) {
    if (str == "v4") {
        return ModelType::YOLOV4;
    } else {
        throw runtime_error("undefined model type");
    }
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("TensorRTTool", "Helps to build model");
    options.add_options()
            ("m, model", "Detector model type", cxxopts::value<string>()->default_value("v4"))
            ("c, config", "YOLO config path", cxxopts::value<string>(),
                    "YOLO V5 config will automatically converted")
            ("w, weights", "YOLO weights path", cxxopts::value<string>(),
                    "YOLO V5 weights will automatically converted")
            ("i, images", "Calibration images path", cxxopts::value<string>());

    auto const result = options.parse(argc, argv);

    Config config_v4;
    config_v4.file_model_cfg = result["c"].as<string>();
    config_v4.file_model_weights = result["w"].as<string>();
    config_v4.calibration_image_list_file_txt = result["i"].as<string>();
    config_v4.net_type = parseModelType(result["m"].as<string>());

    config_v4.
}