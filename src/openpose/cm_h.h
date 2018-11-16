#pragma once

// 2018. 04. 23.
#include <windows.h>

#include <stdexcept>

// Basic
#include <iostream>
#include <gflags/gflags.h>
#include <string>

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/cuda.h>
#include <openpose\face\faceDetector.hpp>
#include <openpose\face\faceDetectorOpenCV.hpp>
#include <openpose\face\faceCpuRenderer.hpp>
#include <openpose\core\netCaffe.hpp>
#include <glog\logging.h>


// Standard
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <iostream>
#include <direct.h>
#include <fstream>
#include <time.h>
#include <utility>
#include <math.h>
#include <direct.h>
#include <fstream>



typedef cv::Mat Image;
typedef std::vector<Image> Image_s;
typedef std::vector<float> Joint;
typedef std::vector<Joint> Joint_s;

#define SUCCESS 1
#define  FAIL -1

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif

// Debugging
DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
	" 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
	" low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg", "Process the desired image.");
// OpenPose
DEFINE_string(model_pose, "COCO", "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
	"`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution, "656x368", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
	" decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
	" closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
	" any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
	" input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
	" e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution, "-1x-1", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
	" input image resolution.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
	" If you want to change the initial scale, you actually want to multiply the"
	" `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
	" background, instead of being rendered into the original image. Related: `part_to_show`,"
	" `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold, 0.08, "Only estimated keypoints whose score confidences are higher than this threshold will be"
	" rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
	" while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
	" more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	" hide it. Only valid for GPU rendering.");

const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
const auto netOutputSize = netInputSize;
const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
const bool enableGoogleLogging = true;

op::CvMatToOpInput cvMatToOpInput;
op::CvMatToOpOutput cvMatToOpOutput;
op::OpOutputToCvMat opOutputToCvMat;
op::FrameDisplayer frameDisplayer{ "OpenPose Tutorial - Example 1", outputSize };
op::ScaleAndSizeExtractor scaleAndSizeExtractor(
	netInputSize
	, outputSize
	, FLAGS_scale_number
	, FLAGS_scale_gap);
op::PoseExtractorCaffe poseExtractorCaffe{
	netInputSize
	, netOutputSize
	, outputSize
	, FLAGS_scale_number
	, poseModel
	, FLAGS_model_folder
	, FLAGS_num_gpu_start
	,{}
	, op::ScaleMode::ZeroToOne
	, enableGoogleLogging };
op::PoseCpuRenderer poseRenderer{
	poseModel
	, (float)FLAGS_render_threshold
	, !FLAGS_disable_blending
	, (float)FLAGS_alpha_pose };

