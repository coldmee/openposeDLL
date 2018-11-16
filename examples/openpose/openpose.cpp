#include "openpose.h"

 __declspec(dllexport) float* __cdecl DetectPose(char* imagePath, unsigned char* input, int width, int height, int &size, unsigned char* output)
{
	float* poseKeypointsArray;
	try
	{
		op::log("Starting OpenPose demo...", op::Priority::High);
		const auto timerBegin = std::chrono::high_resolution_clock::now();

		// ------------------------- INITIALIZATION -------------------------
		// Step 1 - Set logging level
		// - 0 will output all the logging messages
		// - 255 will output nothing
		op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
			__LINE__, __FUNCTION__, __FILE__);
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
		// Step 2 - Read Google flags (user defined configuration)
		// outputSize
		const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
		// poseModel
		const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
		// Check no contradictory flags enabled
		if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
			op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
		if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
			op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
				__LINE__, __FUNCTION__, __FILE__);
		// Logging
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
		// Step 3 - Initialize all required classes
		op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
		op::CvMatToOpInput cvMatToOpInput{ poseModel };
		op::CvMatToOpOutput cvMatToOpOutput;
		op::PoseExtractorCaffe poseExtractorCaffe{ poseModel, FLAGS_model_folder, FLAGS_num_gpu_start };
		op::PoseCpuRenderer poseRenderer{ poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
			(float)FLAGS_alpha_pose };
		op::OpOutputToCvMat opOutputToCvMat;
		op::FrameDisplayer frameDisplayer{ "OpenPose Tutorial - Example 1", outputSize };
		// Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
		poseExtractorCaffe.initializationOnThread();
		poseRenderer.initializationOnThread();

		// ------------------------- POSE ESTIMATION AND RENDERING -------------------------
		// Step 1 - Read and load image, error if empty (possibly wrong path)
		// Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
		//cv::Mat inputImage = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);

		cv::Mat inputImage = cv::Mat(height, width, CV_8UC3);
		inputImage.data = input;
		//op::loadImage(imagePath, CV_LOAD_IMAGE_COLOR);

		
		FLAGS_image_path = imagePath;
		if (inputImage.empty())
			op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
		const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };
		// Step 2 - Get desired scale sizes
		std::vector<double> scaleInputToNetInputs;
		std::vector<op::Point<int>> netInputSizes;
		double scaleInputToOutput;
		op::Point<int> outputResolution;
		std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
			= scaleAndSizeExtractor.extract(imageSize);
		// Step 3 - Format input image to OpenPose input and output formats
		const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
		outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
		// Step 4 - Estimate poseKeypoints
		poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
		const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();

		//cm Ãß°¡
		const int getDetected_People = poseKeypoints.getSize(0);
		const int getKeyPoints = poseKeypoints.getSize(1);
		const int getJoints = poseKeypoints.getSize(2);

		size = getDetected_People * getKeyPoints * getJoints;
		poseKeypointsArray = (float*)malloc(sizeof(float*) *size);

		// Step 5 - Render poseKeypoints
		poseRenderer.renderPose(outputArray, poseKeypoints, scaleInputToOutput);
		// Step 6 - OpenPose output format to cv::Mat
		outputImage = opOutputToCvMat.formatToCvMat(outputArray);


		//////////////////////show

		// Measuring total time
		const auto now = std::chrono::high_resolution_clock::now();
		const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - timerBegin).count()
			* 1e-9;
		const auto message = "OpenPose demo successfully finished. Total time: "
			+ std::to_string(totalTimeSec) + " seconds.";
		op::log(message, op::Priority::High);
		// Return successful message

		// Array -> float*
		int index = 0;
		for (auto person = 0; person < getDetected_People; person++)
		{
			for (auto bodyPart = 0; bodyPart < getKeyPoints; bodyPart++)
			{
				for (auto xyscore = 0; xyscore < getJoints; xyscore++)
				{
					float element = atof(std::to_string(poseKeypoints[{person, bodyPart, xyscore}]).c_str());
					poseKeypointsArray[index++] = element;
				}
			}
		}

		return poseKeypointsArray;
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		return poseKeypointsArray;
	}
}

extern "C" __declspec(dllexport) unsigned char* __cdecl GetRenderedImage(int &size)
//extern "C" __declspec(dllexport) void __cdecl ShowResult()
{
	// ------------------------- SHOWING RESULT AND CLOSING -------------------------
	// Show results
	//frameDisplayer.displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
	//cv::imshow("result", outputImage);
	//cv::waitKey(0);
	
	size = outputImage.rows * outputImage.cols * 3;
	unsigned char* output = (unsigned char*)malloc((sizeof(unsigned char*)) * size); //sizeof(unsigned char*)
	memcpy(output, outputImage.data, size);
	//const auto message2 = "output.data:" + std::to_string(size);
	//op::log(message2, op::Priority::High);
	//op::log("dd");
	//op::log(std::to_string(outputImage.depth()));
	//op::log(std::to_string(outputImage.rows));
	//op::log(std::to_string(outputImage.cols));
	//op::log(std::to_string(outputImage.channels()));
	return output;
	//return outputArray.getPtr();
}
//
//int main(int argc, char *argv[])
//{
//	// Parsing command line flags
//	gflags::ParseCommandLineFlags(&argc, &argv, true);
//
//	//image_path = "";
//
//	// Running openPoseTutorialPose1
//	return openPoseTutorialPose1();
//}

