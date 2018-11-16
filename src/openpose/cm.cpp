#pragma once
#include "cm_h.h"


namespace Engine
{

	std::string curLocalTime_module()
	{
		char buf[80];
		time_t timer;
		struct tm t;

		timer = time(NULL);

		localtime_s(&t, &timer);

		strcpy_s(buf, std::to_string(t.tm_mon + 1).c_str());	strcat_s(buf, "/");
		strcat_s(buf, std::to_string(t.tm_mday).c_str());
		strcat_s(buf, " ");
		strcat_s(buf, std::to_string(t.tm_hour).c_str());
		strcat_s(buf, ":");
		strcat_s(buf, std::to_string(t.tm_min).c_str());
		strcat_s(buf, ":");
		strcat_s(buf, std::to_string(t.tm_sec).c_str());
		strcat_s(buf, " ");

		return buf;
	}

	bool InitializeDetector()
		//bool InitializeDetector(char* processExe)
	{
		try
		{
			//google::InitGoogleLogging(google::GetArgv0());                

			LOG(INFO) << "Initialize Detector";

			op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
				__LINE__, __FUNCTION__, __FILE__);
			op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

			if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
				op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
			if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
				op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
					__LINE__, __FUNCTION__, __FILE__);

			poseExtractorCaffe.initializationOnThread();
			poseRenderer.initializationOnThread();

			return true;
		}
		catch (std::exception ex)
		{
			std::cerr << ex.what() << std::endl;
			return false;
		}
	}

	float* RunDetector(unsigned char* frame, const int frameWidth, const int frameHeight, int &size)
		//op::Array<float> RunDetector(cv::Mat frame)
	{

#pragma region

		cv::Mat inputImage = cv::Mat(frameHeight, frameWidth, CV_8UC3);
		inputImage.data = frame;

		if (inputImage.empty())
			op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
		const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };

		std::vector<double> scaleInputToNetInputs;
		std::vector<op::Point<int>> netInputSizes;
		double scaleInputToOutput;
		op::Point<int> outputResolution;
		std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
			= scaleAndSizeExtractor.extract(imageSize);

		const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
		poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
		const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();

		const int getDetected_People = poseKeypoints.getSize(0);
		const int getKeyPoints = poseKeypoints.getSize(1);
		const int getJoints = poseKeypoints.getSize(2);


		//2018. 04. 20.
		size = getDetected_People * getKeyPoints * getJoints;
		float* poseKeypointsArray = (float*)malloc(sizeof(float*) *size);

#pragma endregion 1. Net Resolution

#pragma region
		//op::FaceDetector poseFaceDetector(poseModel);
		//poseFaceDetector.detectFaces(poseKeypoints, scaleInputToOutput);
		//auto outputFaceArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
		//op::FaceCpuRenderer poseFaceRendering(0.1, !FLAGS_disable_blending, (float)FLAGS_alpha_pose);
		//poseFaceRendering.renderFace(outputFaceArray, poseKeypoints);
		//auto datumFaceImage = opOutputToCvMat.formatToCvMat(outputFaceArray);
		//cv::imshow("FaceTest", datumFaceImage);
		//cv::waitKey(0);

#pragma endregion -- Test Face recognition 

#pragma region
		auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
		poseRenderer.renderPose(outputArray, poseKeypoints, scaleInputToOutput);
		auto datumImage = opOutputToCvMat.formatToCvMat(outputArray);
		cv::imshow("Rendered Image", datumImage);
		cv::waitKey(0);
#pragma endregion -- Skip Rendering

#pragma region
		if (getDetected_People <= 0)
		{
			std::cout << curLocalTime_module();
			std::cout << "[nfs_bodypose.cpp:224] [Warning] Get Detected people <= 0" << std::endl;
			//return op::Array<float>();
			return poseKeypointsArray;
		}
#pragma endregion -- Does not detect   (Return point)

#pragma region
#pragma endregion =============BODY DETECT=========================

#pragma region
		std::vector<float> *x_keyPoints = NULL;
		std::vector<float> *y_keyPoints = NULL;
		std::vector<float> *score = NULL;

		float *min_xPos = NULL, *min_yPos = NULL;
		float *max_xPos = NULL, *max_yPos = NULL;
		float validCheck = 0.f;

		x_keyPoints = new std::vector<float>[getDetected_People];
		y_keyPoints = new std::vector<float>[getDetected_People];
		score = new std::vector<float>[getDetected_People];
		min_xPos = new float[getDetected_People];
		max_xPos = new float[getDetected_People];
		min_yPos = new float[getDetected_People];
		max_yPos = new float[getDetected_People];

#pragma endregion 2. Preprocess Postion by Person 

#pragma region
		for (auto person = 0; person < getDetected_People; person++)
		{
			for (auto bodyPart = 0; bodyPart < getKeyPoints; bodyPart++)
			{
				for (auto xyscore = 0; xyscore < getJoints; xyscore++)
				{
					float element = atof(std::to_string(poseKeypoints[{person, bodyPart, xyscore}]).c_str());
					if (xyscore == 0)
						x_keyPoints[person].push_back(element);
					else if (xyscore == 1)
						y_keyPoints[person].push_back(element);
					else if (xyscore == 2)
						score[person].push_back(element);
				}
			}
			min_xPos[person] = 9999999.f;		max_xPos[person] = -1.f;
			min_yPos[person] = 9999999.f;		max_yPos[person] = -1.f;
		}
#pragma endregion 3. Parsing Position Person each

#pragma region
		std::vector<bool> isFrontalPose;
		std::vector<int> isCriteriaPose;
		std::vector<int> isLargestPose;
#pragma endregion -- Criteria of selected Person

#pragma region
		const int RShoulderIdx = 2;
		const int LShoulderIdx = 5;
		bool isFindFrontalPose = false;

		for (int person = 0; person < getDetected_People; person++)
		{
			if (x_keyPoints[person][RShoulderIdx] != 0
				&& y_keyPoints[person][RShoulderIdx] != 0
				&& x_keyPoints[person][LShoulderIdx] != 0
				&& y_keyPoints[person][LShoulderIdx] != 0
				&& x_keyPoints[person][LShoulderIdx] > x_keyPoints[person][RShoulderIdx])
			{
				isFrontalPose.push_back(true);
				isFindFrontalPose = true;
			}
			else
				isFrontalPose.push_back(false);
		}
#pragma endregion 4.0 Frontal Postion 

#pragma region
		int longestPerson = -1;
		float maxLength = 0;
		bool isFind_ATAN_Pose = false;

		for (int person = 0; person < getDetected_People; person++)
		{
			double atanPI = atan2((y_keyPoints[person][LShoulderIdx] - y_keyPoints[person][RShoulderIdx]), (x_keyPoints[person][LShoulderIdx] - x_keyPoints[person][RShoulderIdx])) * 180 / (3.141592);
			float magicNum = 6.0f;
			float theta = 0.5f;

			if (isFindFrontalPose
				&& score[person][LShoulderIdx] > theta
				&& score[person][RShoulderIdx] > theta
				&& atanPI > -1 * magicNum
				&& atanPI < magicNum
				)
			{
				isFind_ATAN_Pose = true;
				float length = pow(x_keyPoints[person][LShoulderIdx] - x_keyPoints[person][RShoulderIdx], 2) + pow(y_keyPoints[person][LShoulderIdx] - y_keyPoints[person][RShoulderIdx], 2);
				if (maxLength < length)
				{
					longestPerson = person;
					maxLength = length;
				}
			}
		}

		for (int person = 0; person < getDetected_People; person++)
			isCriteriaPose.push_back((person == longestPerson) ? person : 0);
#pragma endregion 4.1 Atan Postion Person Detect

#pragma region
		bool isFindLargestPose = false;
		int largestPerson = 0;
		float selected_Size = -1.f;
		for (auto person = 0; person < getDetected_People; person++)
		{
			std::sort(x_keyPoints[person].begin(), x_keyPoints[person].end());
			std::sort(y_keyPoints[person].begin(), y_keyPoints[person].end());
			std::sort(score[person].begin(), score[person].end());
		}

		for (auto person = 0; person < getDetected_People; person++)
		{
			for (auto keypoints = 0; keypoints < getKeyPoints; keypoints++)
			{
				float tmp_x = x_keyPoints[person][keypoints];
				float tmp_y = y_keyPoints[person][keypoints];

				validCheck = (tmp_x > 0) ? 1.f : 0.f;
				if (tmp_x == 0)
					continue;

				if (tmp_x < min_xPos[person])
					min_xPos[person] = tmp_x;
				if (tmp_x > max_xPos[person])
					max_xPos[person] = tmp_x;

				if (tmp_y < min_yPos[person])
					min_yPos[person] = tmp_y;
				if (tmp_y > max_yPos[person])
					max_yPos[person] = tmp_y;
			}

			float width = (max_xPos[person] - min_xPos[person]);
			float height = (max_yPos[person] - min_yPos[person]);
			float region_Size = std::max(width * height, -1 * width*height);

			if (region_Size >= selected_Size)
			{
				selected_Size = region_Size;
				largestPerson = person;
				isFindLargestPose = true;
			}

			for (auto keypoints = 0; keypoints < getKeyPoints; keypoints++)
			{
				float tmp_x = x_keyPoints[person][keypoints];
				float tmp_y = y_keyPoints[person][keypoints];
			}
		}

		for (auto person = 0; person < getDetected_People; person++)
			isLargestPose.push_back((person == largestPerson) ? person : 0);
#pragma endregion 4.2 Largest Postion Person Detect

#pragma region
		int whichOne = -1;
		for (int person = 0; person < getDetected_People; person++)
		{
			std::cout << curLocalTime_module();
			std::cout << "[nfs_bodypose.cpp:418] "
				<< "(1) Is Front Position(bool) : " << isFrontalPose[person] << "\t"
				<< "(2) Index of Largest Body(int): " << isLargestPose[person] << "\t"
				<< "(3) Index of +Alpha Condition(int): " << isCriteriaPose[person] << "\n";

			if (isFrontalPose[person] == true)  (whichOne = person);
		}
#pragma endregion 4.3 Select  WHO?

#pragma region 
		if (isFindLargestPose)
		{
			whichOne = largestPerson;
			float poseScore = 0;
			for (int xyscore = 0; xyscore < getKeyPoints; xyscore++)
				poseScore += (score[whichOne][xyscore] / getKeyPoints);

			int x = min_xPos[whichOne];
			int y = min_yPos[whichOne];
			int w = (max_xPos[whichOne] - min_xPos[whichOne]);
			int oldWidth = w;
			int h = (max_yPos[whichOne] - min_yPos[whichOne]);

			int rows = inputImage.rows;
			int cols = inputImage.cols;
			int oldHeight = h;

			h *= 1.3;
			w = h * 0.8791;

			int minX = x + (oldWidth - w) / 2;
			int maxX = minX + w;
			int minY = y + (oldHeight - h);
			int maxY = minY + h;

			minX = std::max(0, minX);
			maxX = std::min(cols - 1, maxX);
			minY = std::max(0, minY);
			maxY = std::min(rows - 1, maxY);

			int newW = std::abs(maxX - minX);
			int newH = std::abs(maxY - minY);

			cv::Rect _ROI;
			cv::Mat croppedImage;
		}
#pragma endregion 5. Face Detection

#pragma region
#pragma endregion =============FACE DETECT=========================

#pragma region
		else
		{
			std::cout << curLocalTime_module();
			std::cout << "[nfs_bodypose.cpp:502] Deteced people: " << getDetected_People << std::endl;
			std::cout << curLocalTime_module();
			std::cout << "[nfs_bodypose.cpp:504] Could not find face but detect body pose" << std::endl;
		}
#pragma endregion 6. Fail to Face Detection

#pragma region
		delete[] x_keyPoints;
		delete[] y_keyPoints;
		delete[] score;
		delete[] min_xPos;
		delete[] min_yPos;
		delete[] max_xPos;
		delete[] max_yPos;
#pragma endregion 7. Delete Memory	

#pragma region
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

#pragma endregion (+) 2018. 04. 20.

#pragma region
		//return poseKeypoints;
		return poseKeypointsArray;
#pragma endregion 8. Function End point
	}

	bool ReleaseDetector()
		//bool ReleaseDetector()
	{
		poseExtractorCaffe.~PoseExtractorCaffe();
		/*delete &poseExtractorCaffe;
		delete &poseRenderer;*/
		return true;
	}


}
