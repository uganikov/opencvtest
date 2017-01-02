#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <algorithm>
#include <filesystem> 
#include <iterator>
#if _DEBUG
#include <windows.h>
#endif

using namespace cv;

const Size MAX_DETECT_SIZE = Size(100, 200);
const int MAX_MISS_FRAME = 3;
const double MIN_NEW_DETECT_INTERSECTION_RATE = 0.1;


class MyTracker {
private:
	static int next_id;
	int id;
	int n_miss_frame = 0;
//	bool mark_used = false;
	Rect2d rect;
public:
	// フレーム画像と追跡対象(Rect)で初期化
	MyTracker(const Rect2d& _rect)
		: id(next_id++), rect(_rect) {
	}
	// 新しい検出(Rect)を登録。現在位置と近ければ受理してn_miss_frameをリセット(true)
	// そうでなければ(false)
	bool registerNewDetect(const Rect2d& _new_detect) {
//		if (mark_used)return false;
		double intersection_rate = 1.0 * (_new_detect & rect).area() ;
		bool is_registered = intersection_rate > MIN_NEW_DETECT_INTERSECTION_RATE;
		if (intersection_rate) {
			n_miss_frame = 0;
			rect = _new_detect;
	//		mark_used = true;
		}
		else {
			std::string str;
			str += std::to_string(intersection_rate) + "\n";
			OutputDebugString(str.c_str());
		}
		return is_registered;
	}
	// trackerの現在地を_imageに書き込む
	void draw(Mat& _image) const {
		rectangle(_image, rect, Scalar(255, 0, 0), 2, 1);
		putText(_image, format("%03d", id), Point(rect.x + 5, rect.y + 17),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1, CV_AA);
	}
	void mask(Mat& _image) const {
		rectangle(_image, rect, Scalar(0, 0, 0), CV_FILLED, 1);
	}

	bool update() {
		n_miss_frame++;
		return n_miss_frame < MAX_MISS_FRAME;
	}
	int get_id() {
		return id;
	}
};
int MyTracker::next_id = 0;

class Detection {
public:
	Rect r;
	int sz;
};

int main(int argc, char* argv[]) {
	//			std::vector<MyTracker> trackers;
	//Ptr<BackgroundSubtractor> pMOG = createBackgroundSubtractorMOG2();
	Ptr<BackgroundSubtractorKNN> pKNN = createBackgroundSubtractorKNN();
	//Ptr<BackgroundSubtractorMOG2> pMOG = createBackgroundSubtractorMOG2();
//	pKNN->setShadowValue(10);
	std::vector<MyTracker> trackers;

	

	std::tr2::sys::path p("c:\\nwcam\\161029");
	std::tr2::sys::path outpath("c:\\nwcam\\161029\\out\\");
	int f = 0;

	cv::HOGDescriptor hog;
	hog.setSVMDetector(hog.getDefaultPeopleDetector());
	std::vector< cv::Rect > locations;

	// ビデオライタ
	int fps = 4;
	cv::VideoWriter writer("c:\\nwcam\\161029\\out\\out.mp4", CV_FOURCC('H', '2', '6', '4'), fps, Size(640,360));

	for(std::tr2::sys::directory_iterator next(p),end; next != end; next++){
		VideoCapture video(next->path().string());
		if (video.isOpened()) {
			do{
				Mat display_frame;
				Mat frame;
				Mat gray_frame;
				Mat graymask;
				Mat fgmask;
				video >> frame;
				if (frame.empty()) break;
				display_frame = Mat(frame, Rect(0, 0, 640, 360));
				frame = Mat(frame, Rect(240, 100, 360, 220));
				cvtColor(frame, gray_frame, CV_RGB2GRAY);
				equalizeHist(gray_frame, gray_frame);
				pKNN->apply(gray_frame, graymask);
			//	threshold(graymask, fgmask, 0, 255.0, CV_THRESH_OTSU);
		//		adaptiveThreshold(graymask,fgmask,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 0);
		

			//	morphologyEx(graymask, graymask, MORPH_OPEN, 14);
			//	morphologyEx(graymask, graymask, MORPH_CLOSE, 14);
				medianBlur(graymask, fgmask, 9);
				threshold(fgmask, fgmask, 0, 255.0, CV_THRESH_OTSU);
			//				adaptiveThreshold(fgmask,fgmask,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 13, 8);


							std::vector<std::vector<Point> > contours;
							findContours(fgmask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			//				Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			//				drawContours(mask, contours, -1, Scalar(255), -1);

			//				fgmask = mask;
							std::vector <Rect> detect;
							bool detect_motion = false;
							for (auto& t : contours) {
								double area = contourArea(t);
									if (area < 1000)continue;
								Rect r = boundingRect(t);
								if (r.height < 80) continue;

									if (area / r.area() < 0.2) continue;

								//if (r.width < 64 || r.height < 128) continue;
								detect.push_back(r);
							}

				for (auto t_it = trackers.begin(); t_it != trackers.end();) {
					t_it = (t_it->update()) ? std::next(t_it) : trackers.erase(t_it);
				}
#if 0
Mat out;
frame.copyTo(out);
				cv::Mat LabelImg;
				cv::Mat stats;
				cv::Mat centroids;
				int nLab = cv::connectedComponentsWithStats(fgmask, LabelImg, stats, centroids, 8);

				std::vector <Detection> detect;
				for (int i = 1; i < nLab; ++i) {
					int *param = stats.ptr<int>(i);
					Rect roi;
					int sz;

					roi.x = param[cv::ConnectedComponentsTypes::CC_STAT_LEFT];
					roi.y = param[cv::ConnectedComponentsTypes::CC_STAT_TOP];
					roi.height = param[cv::ConnectedComponentsTypes::CC_STAT_HEIGHT];
					roi.width = param[cv::ConnectedComponentsTypes::CC_STAT_WIDTH];
					sz = param[cv::ConnectedComponentsTypes::CC_STAT_AREA];
					if (roi.area() <1000) continue;
					if ((sz * 1.0) / roi.area() < 0.5) continue;

					cv::rectangle(out, roi, cv::Scalar(0, 0, 255), 2);

					bool merged = false;
					for (auto& d_rect : detect) {
						int s1 = d_rect.r.x;
						int e1 = d_rect.r.x + d_rect.r.width;
						int s2 = roi.x;
						int e2 = roi.x + roi.width;
						if ((s1 > s2 && s1 < e2) || (e1 > s2 && e1 < e2) ||
							(s2 > s1 && s2 < e1) || (e2 > s1 && e2 < e1)) {
							Rect mr = d_rect.r;
							int ms = d_rect.sz;

							mr |= roi;
							ms += sz;
							if ((ms * 1.0) / mr.area() > 0.3) {
								d_rect.r = mr;
								d_rect.sz = ms;
								merged = true;
								break;
							}
						}
					}
					if (!merged) {
						Detection d;
						d.r = roi;
						d.sz = sz;
						detect.push_back(d);
					}

/*
					std::string str;
					str += std::to_string(sz) + "\n";
					OutputDebugString(str.c_str());
					//					if (roi.width < 48 || roi.height < 96 || roi.width > 100 || roi.height > 200) continue;
					if (sz < 1000) continue;
					detect.push_back(roi);
					*/
				}
				std::string str;
				str += "bf: " + std::to_string(detect.size()) ;
		
				for (auto d_it = detect.begin(); d_it != detect.end();) {
cv::rectangle(out, d_it->r, cv::Scalar(0, 255, 0), 2);
					bool i = true;
					i &= (d_it->sz > 500);
					i &= ((d_it->sz * 1.0) / d_it->r.area() > 0.3);
					d_it = i ? std::next(d_it) : detect.erase(d_it);
				}

				str += " af: " + std::to_string(detect.size()) + "\n";

				OutputDebugString(str.c_str());
#endif

				for (auto& d_rect : detect) {
cv::rectangle(frame, d_rect, cv::Scalar(255, 0, 0), 2);
					bool exists = std::any_of(trackers.begin(), trackers.end(),
						[&d_rect](MyTracker& t) {return t.registerNewDetect(d_rect); });
					if (!exists) {
						MyTracker t(d_rect);
						Mat out,tmp;
						display_frame.copyTo(out);
						tmp = Mat(display_frame,Rect(240, 100, 360, 220));
						trackers.push_back(t);
	//					for (auto& t : trackers) t.draw(frame);
						t.draw(tmp);
						std::string path("c:\\nwcam\\161029\\out\\");
						path += std::to_string(t.get_id());
						path += ".jpg";
						imwrite(path, out);
					}
				}

				if (detect.size()) {
					for (auto& t : trackers) t.draw(frame);
					writer << display_frame;
				}
//				cv::rectangle(gray_frame, roi, cv::Scalar(0, 255, 0), 2);


//				double pos = video.get(CV_CAP_PROP_POS_MSEC);
	//			pos += 250;
		//		video.set(CV_CAP_PROP_POS_MSEC, pos);
					f++;
			} while (!(waitKey(1) >= 0));
		}
	}
	return 0;
}



