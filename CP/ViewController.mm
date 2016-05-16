//
//  ViewController.mm
//  CP
//
//  Created by Adriano Soares on 12/05/16.
//  Copyright © 2016 Adriano Soares. All rights reserved.
//

#import "ViewController.h"

#import <dlib/image_processing.h>
#import <dlib/image_processing/frontal_face_detector.h>

#import <dlib/opencv.h>

using namespace cv;
using namespace std;
using namespace dlib;


@interface ViewController () {
    BOOL _cameraInitialized;
    CvVideoCamera *_videoCamera;
    CvPhotoCamera *_photoCamera;
    CascadeClassifier _faceDetector;
    
    frontal_face_detector detector;
    shape_predictor pose_model;
}
@end

@implementation ViewController

- (cv::Rect) dlibRectangleToOpenCV:(dlib::rectangle) r
{
    return cv::Rect(cv::Point2i((int)r.left(), (int)r.top()), cv::Point2i((int)r.right() + 1, (int)r.bottom() + 1));
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.

    [self setupVideoCamera];
    [self setupFaceDetector];
}

- (void)setupVideoCamera {
    if (_cameraInitialized) {
        // already initialized
        return;
    }
    
    _videoCamera = [[CvVideoCamera alloc] initWithParentView:self.view];
    _videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    _videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPresetMedium;
    _videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    _videoCamera.grayscaleMode = NO;
    _videoCamera.rotateVideo = NO;
    _videoCamera.delegate = self;
}

- (void)setupFaceDetector {
    detector = get_frontal_face_detector();
    
    NSString *path = [[NSBundle mainBundle] pathForResource:@"shape_predictor_68_face_landmarks" ofType:@"dat"];
    const char *filePath = [path cStringUsingEncoding:NSUTF8StringEncoding];
    deserialize(filePath) >> pose_model;

}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    [_videoCamera start];
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [_videoCamera stop];
}

#pragma mark - CvVideoCameraDelegate

- (void)processImage:(cv::Mat &)image {
    
    Mat gray;
    Scalar green = Scalar(0, 255, 0);
    Scalar red = Scalar(0, 0, 255);
    cvtColor(image, gray, COLOR_BGR2GRAY);
    cv_image<uchar> dlib_img(gray);
    array2d<uchar> img;
    assign_image(img, dlib_img);
    
    std::vector<dlib::rectangle> dets = detector(img);
    for (int i = 0; i < dets.size(); i++) {
        full_object_detection shape = pose_model(img, dets[i]);
        for (int j = 0; j < shape.num_parts(); j++) {
            cv::Point p((int)shape.part(j).x(), (int)shape.part(j).y());
            cv::circle(image, p, 2, red);
        }
        cv::rectangle(image, [self dlibRectangleToOpenCV:dets[i]], green, 1);
    }
    /*
    _faceDetector.detectMultiScale(gray, faces, 1.15, 2, 0, cv::Size(50, 50));
    for (int i = 0; i < faces.size(); i++) {
        cv::rectangle(image, faces[i], color, 1);
    }
    */
}

@end
