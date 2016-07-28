//
//  NSObject+Detector.m
//  CP
//
//  Created by Adriano Soares on 17/07/16.
//  Copyright © 2016 Adriano Soares. All rights reserved.
//

#import "Detector.h"

using namespace dlib;
using namespace cv;

@implementation Detector {
    NSLock* detectorLock;

    frontal_face_detector detector;
    shape_predictor pose_model;
}

+ (id) sharedManager {
    static Detector *sharedMyManager = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedMyManager = [[self alloc] init];
    });
    return sharedMyManager;
}

- (id) init {
    if (self = [super init]) {
        [self setupFaceDetector];
    }
    return self;
}



- (BOOL) hasFace: (UIImage *)image {
    Mat f;
    UIImageToMat(image, f);
    Mat gray;
    cvtColor(f, gray, COLOR_RGBA2GRAY);
    cv_image<uchar> cv_img(gray);
    array2d<uchar> img;
    array2d<uchar> down;
    assign_image(img, cv_img);
    pyramid_down<2> pyr;
    pyr(img, down);
    std::vector<dlib::rectangle> dets;
    [detectorLock lock];
    @try {
        dets = [self detectFaces:down];
    } @catch (NSException *exception) {

    } @finally {
        [detectorLock unlock];
        return (dets.size() >= 1);
    }
}

- (std::vector <dlib::rectangle>) detectFaces:(array2d<uchar> &) image {
    std::vector<dlib::rectangle> dets;
    dets = detector(image);
    return dets;
}

- (std::vector<cv::Point2f>) detectLandmarks:(cv::Mat &)image andIsBlocking:(BOOL) isBlocking {
    std::vector<dlib::rectangle> dets;
    std::vector<cv::Point2f> landmarks;

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    cv_image<uchar> cv_img(gray);
    array2d<uchar> img;
    assign_image(img, cv_img);
    /*
    if (isBlocking)  {
        [detectorLock lock];
    } else {
        if (![detectorLock tryLock]) return landmarks;
    }
    */
    [detectorLock lock];
    @try {
        dets = [self detectFaces:img];
        if (dets.size() < 1) return landmarks;
        dlib::rectangle biggest_face = dets[1];
        for (int i = 0; i < dets.size(); i++) {
            if (dets[i].area() > biggest_face.area()) biggest_face = dets[i];
        }
        full_object_detection shape = pose_model(img, biggest_face);
        for (int i = 0; i < shape.num_parts(); i++) {
            cv::Point2f p (shape.part(i).x(), shape.part(i).y());
            landmarks.push_back(p);
        }


    } @catch (NSException *exception) {

    } @finally {
        [detectorLock unlock];
        return landmarks;
    }

}

- (void) setupFaceDetector {
    detectorLock = [[NSLock alloc] init];
    detector = get_frontal_face_detector();

    NSString *path = [[NSBundle mainBundle] pathForResource:@"shape_predictor_68_face_landmarks" ofType:@"dat"];
    const char *filePath = [path cStringUsingEncoding:NSUTF8StringEncoding];
    deserialize(filePath) >> pose_model;
}

- (void)dealloc {

}

@end
