//
//  NSObject+Detector.h
//  CP
//
//  Created by Adriano Soares on 17/07/16.
//  Copyright © 2016 Adriano Soares. All rights reserved.
//

#import <UIKit/UIKit.h>

#import <opencv2/opencv.hpp>

#import <opencv2/imgcodecs/ios.h>

#import <dlib/image_processing.h>
#import <dlib/image_processing/frontal_face_detector.h>

#import <dlib/opencv.h>

@interface Detector: NSObject
+ (id)sharedManager;

- (BOOL) hasFace: (UIImage *)image;
- (std::vector<cv::Point2f>) detectLandmarks:(cv::Mat &)image andIsBlocking:(BOOL) isBlocking;

@end
