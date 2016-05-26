//
//  CustomCamera.h
//  CP
//
//  Created by Adriano Soares on 17/05/16.
//  Copyright © 2016 Adriano Soares. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <opencv2/opencv.hpp>
#import <opencv2/videoio/cap_ios.h>

@interface CustomCamera : CvVideoCamera

- (void)updateOrientation;
- (void)layoutPreviewLayer;

@end
