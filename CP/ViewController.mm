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

#import "CustomCamera.h"

using namespace cv;

using namespace dlib;

using namespace std;

@interface ViewController () {
    BOOL _cameraInitialized;
    CustomCamera *_videoCamera;
    
    frontal_face_detector detector;
    shape_predictor pose_model;
    
    __weak IBOutlet UIButton *photoButton;
    __weak IBOutlet UIButton *maskButton;
    __weak IBOutlet UIButton *changeButton;

    BOOL isImageLoaded;
    BOOL isDebug;
    
    
    cv::Mat photo;
    
    
    cv::Mat filter;
    cv::Mat filterMask;
    std::vector<Point2f> filterPoints;
    Subdiv2D filterSubdiv;
    
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
    
    isImageLoaded = NO;
    isDebug = YES;
    
    [self loadImage:[UIImage imageNamed:@"john-cena"]];
}

- (void)setupVideoCamera {
    if (_cameraInitialized) {
        // already initialized
        return;
    }
    
    UIView *camView = [[UIView alloc] initWithFrame:self.view.frame];
    [self.view addSubview:camView];

    
    _videoCamera = [[CustomCamera alloc] initWithParentView:camView];
    _videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    _videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;

    _videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    _videoCamera.grayscaleMode = NO;
    _videoCamera.rotateVideo = NO;
    _videoCamera.delegate = self;
    
    
    [self.view bringSubviewToFront:photoButton];
    [self.view bringSubviewToFront:maskButton];
    [self.view bringSubviewToFront:changeButton];

}

- (void)setupFaceDetector {
    detector = get_frontal_face_detector();
    
    NSString *path = [[NSBundle mainBundle] pathForResource:@"shape_predictor_68_face_landmarks" ofType:@"dat"];
    const char *filePath = [path cStringUsingEncoding:NSUTF8StringEncoding];
    deserialize(filePath) >> pose_model;
    
    [self loadImage: [UIImage imageNamed:@"john-cena"]];
    
}


- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

- (void) loadImage: (UIImage *) image {
    UIImageToMat(image, filter, false);
    cvtColor(filter, filter, COLOR_BGR2RGB);
    filterPoints.clear();
    Scalar white = Scalar(255, 255, 255);

    Mat gray;
    cvtColor(filter, gray, COLOR_BGR2GRAY);
    cv_image<uchar> dlib_img(gray);
    array2d<uchar> img;
    assign_image(img, dlib_img);
    
    std::vector<dlib::rectangle> dets = detector(img);
    for (int i = 0; i < dets.size(); i++) {
        cv::Rect bounds (0, 0, filter.size().width, filter.size().height);
        Subdiv2D subdiv(bounds);
        full_object_detection shape = pose_model(img, dets[i]);
        for (int j = 0; j < shape.num_parts(); j++) {
            cv::Point2f p(shape.part(j).x(), shape.part(j).y());
            filterPoints.push_back(p);
            if (bounds.contains(p))
                subdiv.insert(p);
            
            //if(isDebug)
                //cv::circle(filter, p, 1, red);
        }
        //cv::rectangle(filter, [self dlibRectangleToOpenCV:dets[i]], red, 1);
        filterMask = Mat::zeros(filter.rows, filter.cols, CV_8UC3);
        std::vector<Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);
        if (isDebug) {
            for(int j = 0; j < triangleList.size(); j++ )
            {
                cv::Point pt[3];
                Vec6f t = triangleList[j];
                pt[0] = cv::Point((int)t[0], (int)t[1]);
                pt[1] = cv::Point((int)t[2], (int)t[3]);
                pt[2] = cv::Point((int)t[4], (int)t[5]);
                if (bounds.contains(pt[0]) && bounds.contains(pt[1]) && bounds.contains(pt[2])) {
                    cv::fillConvexPoly(filterMask, pt, 3, Scalar(255,255,255));
                }
            }
        }

        //std::vector<Point2f> hull;
        //convexHull(filterPoints, hull, false, true);
        
        
        //cv::fillConvexPoly(filterMask, hull, Scalar(255,255,255));
    }
    isImageLoaded = YES;

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

- (void)thisImage:(UIImage *)image hasBeenSavedInPhotoAlbumWithError:(NSError *)error usingContextInfo:(void*)ctxInfo {
    NSString *msg = @"";
    if (error) {
        msg = error.description;
        UIAlertController *alertController = [UIAlertController alertControllerWithTitle:@"Photo Saving" message:msg preferredStyle:UIAlertControllerStyleAlert];
        
        UIAlertAction* ok = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil];
        [alertController addAction:ok];
        
        [self presentViewController:alertController animated:YES completion:nil];
    } else {
        msg = @"should work!";
    }
}

- (IBAction)photoButtonClick:(id)sender {
    UIImage *image = MatToUIImage(photo);
    UIImageWriteToSavedPhotosAlbum(image,
                                   self, // send the message to 'self' when calling the callback
                                   @selector(thisImage:hasBeenSavedInPhotoAlbumWithError:usingContextInfo:), // the selector to tell the method to call on completion
                                   NULL);
    
}

- (IBAction)changeButtonClick:(id)sender {
    [self->_videoCamera switchCameras];
}

- (IBAction)maskButtonClick:(id)sender {
    
    
}

#pragma mark - CvVideoCameraDelegate

- (void)processImage:(cv::Mat &)image {
    
    Mat gray;
    Scalar green = Scalar(0, 255, 0);
    Scalar red = Scalar(0, 0, 255);
    cvtColor(image, gray, COLOR_BGR2GRAY);
    cv_image<uchar> dlib_img(gray);
    array2d<uchar> img;
    array2d<uchar> down;
    assign_image(img, dlib_img);
    /*
    pyramid_down<2> pyr;
    pyr(img, down);
    */
    
    std::vector<dlib::rectangle> dets = detector(img);
    for (int i = 0; i < dets.size(); i++) {
        std::vector<cv::Point2f> landmarks;
        full_object_detection shape = pose_model(img, dets[i]);
        cv::Rect bounds (0, 0, image.size().width, image.size().height);
        Subdiv2D subdiv(bounds);
        for (int j = 0; j < shape.num_parts(); j++) {
            cv::Point2f p (shape.part(j).x(), shape.part(j).y());
            landmarks.push_back(p);
            
            if (bounds.contains(p))
                subdiv.insert(p);
            
            if(isDebug)
                cv::circle(image, p, 1, green);
        }
        
        
        std::vector<Vec6f> triangleList;
        //subdiv.getTriangleList(triangleList);
        if (isDebug) {
            for(int j = 0; j < triangleList.size(); j++ )
            {
                std::vector<cv::Point2f> pt(3);
                Vec6f t = triangleList[j];
                pt[0] = Point2f(t[0], t[1]);
                pt[1] = Point2f(t[2], t[3]);
                pt[2] = Point2f(t[4], t[5]);
                if (bounds.contains(pt[0]) && bounds.contains(pt[1]) && bounds.contains(pt[2])) {
                    cv::line(image, pt[0], pt[1], red, 1, CV_AA, 0);
                    cv::line(image, pt[1], pt[2], red, 1, CV_AA, 0);
                    cv::line(image, pt[2], pt[0], red, 1, CV_AA, 0);
                }
            }
            cv::rectangle(image, [self dlibRectangleToOpenCV:dets[i]], green, 1);
        
        }
        if (isImageLoaded) {
            std::vector<Point2f> v1;
            std::vector<Point2f> v2;
            
             
            v1.push_back(landmarks[36]);
            v1.push_back(landmarks[45]);
            v1.push_back(landmarks[8]);
            
            v2.push_back(filterPoints[36]);
            v2.push_back(filterPoints[45]);
            v2.push_back(filterPoints[8]);
            
            Mat warpMat = getAffineTransform(v2, v1);
            
            Mat warpedFilter = Mat::zeros(image.size().height, image.size().width, CV_8UC3);
            Mat warpedMask = Mat::zeros(image.size().height, image.size().width, CV_8UC3);
            
            Mat out_ = image.clone();// = Mat::zeros(image.size().height, image.size().width, CV_8UC3);;
            
            warpAffine(filter, warpedFilter, warpMat, warpedFilter.size());
            warpAffine(filterMask, warpedMask, warpMat, warpedMask.size());
            
            int i = warpedMask.type();
            
            if (warpedFilter.data) {
                //out_ = warpedMask;
                cv::seamlessClone(warpedFilter, image, warpedMask, cv::Point(image.rows/2,image.cols/2), out_, NORMAL_CLONE);
            }

            image = out_;
        }
    }
    photo = image;
}

@end
