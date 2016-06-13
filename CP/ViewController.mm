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

#import "MaskHelper.h"
#import "CustomCamera.h"

#include <Photos/PHPhotoLibrary.h>
#include <Photos/PHFetchOptions.h>
#include <Photos/PHFetchResult.h>
#include <Photos/PHAsset.h>
#include <Photos/PHImageManager.h>
#include <Photos/PHCollection.h>

using namespace cv;
using namespace dlib;
using namespace std;

@interface ViewController () {
    BOOL _cameraInitialized;
    CustomCamera *_videoCamera;
    CvPhotoCamera *_photoCamera;
    
    frontal_face_detector detector;
    shape_predictor pose_model;
    
    __weak IBOutlet UIButton *photoButton;
    __weak IBOutlet UIButton *maskButton;
    __weak IBOutlet UIButton *changeButton;
    __weak IBOutlet UIButton *configButton;

    BOOL isImageLoaded;
    BOOL isDebug;
    BOOL isSeamless;
    
    NSLock* detectorLock;
    
    cv::Mat photo;
    
    
    cv::Mat filter;
    cv::Mat filterMask;
    std::vector<Point2f> filterPoints;
    Subdiv2D filterSubdiv;
    std::vector<cv::Point*> filterTriangulation;
    
    UICollectionView *photoGallery;
    UIAlertController *alert;
    
    NSMutableArray *photosArray;
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
    isDebug = NO;
    isSeamless = YES;
    
    UICollectionViewFlowLayout *layout=[[UICollectionViewFlowLayout alloc] init];
    [layout setItemSize: CGSizeMake(100, 100)];
    [layout setScrollDirection: UICollectionViewScrollDirectionVertical];
    
    photosArray = [[NSMutableArray alloc] init];
    
    photoGallery = [[UICollectionView alloc] initWithFrame:CGRectMake(0, 0, 0, 0) collectionViewLayout:layout];
    
    [photoGallery registerClass:[UICollectionViewCell class] forCellWithReuseIdentifier:@"cellIdentifier"];
    
    photoGallery.dataSource = self;
    photoGallery.delegate   = self;
    photoGallery.backgroundColor = [[UIColor whiteColor] colorWithAlphaComponent:0.1];
    
    [SVProgressHUD setDefaultStyle:SVProgressHUDStyleCustom];
    [SVProgressHUD setBackgroundColor:[[UIColor blackColor] colorWithAlphaComponent:0.5]];
    [SVProgressHUD setForegroundColor:[[UIColor whiteColor] colorWithAlphaComponent:1.0]];
    [SVProgressHUD setDefaultMaskType:SVProgressHUDMaskTypeBlack];
    
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
    _videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPresetLow;
    _videoCamera.defaultFPS = 30;
    
    
    _videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    _videoCamera.grayscaleMode = NO;
    _videoCamera.rotateVideo = NO;
    _videoCamera.delegate = self;
    
    [self.view bringSubviewToFront:photoButton];
    [self.view bringSubviewToFront:maskButton];
    [self.view bringSubviewToFront:changeButton];
    [self.view bringSubviewToFront:configButton];

}

- (void)setupFaceDetector {
    detectorLock = [[NSLock alloc] init];
    detector = get_frontal_face_detector();
    
    NSString *path = [[NSBundle mainBundle] pathForResource:@"shape_predictor_68_face_landmarks" ofType:@"dat"];
    const char *filePath = [path cStringUsingEncoding:NSUTF8StringEncoding];
    deserialize(filePath) >> pose_model;
}


- (void) loadImage: (UIImage *) image {
    UIImageToMat(image, filter);
    cv::Mat out_ (filter.rows, filter.cols, CV_8UC3);
    cvtColor(filter, out_, COLOR_RGBA2BGR);
    out_.copyTo(filter);
    filterPoints.clear();

    Mat gray;
    cvtColor(filter, gray, COLOR_BGR2GRAY);
    cv_image<uchar> dlib_img(gray);
    array2d<uchar> img;
    assign_image(img, dlib_img);
    [detectorLock lock];
    std::vector<dlib::rectangle> dets = detector(img);
    [detectorLock unlock];
    for (int i = 0; i < dets.size(); i++) {
        full_object_detection shape = pose_model(img, dets[i]);
        for (int j = 0; j < shape.num_parts(); j++) {
            cv::Point2f p(shape.part(j).x(), shape.part(j).y());
            filterPoints.push_back(p);
        }
        filterMask = Mat::zeros(filter.rows, filter.cols, CV_8UC3);
        NSArray *triangulation = [MaskHelper triangulation];
        for(int j = 0; j < [triangulation count]; j++ )
        {
            cv::Point pt[3];
            NSArray *t = triangulation[j];
            pt[0] = cv::Point(filterPoints[[t[0] integerValue]]);
            pt[1] = cv::Point(filterPoints[[t[1] integerValue]]);
            pt[2] = cv::Point(filterPoints[[t[2] integerValue]]);
            cv::fillConvexPoly(filterMask, pt, 3, Scalar(255,255,255));
        }
        cvtColor(filterMask, filterMask, COLOR_BGR2GRAY);

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
        UIWindow *window = [[UIApplication sharedApplication] keyWindow];
        UIView *flashView = [[UIView alloc] initWithFrame:window.bounds];
        flashView.backgroundColor = [UIColor whiteColor];
        flashView.alpha = 1.0f;
        [window addSubview:flashView];
        AudioServicesPlaySystemSoundWithCompletion(1108, nil);
        // Fade it out and remove after animation.
        [UIView animateWithDuration:0.3f animations:^{
            flashView.alpha = 0.0f;
        } completion:^(BOOL finished) {
            [flashView removeFromSuperview];
        }];
    }
}

- (IBAction)photoButtonClick:(id)sender {
    Mat out_;
    cv::resize(photo, out_, cv::Size(), 2.0, 2.0);
    //cvtColor(photo, out_, COLOR_BGR2RGBA);
    UIImage *image = MatToUIImage(out_);
    UIImageWriteToSavedPhotosAlbum(image,
        self, // send the message to 'self' when calling the callback
        @selector(thisImage:hasBeenSavedInPhotoAlbumWithError:usingContextInfo:), // the selector to tell the method to call on completion
        NULL);
    
}

- (IBAction)changeButtonClick:(id)sender {
    [self->_videoCamera switchCameras];
}
- (IBAction)configButtonClick:(id)sender {
    UIAlertController *configAlert = [UIAlertController alertControllerWithTitle:@""
                                                message:@""
                                         preferredStyle:UIAlertControllerStyleActionSheet];
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel handler:^(UIAlertAction * action) { }];
    
    UIAlertAction *debugAction = [UIAlertAction actionWithTitle:@"Toggle Debug"
                                                           style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
                                                               isDebug = !isDebug;
                                                           }];
    
    UIAlertAction *seamlessAction = [UIAlertAction actionWithTitle:@"Toggle Seamless Clone"
                                                          style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
                                                              isSeamless = !isSeamless;
                                                          }];
    [configAlert addAction:debugAction];
    [configAlert addAction:seamlessAction];
    [configAlert addAction:cancelAction];

    [self presentViewController:configAlert animated:YES completion:nil];
    
}

- (IBAction)maskButtonClick:(id)sender {
    isImageLoaded = NO;
    [maskButton setEnabled:NO];
    alert = [UIAlertController alertControllerWithTitle:@"\n\n\n\n\n"
                                                message:@""
                                         preferredStyle:UIAlertControllerStyleActionSheet];
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel"
                                                           style:UIAlertActionStyleCancel handler:^(UIAlertAction * action) { }];
    
    CGFloat margin = 8.0;
    [alert addAction:cancelAction];
    
    alert.view.contentMode = UIViewContentModeScaleToFill;
    
    [photoGallery setFrame:CGRectMake(margin, margin, alert.view.bounds.size.width - margin * 4.0F, 100.0F)];
    
    [alert.self.view addSubview:photoGallery];

    [SVProgressHUD show];
    dispatch_async(dispatch_get_global_queue (DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(void){
        photosArray = [self albumImages];
        dispatch_async(dispatch_get_main_queue(), ^(void){
            [photoGallery reloadData];
            [self presentViewController:alert animated:YES completion:nil];
            [maskButton setEnabled:YES];
            [SVProgressHUD dismiss];
        });
    });
}

- (NSMutableArray *) albumImages {
    NSMutableArray *a = [[NSMutableArray alloc] init];
    PHFetchResult *smartAlbums = [PHAssetCollection fetchMomentsWithOptions:nil];//[PHAssetCollection fetchAssetCollectionsWithType:PHAssetCollectionTypeSmartAlbum subtype:PHAssetCollectionSubtypeAlbumRegular options:nil];
    
    //set up fetch options, mediaType is image.
    PHFetchOptions *options = [[PHFetchOptions alloc] init];
    options.sortDescriptors = @[[NSSortDescriptor sortDescriptorWithKey:@"creationDate" ascending:NO]];
    options.predicate = [NSPredicate predicateWithFormat:@"mediaType = %d",PHAssetMediaTypeImage];
    
    for (NSInteger i =0; i < smartAlbums.count; i++) {
        PHAssetCollection *assetCollection = smartAlbums[i];
        PHFetchResult *assetsFetchResult = [PHAsset fetchAssetsInAssetCollection:assetCollection options:options];
        
        if (assetsFetchResult.count > 0) {
            for (PHAsset *asset in assetsFetchResult) {
                PHImageManager *manager = [PHImageManager defaultManager];
                PHImageRequestOptions *assetOptions = [[PHImageRequestOptions alloc] init];
                assetOptions.synchronous = true;                
                [manager requestImageForAsset:asset targetSize:PHImageManagerMaximumSize
                                                  contentMode:PHImageContentModeAspectFit
                                                      options:assetOptions
                                                resultHandler:^(UIImage *image, NSDictionary *info) {
                                                    Mat f;
                                                    UIImageToMat(image, f);
                                                    Mat gray;
                                                    cvtColor(f, gray, COLOR_RGBA2GRAY);
                                                    cv_image<uchar> dlib_img(gray);
                                                    array2d<uchar> img;
                                                    array2d<uchar> down;
                                                    assign_image(img, dlib_img);
                                                    pyramid_down<2> pyr;
                                                    
                                                    pyr(img, down);
                                                    std::vector<dlib::rectangle> dets;
                                                    @try {
                                                        [detectorLock lock];
                                                        dets = detector(down);
                                                        if (dets.size() > 0 ) {
                                                            [a addObject:image];
                                                        }
                                                    } @catch (NSException *e) {
                                                        
                                                    } @finally {
                                                        [detectorLock unlock];
                                                    }

                                                }];
            }
        }
    }
    return [a copy];
}

#pragma mark - CvVideoCameraDelegate

- (void)processImage:(cv::Mat &)image {
    cvtColor(image, image, COLOR_BGRA2RGB);
    try {
        Mat gray;
        Scalar green = Scalar(0, 255, 0);
        cvtColor(image, gray, COLOR_BGR2GRAY);
        cv_image<uchar> dlib_img(gray);
        array2d<uchar> img;
        array2d<uchar> down;
        assign_image(img, dlib_img);
        /*
         pyramid_down<2> pyr;
         pyr(img, down);
         */
        std::vector<dlib::rectangle> dets;
        if ([detectorLock tryLock]){
            dets = detector(img);
            [detectorLock unlock];
        }
        for (int i = 0; i < dets.size(); i++) {
            std::vector<cv::Point2f> landmarks;
            full_object_detection shape = pose_model(img, dets[i]);
            for (int j = 0; j < shape.num_parts(); j++) {
                cv::Point2f p (shape.part(j).x(), shape.part(j).y());
                landmarks.push_back(p);
            }
            std::vector<Vec6f> triangleList;
            NSArray *triangulation = [MaskHelper triangulation];
            for(int j = 0; j < [triangulation count]; j++ )
            {
                cv::Point pt[3];
                NSArray *t = triangulation[j];
                pt[0] = cv::Point(landmarks[[t[0] integerValue]]);
                pt[1] = cv::Point(landmarks[[t[1] integerValue]]);
                pt[2] = cv::Point(landmarks[[t[2] integerValue]]);
                if (isDebug) {
                    cv::line(image, pt[0], pt[1], green, 1, CV_AA, 0);
                    cv::line(image, pt[1], pt[2], green, 1, CV_AA, 0);
                    cv::line(image, pt[2], pt[0], green, 1, CV_AA, 0);
                }

            }
            if (isDebug) cv::rectangle(image, [self dlibRectangleToOpenCV:dets[i]], green, 1);
            if (isImageLoaded) {
                
                Mat warpedFilter = Mat::zeros(image.size().height, image.size().width, CV_8UC3);
                Mat warpedMask   = Mat::zeros(image.size().height, image.size().width, CV_8UC1);
                
                
                
                for(int j = 0; j < [triangulation count]; j++) {
                    std::vector<cv::Point2f> v1;
                    std::vector<cv::Point2f> v2;
                    
                    NSArray *t = triangulation[j];
                    
                    int counter = 0;
                    for (int k = 48; k <= 67 && counter < 3; k++ ) {
                        if ([t containsObject:[NSNumber numberWithInteger:k]]) {
                            counter++;
                        }
                        
                    }
                    if (counter == 3) continue;
                    
                    int i1, i2, i3;
                    i1 = (int)[t[0] integerValue];
                    i2 = (int)[t[1] integerValue];
                    i3 = (int)[t[2] integerValue];
                    
                    v1.push_back(landmarks[i1]);
                    v1.push_back(landmarks[i2]);
                    v1.push_back(landmarks[i3]);
                    
                    v2.push_back(filterPoints[i1]);
                    v2.push_back(filterPoints[i2]);
                    v2.push_back(filterPoints[i3]);
                    
                    Mat newMask = Mat::zeros(filter.rows, filter.cols, CV_8UC1);
                    Mat outMask = Mat::zeros(image .rows, image. cols, CV_8UC1);
                    
                    Mat warpMat = getAffineTransform(v2, v1);
                    

                    cv::Point pt[3];
                    pt[0] = cv::Point(landmarks[[t[0] integerValue]]);
                    pt[1] = cv::Point(landmarks[[t[1] integerValue]]);
                    pt[2] = cv::Point(landmarks[[t[2] integerValue]]);
                    cv::fillConvexPoly(warpedMask, pt, 3, Scalar(255));
                    cv::fillConvexPoly(outMask   , pt, 3, Scalar(255));

                    Mat outFilter = Mat::zeros(image .rows, image. cols, CV_8UC3);
                    warpAffine(filter, outFilter, warpMat, outFilter.size());
                    outFilter.copyTo(warpedFilter, outMask);
                }
                Mat out_ = image.clone();
                if (warpedFilter.data) {
                    cvtColor(warpedFilter, warpedFilter, COLOR_RGBA2BGR);
                    if (isSeamless) {
                        cv::Rect r ( cv::boundingRect(landmarks)); //[self dlibRectangleToOpenCV:dets[i]]);
                        cv::Point2f center = cv::Point2f(r.x+r.width/2, r.y + r.height/2);
                        
                        cv::seamlessClone(warpedFilter, image, warpedMask, center, out_, NORMAL_CLONE);
                    } else {
                        warpedFilter.copyTo(out_, warpedMask);
                    }
                }
                if (out_.data)
                    image = out_;
            }
        }
    } catch (Exception e) {
        //image = Mat::ones(image .rows, image. cols, CV_8UC3);
    }

    photo = image.clone();
}

#pragma mark - UICollectionViewDelegate
- (NSInteger)collectionView:(UICollectionView *)collectionView numberOfItemsInSection:(NSInteger)section {
    return [photosArray count];
}

// The cell that is returned must be retrieved from a call to -dequeueReusableCellWithReuseIdentifier:forIndexPath:
- (UICollectionViewCell *)collectionView:(UICollectionView *)collectionView cellForItemAtIndexPath:(NSIndexPath *)indexPath {
    UICollectionViewCell *cell = [collectionView dequeueReusableCellWithReuseIdentifier:@"cellIdentifier" forIndexPath:indexPath];
    UIImageView *view = [[UIImageView alloc] initWithImage:photosArray[indexPath.row]];
    view.contentMode = UIViewContentModeScaleAspectFill;
    view.clipsToBounds = YES;
    
    cell.backgroundView =view;
    cell.contentMode = UIViewContentModeScaleAspectFit;
    
    return cell;
}

- (void)collectionView:(UICollectionView *)collectionView didSelectItemAtIndexPath:(NSIndexPath *)indexPath {
    [self loadImage:photosArray[indexPath.row]];

}

#pragma mark - UICollectionViewDataSource

@end
