//
//  CustomCamera.m
//  CP
//
//  Created by Adriano Soares on 17/05/16.
//  Copyright © 2016 Adriano Soares. All rights reserved.
//

#import "CustomCamera.h"

@implementation CustomCamera

- (void)updateOrientation;
{
    // nop
}
- (void)layoutPreviewLayer;
{
    if (self.parentView != nil) {
        CALayer* layer = self->customPreviewLayer;
        CGRect bounds = self->customPreviewLayer.bounds;
        layer.position = CGPointMake(self.parentView.frame.size.width/2., self.parentView.frame.size.height/2.);
        layer.bounds = bounds;
    }
}
@end
