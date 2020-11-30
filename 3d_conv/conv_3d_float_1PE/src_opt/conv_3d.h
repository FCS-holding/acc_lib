
#ifndef _CONV_3D_H
#define _CONV_3D_H

#if IMAGE_SIZE == 100
#define image_width         100
#define image_height        100
#define image_depth         100
#define OUT_FACTOR          4
#endif
#if IMAGE_SIZE == 200
#define image_width         200
#define image_height        200
#define image_depth         200
#define OUT_FACTOR          8
#endif
#if IMAGE_SIZE == 400
#define image_width         400
#define image_height        400
#define image_depth         400
#define OUT_FACTOR          8
#endif
#if IMAGE_SIZE == 800
#define image_width         800
#define image_height        800
#define image_depth         800
#define OUT_FACTOR          8
#endif
#if IMAGE_SIZE == 1000
#define image_width         1000
#define image_height        1000
#define image_depth         1000
#define OUT_FACTOR          8
#endif


#define filter_w            24
#define filter_h            24
#define filter_d            24

#define DATA_IN_LENGTH      image_width * image_height * image_depth
#define FILTER_LENGTH       filter_d * filter_w * filter_h
#define DATA_OUT_LENGTH     (image_depth-filter_d+1) * (image_width-filter_w+1) * (image_height-filter_h+1)

#endif
