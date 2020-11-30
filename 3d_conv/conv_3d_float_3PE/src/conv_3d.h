
#ifndef _CONV_3D_H
#define _CONV_3D_H

#if IMAGE_SIZE == 100
#define IMAGE_SIZE 100
#endif
#if IMAGE_SIZE == 200
#define IMAGE_SIZE 200
#endif
#if IMAGE_SIZE == 400
#define IMAGE_SIZE 400
#endif
#if IMAGE_SIZE == 800
#define IMAGE_SIZE 800
#endif
#if IMAGE_SIZE == 1000
#define IMAGE_SIZE 1000
#endif

#define FILTER_SIZE 24
#define PE 3

#define OUT_IMAGE_SIZE (IMAGE_SIZE-FILTER_SIZE+1)
#define STEP ((IMAGE_SIZE-FILTER_SIZE)/PE+1)

#define START_0 (0*STEP)
#define END_0 (1*STEP+FILTER_SIZE-2)
#define IN_LENGTH_0 (END_0 - START_0 + 1)
#define START_SIZE_0 START_0*IMAGE_SIZE*IMAGE_SIZE
#define IN_SIZE_0 (END_0-START_0+1)*IMAGE_SIZE*IMAGE_SIZE 
#define OUT_START_0 (0*STEP)
#define OUT_END_0 (1*STEP)
#define OUT_DEPTH_0 (OUT_END_0 - OUT_START_0) 
#define OUT_SIZE_0 (OUT_DEPTH_0 * OUT_IMAGE_SIZE * OUT_IMAGE_SIZE)  

#if PE>=2
#define START_1 (1*STEP)
#define END_1 (2*STEP+FILTER_SIZE-2)
#define IN_LENGTH_1 (END_1 - START_1 + 1)
#define START_SIZE_1 START_1*IMAGE_SIZE*IMAGE_SIZE
#define IN_SIZE_1 (END_1-START_1+1)*IMAGE_SIZE*IMAGE_SIZE
#define OUT_START_1 (1*STEP)
#define OUT_END_1 (2*STEP)
#define OUT_DEPTH_1 (OUT_END_1 - OUT_START_1)
#define OUT_SIZE_1 (OUT_DEPTH_1 * OUT_IMAGE_SIZE * OUT_IMAGE_SIZE)
#endif

#if PE>=3
#define START_2 (2*STEP)
#define END_2 (3*STEP+FILTER_SIZE-2)
#define IN_LENGTH_2 (END_2 - START_2 + 1)
#define START_SIZE_2 START_2*IMAGE_SIZE*IMAGE_SIZE
#define IN_SIZE_2 (END_2-START_2+1)*IMAGE_SIZE*IMAGE_SIZE
#define OUT_START_2 (2*STEP)
#define OUT_END_2 (3*STEP)
#define OUT_DEPTH_2 (OUT_END_2 - OUT_START_2)
#define OUT_SIZE_2 (OUT_DEPTH_2 * OUT_IMAGE_SIZE * OUT_IMAGE_SIZE)
#endif

#define image_width     IMAGE_SIZE
#define image_height    IMAGE_SIZE
#define image_depth     IMAGE_SIZE

#define filter_w        FILTER_SIZE
#define filter_h        FILTER_SIZE
#define filter_d        FILTER_SIZE

#if PE == 1
#define DATA_IN_LENGTH      image_width * (END_0+1) * image_depth
#endif

#if PE == 2
#define DATA_IN_LENGTH      image_width * (END_1+1) * image_depth
#endif

#if PE == 3
#define DATA_IN_LENGTH      image_width * (END_2+1) * image_depth
#endif


#define FILTER_LENGTH       filter_d * filter_w * filter_h
#define DATA_OUT_LENGTH     (image_depth-filter_d+1) * (image_width-filter_w+1) * (image_height-filter_h+1)

#define KERNEL_IN_LENGTH    image_width * (STEP+FILTER_SIZE-1) * image_depth
#define KERNEL_OUT_LENGTH   (image_depth-filter_d+1) * (image_width-filter_w+1) * STEP

#endif
