
typedef struct {
    int x;
    int y;
} point;

typedef struct {
    point point1;
    point point2;
} test;

void atomic_add_float_global(__global float* p, float val) // stackoverflow
{
    asm volatile(
        "atom.global.add.f32 _, [%0], %1;"  // "_" or omit output
        :
        : "l"(p), "f"(val)
        : "memory"
    );
}



__kernel void gradient_convolution(__global uchar* image_data,
                                   __global float* Jx,
                                   __global float* Jy,
                                   __global float* Jxy,
                                   const int width) {

    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    const size_t idx_m1 = (i - 1) * width + j;
    const size_t idx = i * width + j;
    const size_t idx_p1 = (i + 1) * width + j;

    // Jx[idx] = 69.0;  colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);


    float sumx[3] = {0};
    float sumy[3] = {0};

    // ptr_src1 = image_data.ptr<uchar>(idx - 1);
    // ptr_src2 = image_data.ptr<uchar>(idx);
    // ptr_src3 = image_data.ptr<uchar>(idx + 1);


    for (short k = -1; k <= 1; k++) {
        sumx[k + 1] = (float)(image_data[idx_m1 + k]) - (float)(image_data[idx_p1 + k]); // // [1, 0, -1] T
        sumy[k + 1] = (float)(image_data[idx_m1 + k]) + 2.0f * (float)(image_data[idx + k]) + (float)(image_data[idx_p1 + k]); // [1, 2, 1] T

    }

    Jx[idx] = sumx[0] + 2.0f * sumx[1] + sumx[2]; // [1, 2, 1]
    Jy[idx]  = sumy[0] - sumy[2]; // [1, 0, -1]
    Jxy[idx]  = sumx[0] - sumx[2]; // [1, 0, -1]

}


__kernel void shitomasi_response(__global float* R_renponse,
                                 __global const float* Jx,
                                 __global const float* Jy,
                                 __global const float* Jxy,
                                 const int width,
                                 const float RESPONSE_THRESHOLD) {

    float jx2 = 0;
    float jy2 = 0;
    float sumjxy = 0;
    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    const size_t idx = i * width + j;


    for (int m = -2; m <= 2; m++) {
        for (int n = -2; n <= 2; n++) {
            const size_t neighbor_idx = (i + m) * width + (j + n);
            const float jx = Jx[neighbor_idx];
            const float jy = Jy[neighbor_idx];
            const float jxy = Jxy[neighbor_idx];

            sumjxy += jxy;
            jx2 += jx * jx;
            jy2 += jy * jy;
        }
    }

    const float det = (jx2 * jy2) - (sumjxy * sumjxy);
    const float trace = jx2 + jy2;
    const float R = (trace / 2) - (0.5 * sqrt(trace * trace - 4 * det));

    R_renponse[idx] = R > RESPONSE_THRESHOLD ? R : 0;
//    R_renponse[idx] = 1000.0f * sin((float)(i + j) / 10.0f);
//	R_renponse[idx] = R;
    // R_array[i][j] = R;
    // max_R = std::max(R, max_R);

}

__kernel void compute_orientation(__global const uchar* image_data,
                                  __global const test* test_cases,
                                  __global float* O_x,
                                  __global float* O_y,
                                  __global point* key_points,
                                  const int key_point_index,
                                  const int width) {

//	size_t i = get_global_id(1);
//    size_t j = get_global_id(0); i = point1 <===> j = point2

	const size_t id = get_global_id(0);

	const float point1_intensity = (float)(image_data[(test_cases[id].point1.y + key_points[key_point_index].y) * width + test_cases[id].point1.x + key_points[key_point_index].x]);

    const float intensity_change = point1_intensity - (float)(image_data[(test_cases[id].point2.y + key_points[key_point_index].y) * width + test_cases[id].point2.x  + key_points[key_point_index].x]);


	// norm of 2 vectors
     const double norm = sqrt(pow(test_cases[id].point1.y + key_points[key_point_index].y - test_cases[id].point2.y + key_points[key_point_index].y, 2.0) +
         pow(test_cases[id].point1.x + key_points[key_point_index].x - test_cases[id].point2.x + key_points[key_point_index].x, 2.0) );


    atomic_add_float_global(O_x, intensity_change * (test_cases[id].point1.x - test_cases[id].point2.x) / norm);
    atomic_add_float_global(O_y, intensity_change * (test_cases[id].point1.y - test_cases[id].point2.y) / norm);


}

__kernel void merge_orientation_tasks(__global float* O_x,
                                      __global float* O_y,
                                      __global float* rotation_matrix) {

//  	const float o_x = atomic_load(O_x);
//	const float o_y = atomic_load(O_y);


    if (isnan(*O_x) || isnan(*O_y)) { // a check for isinf() might be useful here
        const float angle = 0.0f;
    }
    const float angle = atan2(*O_y, *O_x);

    rotation_matrix[0] = cos(angle);
    rotation_matrix[1] = -1.0f * sin(angle);
    rotation_matrix[2] = sin(angle);
    rotation_matrix[3] = cos(angle);

    *O_x = 0.0;
    *O_y = 0.0;

}

__kernel void compute_descriptor(__global uchar* descriptor,
                                 __global const uchar* image,
                                 __global const size_t* patch_description_points,
                                 __global point* key_points,
                                  const int key_point_index,
                                 __global const float* rotation_matrix,
                                 __global const test* test_cases,
                                 const int width) {

	const size_t id = get_global_id(0); // j

    const test cur_patch = test_cases[patch_description_points[id]];

    const point pt1 = cur_patch.point1;
    const point pt2 = cur_patch.point2;

    const point pnt1 = {(int)(key_points[key_point_index].x +
        pt1.x * rotation_matrix[0] + pt1.y * rotation_matrix[2]),
        (int)(key_points[key_point_index].y + (-1) * pt1.x * rotation_matrix[1] + pt1.y * rotation_matrix[3])};

    const point pnt2 = {(int)(key_points[key_point_index].x +
        pt2.x * rotation_matrix[0] + pt2.y * rotation_matrix[2]),
        (int)(key_points[key_point_index].y + (-1) * pt2.x * rotation_matrix[1] + pt2.y * rotation_matrix[3])};


    descriptor[key_point_index * 512 + id] = image[width * pnt1.y + pnt1.x] > image[width * pnt2.y + pnt2.x] ? 1 : 0;
//	descriptor[key_point_index * 512 + id] = 69;
}
