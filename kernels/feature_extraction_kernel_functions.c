
typedef struct {
    int x;
    int y;
} point;

typedef struct {
    point point1;
    point point2;
} test;

//typedef struct {
//    int first;
//    int second;
//} pair;

//void atomic_add_float_global(__global float* p, float val) // stackoverflow
//{
//    asm volatile(
//        "atom.global.add.f32 _, [%0], %1;"  // "_" or omit output
//        :
//        : "l"(p), "f"(val)
//        : "memory"
//    );
//}

inline void atomic_add_float_global(__global float* source, float operand) { // chat
    union {
        unsigned int intVal;
        float floatVal;
    } next, expected, current;

    do {
        current.floatVal = *source;
        next.floatVal = current.floatVal + operand;
        expected.intVal = as_uint(current.floatVal);
    } while (atomic_cmpxchg((volatile __global int*)source, expected.intVal, as_uint(next.floatVal)) != expected.intVal);
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
    const float R = (trace / 2) - (0.5f * sqrt(trace * trace - 4 * det));

    R_renponse[idx] = R > RESPONSE_THRESHOLD ? R : 0;
//    R_renponse[idx] = 1000.0f * sin((float)(i + j) / 10.0f);
//	R_renponse[idx] = R;
    // R_array[i][j] = R;
    // max_R = std::max(R, max_R);

}



__kernel void compute_all_orientations( __global uchar* image_data,
    									__global const int* test_cases,
    									__global float* O_x,
    									__global float* O_y,
    									__global int* key_points,
    									const int width) {

    const size_t key_point_id = get_global_id(0);
    const size_t test_id = get_global_id(1);

    const int key_x = key_points[2 * key_point_id + 0];
    const int key_y = key_points[2 * key_point_id + 1];

    const int p1_x = test_cases[4 * test_id + 0];
    const int p1_y = test_cases[4 * test_id + 1];
    const int p2_x = test_cases[4 * test_id + 2];
    const int p2_y = test_cases[4 * test_id + 3];

    const int x1 = p1_x + key_x;
    const int y1 = p1_y + key_y;
    const int x2 = p2_x + key_x;
    const int y2 = p2_y + key_y;

    const float intensity1 = (float)image_data[width * y1 + x1];
    const float intensity2 = (float)image_data[width * y2 + x2];

    const float intensity_change = intensity1 - intensity2;


    const float dx = (float)(p1_x - p2_x);
    const float dy = (float)(p1_y - p2_y);
    const float norm = sqrt(dx * dx + dy * dy);

    if (norm == 0.0f) return;

    const float add_o_x = intensity_change * dx / norm;
    const float add_o_y = intensity_change * dy / norm;

    atomic_add_float_global(&O_x[key_point_id], add_o_x);
    atomic_add_float_global(&O_y[key_point_id], add_o_y);


}


__kernel void merge_all_orientations(__global float* O_x,
                                      __global float* O_y,
                                      __global float* rotation_matrix) {

//  	const float o_x = atomic_load(O_x);
//	const float o_y = atomic_load(O_y);
	const size_t key_point_id = get_global_id(0);

    const float o_x = O_x[key_point_id];
    const float o_y = O_y[key_point_id];

//	printf("O_x: %f\tO_y: %f\n", o_x, o_y);

    float angle = 0.0f;

    if (!(isnan(o_x) || isnan(o_y))) { // a check for isinf() might be useful here
        angle = atan2(o_y, o_x);
    }

//    printf("THE ANGLE: %f", angle);

    rotation_matrix[4 * key_point_id] = cos(angle);
    rotation_matrix[4 * key_point_id + 1] = -1.0f * sin(angle);
    rotation_matrix[4 * key_point_id + 2] = sin(angle);
    rotation_matrix[4 * key_point_id + 3] = cos(angle);

}


__kernel void compute_all_descriptors(__global uchar* descriptor,
                                 __global const uchar* image,
                                 __global const size_t* patch_description_points,
                                 __global point* key_points,
                                 __global const float* rotation_matrix,
                                 __global const test* test_cases,
                                 const int width) {

	const size_t key_point_id = get_global_id(0); // i
	const size_t desc_point_id = get_global_id(1); // j

    const test cur_patch = test_cases[patch_description_points[desc_point_id]];

    const point pt1 = cur_patch.point1;
    const point pt2 = cur_patch.point2;

    const point pnt1 = {(int)(key_points[key_point_id].x +
        pt1.x * rotation_matrix[key_point_id * 4 + 0] + pt1.y * rotation_matrix[key_point_id * 4 + 2]),
        (int)(key_points[key_point_id].y + (-1) * pt1.x * rotation_matrix[key_point_id * 4 + 1] + pt1.y * rotation_matrix[key_point_id * 4 + 3])};

    const point pnt2 = {(int)(key_points[key_point_id].x +
        pt2.x * rotation_matrix[key_point_id * 4 + 0] + pt2.y * rotation_matrix[key_point_id * 4 + 2]),
        (int)(key_points[key_point_id].y + (-1) * pt2.x * rotation_matrix[key_point_id * 4 + 1] + pt2.y * rotation_matrix[key_point_id * 4 + 3])};


    descriptor[key_point_id * 512 + desc_point_id] = image[width * pnt1.y + pnt1.x] > image[width * pnt2.y + pnt2.x] ? 1 : 0;
//	descriptor[key_point_id * 512 + id] = 69;
}
