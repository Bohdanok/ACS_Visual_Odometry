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