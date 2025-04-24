__kernel void gradient_convolution(__global uchar* image_data,
                                   __global double* Jx,
                                   __global double* Jy,
                                   __global double* Jxy,
                                   const int width) {

    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    const size_t idx_m1 = (i - 1) * width + j;
    const size_t idx = i * width + j;
    const size_t idx_p1 = (i + 1) * width + j;

    // Jx[idx] = 69.0;  colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);


    double sumx[3] = {0};
    double sumy[3] = {0};

    // ptr_src1 = image_data.ptr<uchar>(idx - 1);
    // ptr_src2 = image_data.ptr<uchar>(idx);
    // ptr_src3 = image_data.ptr<uchar>(idx + 1);


    for (short k = -1; k <= 1; k++) {
        sumx[k + 1] = (double)(image_data[idx_m1 + k]) - (double)(image_data[idx_p1 + k]); // // [1, 0, -1] T
        sumy[k + 1] = (double)(image_data[idx_m1 + k]) + 2.0 * (double)(image_data[idx + k]) + (double)(image_data[idx_p1 + k]); // [1, 2, 1] T

    }

    Jx[idx] = sumx[0] + 2.0 * sumx[1] + sumx[2]; // [1, 2, 1]
    Jy[idx]  = sumy[0] - sumy[2]; // [1, 0, -1]
    Jxy[idx]  = sumx[0] - sumx[2]; // [1, 0, -1]

}


__kernel void shitomasi_response(__global double* R_renponse,
                                 __global const double* Jx,
                                 __global const double* Jy,
                                 __global const double* Jxy,
                                 const int width,
                                 const double RESPONSE_THRESHOLD) {

    double jx2 = 0;
    double jy2 = 0;
    double sumjxy = 0;
    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    const size_t idx = i * width + j;


    for (int m = -2; m <= 2; m++) {
        for (int n = -2; n <= 2; n++) {
            const size_t neighbor_idx = (i + m) * width + (j + n);
            const double jx = Jx[neighbor_idx];
            const double jy = Jy[neighbor_idx];
            const double jxy = Jxy[neighbor_idx];

            sumjxy += jxy;
            jx2 += jx * jx;
            jy2 += jy * jy;
        }
    }

    const double det = (jx2 * jy2) - (sumjxy * sumjxy);
    const double trace = jx2 + jy2;
    const double R = (trace / 2) - (0.5 * sqrt(trace * trace - 4 * det));

    R_renponse[idx] = R > RESPONSE_THRESHOLD ? R : 0;
//    R_renponse[idx] = 50000;
    // R_array[i][j] = R;
    // max_R = std::max(R, max_R);

}