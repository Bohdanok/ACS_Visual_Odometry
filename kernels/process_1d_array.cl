__kernel void process_1d_array(__global int* input_data, __global int* output_data) {
    output_data[get_global_id(0)] = input_data[get_global_id(0)] * 2;
}