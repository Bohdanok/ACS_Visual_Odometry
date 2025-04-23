kernel void process_2d_array(global int* data) {
    size_t id = (get_global_id(1) * get_global_size(0)) + get_global_id(0);
    data[id] = 69420;
}