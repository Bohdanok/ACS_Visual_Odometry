kernel void numerical_reduction(global int* data, local int* local_data, global int* output_data) {
    size_t global_id = get_global_id(0);
    size_t local_size = get_local_size(0);
    size_t local_id = get_local_id(0);

    local_data[local_id] = data[global_id];


    // Force all the work items to finish executing to proceed forward
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = local_size >> 1; i > 0; i >>= 1)
        if (local_id < i) {
            local_data[local_id] += local_data[local_id + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        output_data[get_group_id(0)] = local_data[0];
    }

}