static const __constant float float_map[16][4] = {
    {0, 0, 0, 0}, // 0
    {0, 0, 0, 1}, // 1
    {0, 0, 1, 0}, // 2
    {0, 0, 1, 1}, // 3
    {0, 1, 0, 0}, // 4
    {0, 1, 0, 1}, // 5
    {0, 1, 1, 0}, // 6
    {0, 1, 1, 1}, // 7
    {1, 0, 0, 0}, // 8
    {1, 0, 0, 1}, // 9
    {1, 0, 1, 0}, // a
    {1, 0, 1, 1}, // b
    {1, 1, 0, 0}, // c
    {1, 1, 0, 1}, // d
    {1, 1, 1, 0}, // e
    {1, 1, 1, 1}, // f
};

inline void expand_floats(float* output, uchar* input_bytes) {
    uchar key0;
    uchar key1;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        key0 = (input_bytes[i] >> 4) & 0x0f;
        key1 = (input_bytes[i]) & 0x0f;
        output[0 + 8 * i] = float_map[key0][0];
        output[1 + 8 * i] = float_map[key0][1];
        output[2 + 8 * i] = float_map[key0][2];
        output[3 + 8 * i] = float_map[key0][3];

        output[4 + 8 * i] = float_map[key1][0];
        output[5 + 8 * i] = float_map[key1][1];
        output[6 + 8 * i] = float_map[key1][2];
        output[7 + 8 * i] = float_map[key1][3];
    }
}

// #pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
static const __constant ulong keccakf_rndc[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008 };

static const __constant uint keccakf_rotc[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44 };

static const __constant uint keccakf_piln[24] = { 10, 7,  11, 17, 18, 3,  5,  16,
                                                 8,  21, 24, 4,  15, 23, 19, 13,
                                                 12, 2,  20, 14, 22, 9,  6,  1 };

inline void keccakf1600_2(ulong* st) {
    int i, round;
    ulong t, bc[5];

#pragma unroll 1
    for (round = 0; round < 24; ++round) {
        // Theta

        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20] ^
            rotate(st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22], 1UL);
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21] ^
            rotate(st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23], 1UL);
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22] ^
            rotate(st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24], 1UL);
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23] ^
            rotate(st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20], 1UL);
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24] ^
            rotate(st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21], 1UL);

        st[0] ^= bc[4];
        st[5] ^= bc[4];
        st[10] ^= bc[4];
        st[15] ^= bc[4];
        st[20] ^= bc[4];

        st[1] ^= bc[0];
        st[6] ^= bc[0];
        st[11] ^= bc[0];
        st[16] ^= bc[0];
        st[21] ^= bc[0];

        st[2] ^= bc[1];
        st[7] ^= bc[1];
        st[12] ^= bc[1];
        st[17] ^= bc[1];
        st[22] ^= bc[1];

        st[3] ^= bc[2];
        st[8] ^= bc[2];
        st[13] ^= bc[2];
        st[18] ^= bc[2];
        st[23] ^= bc[2];

        st[4] ^= bc[3];
        st[9] ^= bc[3];
        st[14] ^= bc[3];
        st[19] ^= bc[3];
        st[24] ^= bc[3];

        // Rho Pi
        t = st[1];
#pragma unroll
        for (i = 0; i < 24; ++i) {
            bc[0] = st[keccakf_piln[i]];
            st[keccakf_piln[i]] = rotate(t, (ulong)keccakf_rotc[i]);
            t = bc[0];
        }

#pragma unroll
        for (int i = 0; i < 25; i += 5) {
            ulong tmp1 = st[i], tmp2 = st[i + 1];

            st[i] = bitselect(st[i] ^ st[i + 2], st[i], st[i + 1]);
            st[i + 1] = bitselect(st[i + 1] ^ st[i + 3], st[i + 1], st[i + 2]);
            st[i + 2] = bitselect(st[i + 2] ^ st[i + 4], st[i + 2], st[i + 3]);
            st[i + 3] = bitselect(st[i + 3] ^ tmp1, st[i + 3], st[i + 4]);
            st[i + 4] = bitselect(st[i + 4] ^ tmp2, st[i + 4], tmp1);
        }

        //  Iota
        st[0] ^= keccakf_rndc[round];
    }
}

// #pragma OPENCL EXTENSION cl_amd_media_ops2 : enable

typedef union _nonce_t {
    ulong uint64_t[2];
    uchar uint8_t[16];
} nonce_t;

#define OFFSET 28
// __attribute__((reqd_work_group_size(WORKSIZE, 8, 1)))
__kernel void genkeccakmat(__global ulong* header,
    __global float* mats, __global ulong* hash) {
    ulong id = get_local_id(0);
    ulong divisor = 1;
    __local ulong midstate[25];
    ulong priv_state[25] = { 0 };
    float float_mat[256] = { 0.0f };

    nonce_t nonce_array;

    if (id == 0) {
        priv_state[0] = header[0];
        priv_state[1] = header[1];
        priv_state[2] = header[2];
        priv_state[3] = header[3];
        priv_state[4] = header[4];
        priv_state[5] = header[5];
        priv_state[6] = header[6];
        priv_state[7] = header[7];
        priv_state[8] = header[8];
        priv_state[9] = header[9];
        priv_state[10] = header[10];
        priv_state[11] = header[11];
        priv_state[12] = header[12];
        priv_state[13] = header[13];
        priv_state[14] = header[14];
        priv_state[15] = header[15];
        priv_state[16] = header[16];
        keccakf1600_2(priv_state);

        priv_state[0] ^= header[17 + 0];
        priv_state[1] ^= header[17 + 1];
        priv_state[2] ^= header[17 + 2];
        priv_state[3] ^= header[17 + 3];
        priv_state[4] ^= header[17 + 4];
        priv_state[5] ^= header[17 + 5];
        priv_state[6] ^= header[17 + 6];
        priv_state[7] ^= header[17 + 7];
        priv_state[8] ^= header[17 + 8];
        priv_state[9] ^= header[17 + 9];
        priv_state[10] ^= header[17 + 10];
        priv_state[11] ^= header[17 + 11];
        priv_state[12] ^= header[17 + 12]; // +5

        priv_state[15] ^= 0x0000000001808080; // +3 = 1
        priv_state[16] ^= 0x8000000000000000; // 00..0080

#pragma unroll
        for (int i = 0; i < 25; i++) {
            midstate[i] = priv_state[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ulong nonce = id;
#pragma unroll
    for (int pos = 0; pos < 16; pos++) {
        nonce_array.uint8_t[15 - pos] = 0x30 + (nonce / divisor) % 10;
        divisor *= 10;
    }

#pragma unroll
    for (int i = 0; i < 25; i++) {
        priv_state[i] = midstate[i];
    }

    priv_state[13] ^= nonce_array.uint64_t[0];
    priv_state[14] ^= nonce_array.uint64_t[1];

    keccakf1600_2(priv_state);

    // hash[0 + 4 * id] = priv_state[0];
    // hash[1 + 4 * id] = priv_state[1];
    // hash[2 + 4 * id] = priv_state[2];
    // hash[3 + 4 * id] = priv_state[3];

    expand_floats(float_mat, (uchar*)priv_state);
    for (int i = 0; i < 256; i++) {
        *(mats + 256 * id + i) = float_mat[i];
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

#define MATRIX_DIM 30

__kernel void qr(__local float* u_vec, __global float* input_mat,
    __global float* det) {
    local float local_mat[MATRIX_DIM * MATRIX_DIM];
    local float u_length_squared, dot;
    float prod = 0.0f, vec_length = 0.0f;

    int id = get_local_id(0);
    int col_offset = get_global_id(1);
    int row_offset = get_global_id(2);

#pragma unroll
    for (int i = 0; i < MATRIX_DIM; i++) {
        // copy cols
        local_mat[id * MATRIX_DIM + i] =
            input_mat[(id + row_offset) * 256 + col_offset + i];
    }
    /* Load first column into local memory as u vector */
    u_vec[id] = local_mat[id * MATRIX_DIM];
    barrier(CLK_LOCAL_MEM_FENCE);
    // *(det+col_offset) = local_mat[1*30+0];

    /* Find length of first A column and u vector */
    if (id == 0) {
        for (int i = 1; i < MATRIX_DIM; i++) {
            // vec_length += u_vec[i] * u_vec[i];
            vec_length += u_vec[i];
        }
        u_length_squared = vec_length;
        // vec_length = sqrt(vec_length + u_vec[0] * u_vec[0]);
        vec_length = sqrt(vec_length + u_vec[0]);
        local_mat[0] = vec_length;
        u_vec[0] -= vec_length;
        u_length_squared += u_vec[0] * u_vec[0];
    }
    else {
        local_mat[id * MATRIX_DIM] = 0.0f;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* Transform further columns of A */
    for (int i = 1; i < MATRIX_DIM; i++) {
        dot = 0.0f;
        if (id == 0) {
            for (int j = 0; j < MATRIX_DIM; j++) {
                dot += local_mat[j * MATRIX_DIM + i] * u_vec[j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        local_mat[id * MATRIX_DIM + i] -= 2 * u_vec[id] * dot / u_length_squared;
    }

    /* Loop through other columns */
    for (int col = 1; col < MATRIX_DIM - 1; col++) {

        /* Load new column into memory */
        u_vec[id] = local_mat[id * MATRIX_DIM + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Find length of A column and u vector */
        if (id == col) {
            vec_length = 0.0f;
            for (int i = col + 1; i < MATRIX_DIM; i++) {
                vec_length += u_vec[i] * u_vec[i];
            }
            u_length_squared = vec_length;
            vec_length = sqrt(vec_length + u_vec[col] * u_vec[col]);
            u_vec[col] -= vec_length;
            u_length_squared += u_vec[col] * u_vec[col];
            local_mat[col * MATRIX_DIM + col] = vec_length;
        }
        else if (id > col) {
            local_mat[id * MATRIX_DIM + col] = 0.0f;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        /* Transform further columns of A */
        for (int i = col + 1; i < MATRIX_DIM; i++) {
            if (id == 0) {
                dot = 0.0f;
                for (int j = col; j < MATRIX_DIM; j++) {
                    dot += local_mat[j * MATRIX_DIM + i] * u_vec[j];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (id >= col)
                local_mat[id * MATRIX_DIM + i] -=
                2 * u_vec[id] * dot / u_length_squared;
            barrier(CLK_GLOBAL_MEM_FENCE);
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (id == 0) {
        float priv_det = 1;
        for (int i = 0; i < MATRIX_DIM; i++) {
            priv_det *= local_mat[i * MATRIX_DIM + i];
        }
        *(det + (row_offset * 226) + col_offset) = -priv_det;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void check_dets(__global float* det, float target,
    __global uint* has_nonce, __global float* target_pos_det) {
    int row = get_global_id(0);
    int nonZeroCnt = 0;
    float cur_det = 0.0;
    for (int col = 0; col < 226; col++) {
        cur_det = *(det + row * 226 + col);
        if (cur_det > 0) {
            nonZeroCnt++;
        }
        if (nonZeroCnt == 119 && cur_det > target) {
            has_nonce[row] = col;
            target_pos_det[row] = cur_det;
            return;
        }
    }
    has_nonce[row] = 0;
    target_pos_det[row] = 0;
    // cur_det = *(det + row * 226 + 224);
    // has_nonce[row] = 1;
    // target_pos_det[row] = cur_det;
    barrier(CLK_GLOBAL_MEM_FENCE);
}