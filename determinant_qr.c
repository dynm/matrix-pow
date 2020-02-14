#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>
#include "sph_keccak.h"

static const double double_map[16][4] = {
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

void expand_doubles(double *output, uint8_t *input_bytes)
{
    uint8_t key0;
    uint8_t key1;
    for (int i = 0; i < 32; i++)
    {
        key0 = (input_bytes[i] >> 4) & 0x0f;
        key1 = (input_bytes[i]) & 0x0f;
        output[0 + 8 * i] = double_map[key0][0];
        output[1 + 8 * i] = double_map[key0][1];
        output[2 + 8 * i] = double_map[key0][2];
        output[3 + 8 * i] = double_map[key0][3];

        output[4 + 8 * i] = double_map[key1][0];
        output[5 + 8 * i] = double_map[key1][1];
        output[6 + 8 * i] = double_map[key1][2];
        output[7 + 8 * i] = double_map[key1][3];
    }
}

#define MATRIX_DIM 30

double reduce_vector2(__m256d input) {
  __m256d temp = _mm256_hadd_pd(input, input);
  __m128d sum_high = _mm256_extractf128_pd(temp, 1);
  __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(temp));
  return ((double*)&result)[0];
}

double dot_product(const double *a, const double *b) {
  __m256d sum_vec = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

  /* Add up partial dot-products in blocks of 256 bits */
  for(int ii = 0; ii < MATRIX_DIM/4; ++ii) {
    __m256d x = _mm256_load_pd(a+4*ii);
    __m256d y = _mm256_load_pd(b+4*ii);
    __m256d z = _mm256_mul_pd(x,y);
    sum_vec = _mm256_add_pd(sum_vec, z);
  }

  /* Find the partial dot-product for the remaining elements after
   * dealing with all 256-bit blocks. */
  double final = 0.0;
  for(int ii = MATRIX_DIM-MATRIX_DIM%4; ii < MATRIX_DIM; ++ii)
    final += a[ii] * b[ii];

  return reduce_vector2(sum_vec) + final;
}

double sqrt_cache[30];
void sqrt_cache_init()
{
    for (int i = 0; i < 30; i++)
    {
        sqrt_cache[i] = sqrt((double)i);
    }
}

double sqrt_cache_get(double input)
{
    return sqrt_cache[(int)input];
}

void qr(double *input_mat, double *det)
{
    __attribute__ ((aligned (32))) double u_vec[MATRIX_DIM] = {0.0};
    __attribute__ ((aligned (32))) double col_vec[MATRIX_DIM] = {0.0};
    double local_mat[MATRIX_DIM][MATRIX_DIM] = {0.0};
    double u_length_squared, dot;

    double prod = 0.0f, vec_length = 0.0f;

    int row;
    double priv_det = 1;

    for (int row = 0; row < MATRIX_DIM; row++)
    {
        for (int j = 0; j < MATRIX_DIM; j++)
        {
            local_mat[row][j] = input_mat[row * MATRIX_DIM + j];
        }
    }

    /* Load first column into local memory as u vector */
    for (row = 0; row < MATRIX_DIM; row++)
    {
        u_vec[row] = local_mat[row][0];
    }

    /* Find length of first A column and u vector */
    for (int i = 1; i < MATRIX_DIM; i++)
    {
        // vec_length += u_vec[i] * u_vec[i];
        vec_length += u_vec[i];
    }
    u_length_squared = vec_length;
    vec_length = sqrt_cache_get(vec_length + u_vec[0]);
    local_mat[0][0] = vec_length;
    u_vec[0] -= vec_length;
    u_length_squared += u_vec[0] * u_vec[0];

    for (row = 1; row < MATRIX_DIM; row++)
    {
        local_mat[row][0] = 0.0f;
    }

    for (int i = 1; i < MATRIX_DIM; i++)
    {
        dot = 0.0f;
        for (int j = 0; j < MATRIX_DIM; j++)
        {
            dot += local_mat[j][i] * u_vec[j];
            // col_vec[j] = local_mat[j][i];
        }
        // dot = dot_product(col_vec, u_vec);
        for (row = 0; row < MATRIX_DIM; row++)
        {
            local_mat[row][i] -= 2 * u_vec[row] * dot / u_length_squared;
        }
    }

    /* Load new column into memory */
    for (int col = 1; col < MATRIX_DIM - 1; col++)
    {
        for (row = 0; row < MATRIX_DIM; row++)
        {
            u_vec[row] = local_mat[row][col];
        }
        for (row = 0; row < MATRIX_DIM; row++)
        {
            if (row == col)
            {
                vec_length = 0.0f;
                for (int i = row + 1; i < MATRIX_DIM; i++)
                {
                    vec_length += u_vec[i] * u_vec[i];
                }
                u_length_squared = vec_length;
                vec_length = sqrt(vec_length + u_vec[row] * u_vec[row]);
                u_vec[row] -= vec_length;
                u_length_squared += u_vec[row] * u_vec[row];
                local_mat[row][row] = vec_length;
            }
            // else if (row > col)
            // {
            //     local_mat[row][col] = 0.0f;
            // }
        }
        /* Transform further columns of A */
        for (int i = col + 1; i < MATRIX_DIM; i++)
        {
            dot = 0.0f;
            for (int j = col; j < MATRIX_DIM; j++)
            {
                dot += local_mat[j][i] * u_vec[j];
            }
            
            // for (int j = 0; j < MATRIX_DIM; j++)
            // {
            //     // dot += local_mat[j][i] * u_vec[j];
            //     if (j<col){
            //         col_vec[j] = 0;
            //     }else{
            //         col_vec[j] = local_mat[j][i];
            //     }
            // }
            // dot = dot_product(col_vec, u_vec);
            for (row = 0; row < MATRIX_DIM; row++)
            {
                if (row >= col)
                    local_mat[row][i] -= 2 * u_vec[row] * dot / u_length_squared;
            }
        }
    }

    for (int i = 0; i < MATRIX_DIM; i++)
    {
        priv_det *= local_mat[i][i];
    }
    *(det) = -priv_det;
}
char *header =
    "\xf9\x01\x00\xa0\x93\x54\x69\x17\x41\x5f\xa4\xac\x1d\x4b\xb0\xda"
    "\x26\x95\xd6\x3e\x46\xaa\xd4\x94\x94\x18\x34\x43\x64\x3e\x0c\x67"
    "\x02\x8b\xd0\x47\x94\xba\xd5\x58\xfc\x41\xfa\xb4\x29\xac\xf3\x9c"
    "\xf8\xae\x1c\x6e\xe9\x5d\x52\xb0\xa1\xa0\x9d\x81\x21\x1c\xe9\x41"
    "\x2c\x59\x6b\xdb\x32\x0f\xa8\xbb\x68\x5b\xa9\x08\x33\x10\x11\xff"
    "\x7b\x11\x61\x69\x80\xd3\xb0\xfa\xa6\x0e\xa0\xf1\x88\x78\x00\xe0"
    "\x6a\x2d\xbf\x41\xc6\x5a\xab\x7a\x92\x6e\x57\xd1\x69\x8d\x9e\xef"
    "\xa9\xaf\x57\xeb\xfa\xa0\x56\x22\xb6\xe9\xa0\xa0\x17\xaa\xdb\x58"
    "\x87\xfe\xe6\xc7\xa4\x72\x1c\xbb\xaf\xd5\xf4\xf6\x09\xc8\x56\x70"
    "\xb8\xc2\x1f\xee\xbf\xf4\xf2\x9e\xb3\x2f\x05\x94\xa0\x00\x00\x00"
    "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa0\x00\x00"
    "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x84\x03"
    "\x80\x36\xfe\x83\x14\x39\x3e\x84\x5d\xd1\x5f\x75\x93\x31\x30\x30"
    "\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30\x30"
    "\x80\x80\x80";

typedef union _nonce_t {
    uint64_t uint64_t[2];
    uint8_t uint8_t[16];
} nonce_t;

void run_mpow(uint64_t offset, double target)
{
    double mat[MATRIX_DIM * MATRIX_DIM];
    uint8_t hash_result[32];
    double hash_mat[MATRIX_DIM][256];
    nonce_t nonce_array;
    uint64_t nonce = 0;
    sph_keccak_context ctx;
    uint8_t header_mutable[259];
    double dets[226];
    memcpy(header_mutable, header, 259);
    int has_chance = 0;

    for (nonce = 0; nonce < MATRIX_DIM; nonce++)
    {
        uint64_t divisor = 1;
        for (int pos = 0; pos < 16; pos++)
        {
            nonce_array.uint8_t[15 - pos] = 0x30 + ((nonce + offset) / divisor) % 10;
            divisor *= 10;
        }
        memcpy(header_mutable + 240, nonce_array.uint8_t, 16);

        sph_keccak256_init(&ctx);
        sph_keccak256(&ctx, header_mutable, 259);
        sph_keccak256_close(&ctx, hash_result);

        expand_doubles(hash_mat[nonce], hash_result);
    }

    for (int det_offset = 206; det_offset < 226; det_offset++)
    {
        for (int i = 0; i < MATRIX_DIM; i++)
        {
            for (int j = 0; j < MATRIX_DIM; j++)
            {
                mat[i * MATRIX_DIM + j] = hash_mat[i][j + det_offset];
            }
        }
        double det;
        qr(mat, &det);
        dets[det_offset] = det;
        if (det > target)
        {
            has_chance = 1;
        }
        // printf("det: %f\n", det);
    }
    if (!has_chance)
    {
        return;
    }

    // printf("possible cond\n");

    int positive_cnt = 0;
    for (int det_offset = 1; det_offset < 206; det_offset++)
    {
        for (int i = 0; i < MATRIX_DIM; i++)
        {
            for (int j = 0; j < MATRIX_DIM; j++)
            {
                mat[i * MATRIX_DIM + j] = hash_mat[i][j + det_offset];
            }
        }
        double det;
        qr(mat, &det);
        dets[det_offset] = det;
        if (det > 0)
        {
            positive_cnt++;
        }
        if (positive_cnt == 119)
        {
            if (det > target)
            {
                printf("found nonce in origin: %ld, det: %f\n", offset, det);
            }
            return;
        }
        if (positive_cnt + (256 - det_offset - 30) < 119)
        {
            return;
        }
    }
    for (int det_offset = 206; det_offset < 226; det_offset++)
    {
        double det;
        det = dets[det_offset];
        if (det > 0)
        {
            positive_cnt++;
        }
        if (positive_cnt == 119)
        {
            if (det > target)
            {
                printf("found nonce in speculer: %ld, det: %f\n", offset, det);
            }
            return;
        }
        if (positive_cnt + (256 - det_offset - 30) < 119)
        {
            return;
        }
    }
}

double target = 3000000000.0;
#define MIN_NONCE 995313
#define MAX_NONCE 9999999999999999

int main()
{
    sqrt_cache_init();
    clock_t begin = clock();
    for (int try = MIN_NONCE; try < MIN_NONCE+10000; try ++)
    {
        run_mpow(try, target);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("clock: %lf\n", time_spent);
    // getchar();
}