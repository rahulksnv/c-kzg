/*
    Test bench.
*/
#include <assert.h> // assert()
#include <stdio.h>  // printf()
#include <stdlib.h> // malloc(), free(), atoi()
#include <string.h>
#include <time.h>
#include <unistd.h> // EXIT_SUCCESS/FAILURE
#include "bench_util.h"
#include "test_util.h"
#include "c_kzg.h"

#define BENCH_BASELINE  1
#define BENCH_AFFINE    2

// 32K * 32 bytes = 1 MB worth of data.
const uint64_t MAX_FRS = 32 * 1024;

// The time stats.
typedef struct {
    unsigned long data_bytes;
    unsigned long polynomial_len;
    unsigned long interpolate_time;
    unsigned long commit_time;
    unsigned long eval_time;
    unsigned long compute_proof_time;
    unsigned long check_proof_time;
} run_time_t;

/**
 * Initializes the run time stats.
 *
 * @param[out]  run_time   Run time stats
 */
void init_run_time(run_time_t* run_time) {
    run_time->data_bytes = 0;
    run_time->polynomial_len = 0;
    run_time->interpolate_time = 0;
    run_time->commit_time = 0;
    run_time->eval_time = 0;
    run_time->compute_proof_time = 0;
    run_time->check_proof_time = 0;
}

/**
 * Initializes the data with random Fr values.
 *
 * @param[in]  num_fr   Number of field element to initialize
 */
fr_t* init_data(uint64_t num_fr) {
    fr_t *data = malloc(num_fr * sizeof(fr_t));
    for (int i = 0; i < num_fr; i++) {
        uint64_t a[4];
        a[0] = rand_uint64();
        a[1] = rand_uint64();
        a[2] = rand_uint64();
        a[3] = rand_uint64();
        // Zero out the last byte of the 32 bytes, effectively making it a 31 byte entity.
        a[3] &= 0xFFFFFF00;
        fr_from_uint64s(&data[i], a);
    }
    return data;
}


/**
 * Initializes the settings.
 *
 * @param[out] fs      Initialized FFT settings
 * @param[out] ks      Initialized KZG settings
 * @param[in]  scale   log of polynomial length(1 more than the polynomial degree)
 */
void init_trusted_setup(FFTSettings *fs, KZGSettings *ks, int scale) {
    assert(C_KZG_OK == new_fft_settings(fs, scale));

    g1_t *s1 = malloc(fs->max_width * sizeof(g1_t));
    g2_t *s2 = malloc(fs->max_width * sizeof(g2_t));
    generate_trusted_setup(s1, s2, &secret, fs->max_width);
    assert(C_KZG_OK == new_kzg_settings(ks, s1, s2, fs->max_width, fs));
}

/*
 * Runs the baseline benchmark for the specified time.
 *
 * @param[out] run_time             Run time stats
 * @param[in]  data                 Polynomial eval
 * @param[in]  scale                log of polynomial length(1 more than the polynomial degree)
 * @param[in]  max_seconds          Test duration
*/
void run_bench_baseline(
    run_time_t* run_time,
    fr_t* data,
    int scale,
    int max_seconds
) {
    timespec_t start_ts, end_ts;
    timespec_t t0, t1;
    unsigned long total_time = 0, iter = 0;
    FFTSettings fs;
    KZGSettings ks;
    g1_t commitment, proof;
    fr_t eval_x, eval_value;
    poly p;
    bool result;

    init_trusted_setup(&fs, &ks, scale);
    fr_from_uint64(&eval_x, 1234);
    init_run_time(run_time);
    while (total_time < max_seconds * NANO) {
        iter++;
        clock_gettime(CLOCK_REALTIME, &start_ts);

        // Interpolate the polynomial from the data points.
        clock_gettime(CLOCK_REALTIME, &t0);
        assert(C_KZG_OK == new_poly(&p, fs.max_width));
        assert(C_KZG_OK == fft_fr(p.coeffs, data, true, fs.max_width, ks.fs));
        clock_gettime(CLOCK_REALTIME, &t1);
        run_time->data_bytes += fs.max_width * sizeof(fr_t);
        run_time->polynomial_len += p.length;
        run_time->interpolate_time += tdiff_usec(t0, t1);

        // Create commitment
        clock_gettime(CLOCK_REALTIME, &t0);
        assert(C_KZG_OK == commit_to_poly(&commitment, &p, &ks));
        clock_gettime(CLOCK_REALTIME, &t1);
        run_time->commit_time += tdiff_usec(t0, t1);

        // Evaluate the polynomial
        clock_gettime(CLOCK_REALTIME, &t0);
        eval_poly(&eval_value, &p, &eval_x);
        clock_gettime(CLOCK_REALTIME, &t1);
        run_time->eval_time += tdiff_usec(t0, t1);

        // Create witness
        clock_gettime(CLOCK_REALTIME, &t0);
        assert(C_KZG_OK == compute_proof_single(&proof, &p, &eval_x, &ks));
        clock_gettime(CLOCK_REALTIME, &t1);
        run_time->compute_proof_time += tdiff_usec(t0, t1);

        // Check the evaluation
        clock_gettime(CLOCK_REALTIME, &t0);
        assert(C_KZG_OK == check_proof_single(&result, &commitment, &proof, &eval_x, &eval_value, &ks));
        assert(result);
        clock_gettime(CLOCK_REALTIME, &t1);
        run_time->check_proof_time += tdiff_usec(t0, t1);

        clock_gettime(CLOCK_REALTIME, &end_ts);
        total_time += tdiff(start_ts, end_ts);
        free_poly(&p);
    }

    run_time->data_bytes /= iter;
    run_time->polynomial_len /= iter;
    run_time->interpolate_time /= iter;
    run_time->commit_time /= iter;
    run_time->eval_time /= iter;
    run_time->compute_proof_time /= iter;
    run_time->check_proof_time /= iter;

    free_kzg_settings(&ks);
    free_fft_settings(&fs);
}

/*
 * Runs the benchmark for g1 -> affine transformation.
 *
 * @param[out] run_time             Run time (usec)
 * @param[in]  data                 Polynomial eval
 * @param[in]  scale                log of polynomial length(1 more than the polynomial degree)
 * @param[in]  max_seconds          Test duration
*/
void run_bench_affine_transformation(
    unsigned long* run_time,
    const g1_t *p,
    poly *poly,
    const uint64_t len,
    int max_seconds
) {
    timespec_t start_ts, end_ts;
    timespec_t t0, t1;
    unsigned long total_time = 0, iter = 0;

    *run_time= 0;
    while (total_time < max_seconds * NANO) {
        iter++;
        clock_gettime(CLOCK_REALTIME, &start_ts);

        clock_gettime(CLOCK_REALTIME, &t0);

        blst_p1_affine *p_affine = malloc(len * sizeof(blst_p1_affine));
        const blst_p1 *p_arg[2] = {p, NULL};
        blst_p1s_to_affine(p_affine, p_arg, len);

        blst_scalar *scalars = malloc(len * sizeof(blst_scalar));
        for (int i = 0; i < len; i++) {
            blst_scalar_from_fr(&scalars[i], &poly->coeffs[i]);
        }

        void *scratch = malloc(blst_p1s_mult_pippenger_scratch_sizeof(len));
        const byte *scalars_arg[2] = {(byte *)scalars, NULL};
        const blst_p1_affine *points_arg[2] = {p_affine, NULL};
        g1_t out;
        blst_p1s_mult_pippenger(&out, points_arg, len, scalars_arg, 256, scratch);

        free(scratch);
        free(p_affine);
        free(scalars);

        clock_gettime(CLOCK_REALTIME, &t1);
        *run_time += tdiff_usec(t0, t1);

        clock_gettime(CLOCK_REALTIME, &end_ts);
        total_time += tdiff(start_ts, end_ts);
    }
    *run_time /= iter;
}

int main(int argc, char *argv[]) {
    int nsec = 0;
    int baseline_type = BENCH_BASELINE;

    switch (argc) {
    case 1:
        nsec = NSEC;
        break;
    case 2:
        nsec = atoi(argv[1]);
        break;
    case 3:
        nsec = atoi(argv[1]);
        if (!strcmp(argv[2], "baseline")) {
            baseline_type = BENCH_BASELINE;
        } if (!strcmp(argv[2], "affine")) {
            baseline_type = BENCH_AFFINE;
        } else {
            printf("Invalid bench type: Usage: %s [test time in seconds > 0] [baseline|affine]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
        break;
    default:
        break;
    };

    if (nsec == 0) {
        printf("Usage: %s [test time in seconds > 0] [baseline]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    fr_t *data = init_data(MAX_FRS);

    if (baseline_type == BENCH_BASELINE) {
        printf("*** Benchmarking baseline c_kzg, %d second%s per test.\n", nsec, nsec == 1 ? "" : "s");
        run_time_t run_time;
        run_bench_baseline(&run_time, data, 1, nsec);
        run_bench_baseline(&run_time, data, 2, nsec);
        for (int scale = 1; scale <= 15; scale++) {
            run_bench_baseline(&run_time, data, scale, nsec);
            printf("data = %7lu bytes(polynomial_len = %5lu): create-polynomial = %6lu, commit = %6lu, eval = %6lu, "
                    "create-witness = %6lu, verify = %6lu  (usec/op)\n",
                    run_time.data_bytes, run_time.polynomial_len,
                    run_time.interpolate_time, run_time.commit_time, run_time.eval_time,
                    run_time.compute_proof_time, run_time.check_proof_time);
        }
    } else if (baseline_type == BENCH_AFFINE) {
        FFTSettings fs;
        KZGSettings ks;
        unsigned long run_time;

        printf("*** Benchmarking affine transform, %d second%s per test.\n", nsec, nsec == 1 ? "" : "s");
        for (int scale = 1; scale <= 15; scale++) {
            poly p;
            uint64_t len = 1 << scale;

            init_trusted_setup(&fs, &ks, scale);
            assert(C_KZG_OK == new_poly(&p, fs.max_width));
            run_bench_affine_transformation(&run_time, ks.secret_g1, &p, len, nsec);
            free_kzg_settings(&ks);
            free_fft_settings(&fs);
            free_poly(&p);

            printf("len = %5llu: g1->affine transform time = %6lu (usec/op)\n", len, run_time);
        }
    } else {
        printf("Bench type: Usage: %s [test time in seconds > 0] [baseline|affine]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
}
