use accelerate_rs::{
    cblas_dgemm, cblas_dgemv, cblas_dsymv, dpotrf_, CBLAS_ORDER_CblasRowMajor,
    CBLAS_TRANSPOSE_CblasNoTrans, CBLAS_TRANSPOSE_CblasTrans, CBLAS_UPLO_CblasLower,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn generate_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = vec![0.; n * n];
    let mut m = vec![0.; n * n];
    let mut v = vec![0.; n];
    for i in 0..n {
        for j in 0..(i + 1) {
            if i == j {
                r[i * n + j] = 1. * (i as f64 + 1.);
            } else {
                r[i * n + j] = 0.1 * (j as f64);
            }
        }

        v[i] = i as f64;
    }
    unsafe {
        cblas_dgemm(
            CBLAS_ORDER_CblasRowMajor,
            CBLAS_TRANSPOSE_CblasNoTrans,
            CBLAS_TRANSPOSE_CblasTrans,
            n as i32,
            n as i32,
            n as i32,
            1.,
            r.as_ptr(),
            n as i32,
            r.as_ptr(),
            n as i32,
            0.,
            m.as_mut_ptr(),
            n as i32,
        );
    }

    (m, v)
}

fn linalg_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg-ops");

    for i in [10, 100, 1000] {
        let p = generate_data(i);
        group.bench_with_input(BenchmarkId::new("gemv", i), &p, |b, p| {
            b.iter(|| {
                let mut r = vec![0.; i];
                unsafe {
                    cblas_dgemv(
                        CBLAS_ORDER_CblasRowMajor,
                        CBLAS_TRANSPOSE_CblasNoTrans,
                        i as i32,
                        i as i32,
                        1.,
                        p.0.as_ptr(),
                        i as i32,
                        p.1.as_ptr(),
                        1,
                        0.,
                        r.as_mut_ptr(),
                        1,
                    );
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("symv", i), &p, |b, p| {
            b.iter(|| {
                let mut r = vec![0.; i];
                unsafe {
                    cblas_dsymv(
                        CBLAS_ORDER_CblasRowMajor,
                        CBLAS_UPLO_CblasLower,
                        i as i32,
                        1.,
                        p.0.as_ptr(),
                        i as i32,
                        p.1.as_ptr(),
                        1,
                        0.,
                        r.as_mut_ptr(),
                        1,
                    );
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("potrf", i), &p, |b, p| {
            b.iter(|| unsafe {
                let uplo: i8 = 'U' as i8;
                let n: i32 = i as i32;
                let mut r = p.0.clone();
                let mut info: i32 = 0;
                dpotrf_(&uplo, &n, r.as_mut_ptr(), &n, &mut info);
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(10000, Output::Protobuf));
    targets = linalg_ops
}

criterion_main!(benches);
