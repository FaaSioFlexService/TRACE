use trace_numerical_core::{dot_product, matrix_matrix_multiply, matrix_vector_multiply, vector_add, vector_norm2, vector_scale, DenseMatrix, RuntimePolicy, Vector};

#[test]
fn vector_operations_match_reference() {
    let policy = RuntimePolicy { min_parallel_elements: 1_000_000, ..RuntimePolicy::default() };
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    assert_eq!(vector_add(&a, &b, policy).unwrap(), Vector::new(vec![5.0, 7.0, 9.0]));
    assert_eq!(vector_scale(&a, 2.0, policy), Vector::new(vec![2.0, 4.0, 6.0]));
    assert_eq!(dot_product(&a, &b, policy).unwrap(), 32.0);
    assert!((vector_norm2(&a, policy) - 14f64.sqrt()).abs() < 1e-12);
}

#[test]
fn matrix_operations_match_reference() {
    let policy = RuntimePolicy { min_parallel_elements: 1_000_000, ..RuntimePolicy::default() };
    let a = DenseMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = DenseMatrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let x = Vector::new(vec![1.0, 2.0, 3.0]);
    assert_eq!(matrix_vector_multiply(&a, &x, policy).unwrap(), Vector::new(vec![14.0, 32.0]));
    assert_eq!(matrix_matrix_multiply(&a, &b, policy).unwrap(), DenseMatrix::new(2, 2, vec![58.0, 64.0, 139.0, 154.0]).unwrap());
}

#[test]
fn advanced_blas_operations_match_reference() {
    use trace_numerical_core::{axpy, cross_product, determinant, ger, gemv, kronecker_product, trace, transpose, triangular_solve, RuntimePolicy};
    let policy = RuntimePolicy { min_parallel_elements: 1_000_000, ..RuntimePolicy::default() };
    let x = Vector::new(vec![1.0, 2.0, 3.0]);
    let y = Vector::new(vec![4.0, 5.0, 6.0]);
    assert_eq!(axpy(2.0, &x, &y, policy).unwrap(), Vector::new(vec![6.0, 9.0, 12.0]));
    assert_eq!(cross_product(&x, &y).unwrap(), Vector::new(vec![-3.0, 6.0, -3.0]));

    let matrix = DenseMatrix::new(2, 2, vec![2.0, 1.0, 1.0, 3.0]).unwrap();
    assert_eq!(trace(&matrix).unwrap(), 5.0);
    assert_eq!(determinant(&matrix, policy).unwrap(), 5.0);
    assert_eq!(transpose(&matrix).unwrap(), matrix);
    let rhs = Vector::new(vec![5.0, 10.0]);
    assert_eq!(gemv(1.0, &matrix, &rhs, 0.0, &Vector::new(vec![0.0, 0.0]), policy).unwrap(), Vector::new(vec![20.0, 35.0]));
    let lower = DenseMatrix::new(2, 2, vec![2.0, 0.0, 1.0, 3.0]).unwrap();
    assert_eq!(triangular_solve(&lower, &Vector::new(vec![2.0, 7.0]), true, false).unwrap(), Vector::new(vec![1.0, 2.0]));
    let update = ger(1.0, &Vector::new(vec![1.0, 2.0]), &Vector::new(vec![3.0, 4.0]), &matrix, policy).unwrap();
    assert_eq!(update, DenseMatrix::new(2, 2, vec![5.0, 5.0, 7.0, 11.0]).unwrap());
    assert_eq!(kronecker_product(&DenseMatrix::new(1, 2, vec![1.0, 2.0]).unwrap(), &DenseMatrix::new(2, 1, vec![3.0, 4.0]).unwrap()).unwrap(), DenseMatrix::new(2, 2, vec![3.0, 6.0, 4.0, 8.0]).unwrap());
}

#[test]
fn factorizations_reconstruct_known_matrices() {
    use trace_numerical_core::{cholesky, ldlt, lu, matrix_matrix_multiply, qr, svd, symmetric_eigen, RuntimePolicy};
    let policy = RuntimePolicy { min_parallel_elements: 1_000_000, ..RuntimePolicy::default() };
    let a = DenseMatrix::new(3, 3, vec![4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 6.0]).unwrap();

    let lu_factors = lu(&a, policy).unwrap();
    let lu_product = matrix_matrix_multiply(&lu_factors.l, &lu_factors.u, policy).unwrap();
    for value in lu_product.data.iter() { assert!(value.is_finite()); }

    let l = cholesky(&a).unwrap();
    let lt = trace_numerical_core::transpose(&l).unwrap();
    let reconstructed = matrix_matrix_multiply(&l, &lt, policy).unwrap();
    for (left, right) in reconstructed.data.iter().zip(a.data.iter()) { assert!((left - right).abs() < 1e-10); }

    let (ldlt_l, d) = ldlt(&a).unwrap();
    let mut d_matrix = vec![0.0; 9];
    for index in 0..3 { d_matrix[index * 3 + index] = d.data[index]; }
    let ld = matrix_matrix_multiply(&ldlt_l, &DenseMatrix::new(3, 3, d_matrix).unwrap(), policy).unwrap();
    let ldlt_product = matrix_matrix_multiply(&ld, &trace_numerical_core::transpose(&ldlt_l).unwrap(), policy).unwrap();
    for (left, right) in ldlt_product.data.iter().zip(a.data.iter()) { assert!((left - right).abs() < 1e-10); }

    let (q, r) = qr(&a).unwrap();
    let qr_product = matrix_matrix_multiply(&q, &r, policy).unwrap();
    for (left, right) in qr_product.data.iter().zip(a.data.iter()) { assert!((left - right).abs() < 1e-8); }

    let (u, singular_values, v) = svd(&a, policy).unwrap();
    assert_eq!(singular_values.len(), 3);
    assert_eq!(u.columns, 3);
    assert_eq!(v.columns, 3);

    let (eigenvalues, eigenvectors) = symmetric_eigen(&a, policy).unwrap();
    assert_eq!(eigenvalues.len(), 3);
    assert_eq!(eigenvectors.rows, 3);
}

#[test]
fn sparse_csc_matches_dense_spmv() {
    use trace_numerical_core::{dense_to_csc, matrix_vector_multiply_any, Matrix, RuntimePolicy};
    let policy = RuntimePolicy { min_parallel_elements: 1_000_000, ..RuntimePolicy::default() };
    let dense = DenseMatrix::new(3, 3, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]).unwrap();
    let csc = dense_to_csc(&dense, 0.0);
    let vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let result = matrix_vector_multiply_any(&Matrix::Csc(csc), &vector, policy).unwrap();
    assert_eq!(result, Vector::new(vec![7.0, 6.0, 19.0]));
}

#[test]
fn rank_revealing_and_polar_are_available() {
    use trace_numerical_core::{polar, rank_revealing_qr};
    let matrix = DenseMatrix::new(3, 2, vec![2.0, 0.0, 0.0, 3.0, 1.0, 1.0]).unwrap();
    let (q, r, permutation, rank) = rank_revealing_qr(&matrix).unwrap();
    assert_eq!(q.rows, 3);
    assert_eq!(r.columns, 2);
    assert_eq!(permutation.len(), 2);
    assert_eq!(rank, 2);

    let (unitary, positive) = polar(&matrix, RuntimePolicy::default()).unwrap();
    assert_eq!(unitary.rows, 3);
    assert_eq!(positive.rows, 2);
}
