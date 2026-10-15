/**
 * Catálogo único de operaciones algebraicas.
 *
 * La interfaz de usuario, el benchmark y los backends consultan este catálogo
 * para evitar listas duplicadas y mantener una semántica común entre browser
 * y servidor.
 */
import type { OperationName } from "./types";

export type OperationCategory = "level1" | "level2" | "level3" | "matrix" | "factorization" | "eigen" | "exact";
export type OperationImplementationStatus = "implemented" | "planned";

export interface OperationSpec {
  readonly id: OperationName;
  readonly category: OperationCategory;
  readonly complexity: string;
  readonly description: string;
  readonly formula: string;
  readonly requiresSquare?: boolean;
  readonly status: OperationImplementationStatus;
}

export const OPERATION_CATALOG: readonly OperationSpec[] = [
  { id: "vectorAdd", category: "level1", complexity: "O(n)", description: "Vector addition", formula: "z = x + y", status: "implemented" },
  { id: "vectorScale", category: "level1", complexity: "O(n)", description: "Vector scaling", formula: "y = αx", status: "implemented" },
  { id: "dotProduct", category: "level1", complexity: "O(n)", description: "Dot product", formula: "xᵀy", status: "implemented" },
  { id: "vectorNorm2", category: "level1", complexity: "O(n)", description: "Euclidean vector norm", formula: "||x||₂", status: "implemented" },
  { id: "axpy", category: "level1", complexity: "O(n)", description: "Scaled vector addition", formula: "y = αx + y", status: "implemented" },
  { id: "crossProduct", category: "level1", complexity: "O(1)", description: "3D cross product", formula: "x × y", status: "implemented" },
  { id: "gemv", category: "level2", complexity: "O(mn)", description: "General matrix-vector multiplication", formula: "y = αAx + βy", status: "implemented" },
  { id: "matrixVectorMultiply", category: "level2", complexity: "O(mn)", description: "Matrix-vector multiplication", formula: "y = Ax", status: "implemented" },
  { id: "triangularSolve", category: "level2", complexity: "O(n²)", description: "Triangular system solve", formula: "Tx = b", status: "implemented" },
  { id: "ger", category: "level2", complexity: "O(mn)", description: "Rank-1 update", formula: "A = A + αxyᵀ", status: "implemented" },
  { id: "matrixMatrixMultiply", category: "level3", complexity: "O(mnk)", description: "General matrix multiplication", formula: "C = αAB + βC", status: "implemented" },
  { id: "trsm", category: "level3", complexity: "O(n²k)", description: "Triangular solve with multiple RHS", formula: "TX = B", status: "implemented" },
  { id: "syrk", category: "level3", complexity: "O(m²k)", description: "Symmetric rank-k update", formula: "C = αAAᵀ + βC", status: "implemented" },
  { id: "transpose", category: "matrix", complexity: "O(mn)", description: "Matrix transpose", formula: "B = Aᵀ", status: "implemented" },
  { id: "inverse", category: "matrix", complexity: "O(n³)", description: "Matrix inverse", formula: "A A⁻¹ = I", requiresSquare: true, status: "implemented" },
  { id: "determinant", category: "matrix", complexity: "O(n³)", description: "Determinant via LU", formula: "det(A)", requiresSquare: true, status: "implemented" },
  { id: "trace", category: "matrix", complexity: "O(n)", description: "Matrix trace", formula: "tr(A)", requiresSquare: true, status: "implemented" },
  { id: "matrixNormFrobenius", category: "matrix", complexity: "O(mn)", description: "Frobenius norm", formula: "||A||F", status: "implemented" },
  { id: "matrixNormOne", category: "matrix", complexity: "O(mn)", description: "Matrix 1-norm", formula: "maxⱼ Σᵢ |aᵢⱼ|", status: "implemented" },
  { id: "matrixNormInf", category: "matrix", complexity: "O(mn)", description: "Matrix infinity norm", formula: "maxᵢ Σⱼ |aᵢⱼ|", status: "implemented" },
  { id: "matrixNormTwo", category: "matrix", complexity: "O(kmn)", description: "Spectral norm", formula: "σmax(A)", status: "implemented" },
  { id: "kroneckerProduct", category: "matrix", complexity: "O(mnpq)", description: "Kronecker product", formula: "A ⊗ B", status: "implemented" },
  { id: "solveLinearSystem", category: "matrix", complexity: "O(n³)", description: "Linear system solve", formula: "Ax = b", requiresSquare: true, status: "implemented" },
  { id: "solveMultipleRhs", category: "matrix", complexity: "O(n³ + n²k)", description: "Multiple right-hand sides", formula: "AX = B", requiresSquare: true, status: "implemented" },
  { id: "lu", category: "factorization", complexity: "O(n³)", description: "LU factorization with partial pivoting", formula: "PA = LU", requiresSquare: true, status: "implemented" },
  { id: "cholesky", category: "factorization", complexity: "O(n³/3)", description: "Cholesky factorization", formula: "A = LLᵀ", requiresSquare: true, status: "implemented" },
  { id: "ldlt", category: "factorization", complexity: "O(n³/3)", description: "LDLᵀ factorization", formula: "A = LDLᵀ", requiresSquare: true, status: "implemented" },
  { id: "bunchKaufman", category: "factorization", complexity: "O(n³)", description: "Pivoted symmetric-indefinite factorization", formula: "PAPᵀ = LDLᵀ", requiresSquare: true, status: "planned" },
  { id: "qr", category: "factorization", complexity: "O(mn²)", description: "Householder QR factorization", formula: "A = QR", status: "implemented" },
  { id: "svd", category: "factorization", complexity: "O(mn min(m,n))", description: "Singular value decomposition", formula: "A = UΣVᵀ", status: "implemented" },
  { id: "rankRevealing", category: "factorization", complexity: "O(mn²)", description: "Rank-revealing QR factorization", formula: "AP = QR", status: "implemented" },
  { id: "polar", category: "factorization", complexity: "O(n³)", description: "Polar decomposition", formula: "A = UP", status: "implemented" },
  { id: "eigen", category: "eigen", complexity: "O(n³)", description: "Eigenvalues and eigenvectors", formula: "AX = XΛ", requiresSquare: true, status: "planned" },
  { id: "symmetricEigen", category: "eigen", complexity: "O(n³)", description: "Symmetric spectral decomposition", formula: "A = QΛQᵀ", requiresSquare: true, status: "implemented" },
  { id: "schur", category: "eigen", complexity: "O(n³)", description: "Schur decomposition", formula: "A = QTQᵀ", requiresSquare: true, status: "planned" },
  { id: "hessenberg", category: "eigen", complexity: "O(n³)", description: "Hessenberg reduction", formula: "A = QHQᵀ", requiresSquare: true, status: "implemented" },
  { id: "jordan", category: "eigen", complexity: "O(n³)", description: "Jordan canonical form", formula: "A = PJP⁻¹", requiresSquare: true, status: "planned" },
  { id: "frobenius", category: "eigen", complexity: "O(n³)", description: "Frobenius companion form", formula: "A = PFP⁻¹", requiresSquare: true, status: "planned" },
  { id: "hermite", category: "exact", complexity: "domain dependent", description: "Hermite normal form", formula: "A = UH", status: "planned" },
  { id: "smith", category: "exact", complexity: "domain dependent", description: "Smith normal form", formula: "A = UDV", status: "planned" },
  { id: "popov", category: "exact", complexity: "domain dependent", description: "Popov normal form", formula: "A = UP", status: "planned" },
] as const;

export const OPERATION_CATEGORIES: readonly OperationCategory[] = ["level1", "level2", "level3", "matrix", "factorization", "eigen", "exact"];
