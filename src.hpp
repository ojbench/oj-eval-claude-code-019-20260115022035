#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Do all operations in HBM to avoid IO/calculation conflicts

    // Build K_all by concatenating K[0], K[1], ..., K[i]
    Matrix* K_all;
    if (i == 0) {
      K_all = keys[0];
    } else {
      K_all = matrix_memory_allocator.Allocate("K_all_" + std::to_string(i));
      gpu_sim.Concat(keys[0], keys[1], K_all, 0, sjtu::kInGpuHbm);
      for (size_t j = 2; j <= i; ++j) {
        Matrix* temp = matrix_memory_allocator.Allocate("K_temp_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(K_all, keys[j], temp, 0, sjtu::kInGpuHbm);
        gpu_sim.ReleaseMatrix(K_all);
        K_all = temp;
      }
    }

    // Build V_all by concatenating V[0], V[1], ..., V[i]
    Matrix* V_all;
    if (i == 0) {
      V_all = values[0];
    } else {
      V_all = matrix_memory_allocator.Allocate("V_all_" + std::to_string(i));
      gpu_sim.Concat(values[0], values[1], V_all, 0, sjtu::kInGpuHbm);
      for (size_t j = 2; j <= i; ++j) {
        Matrix* temp = matrix_memory_allocator.Allocate("V_temp_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(V_all, values[j], temp, 0, sjtu::kInGpuHbm);
        gpu_sim.ReleaseMatrix(V_all);
        V_all = temp;
      }
    }

    // Transpose K_all
    gpu_sim.Transpose(K_all, sjtu::kInGpuHbm);

    // Now move matrices to SRAM for faster computation
    gpu_sim.MoveMatrixToSharedMem(K_all);
    gpu_sim.MoveMatrixToSharedMem(V_all);
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Compute Q * K_all^T
    Matrix* QK = matrix_memory_allocator.Allocate("QK_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_all, QK);

    // Compute softmax row-wise: exp(QK) / sum(exp(QK))
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp_" + std::to_string(i));
    gpu_sim.MatExp(QK, QK_exp);

    // For each row, normalize it
    Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax_result_" + std::to_string(i));
    size_t num_rows = QK_exp->GetRowNum();

    for (size_t row = 0; row < num_rows; ++row) {
      Matrix* row_mat = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.GetRow(QK_exp, row, row_mat, sjtu::kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.Sum(row_mat, row_sum);

      Matrix* normalized_row = matrix_memory_allocator.Allocate("normalized_row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.MatDiv(row_mat, row_sum, normalized_row);

      if (row == 0) {
        softmax_result = normalized_row;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("softmax_temp_" + std::to_string(i) + "_" + std::to_string(row));
        gpu_sim.Concat(softmax_result, normalized_row, temp, 0, sjtu::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_result);
        softmax_result = temp;
      }
    }

    // Compute attention * V_all
    Matrix* result = matrix_memory_allocator.Allocate("result_" + std::to_string(i));
    gpu_sim.MatMul(softmax_result, V_all, result);

    gpu_sim.Run(false, &matrix_memory_allocator);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);
    gpu_sim.Run(false, &matrix_memory_allocator);

    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
