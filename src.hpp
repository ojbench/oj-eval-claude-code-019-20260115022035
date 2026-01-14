#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Compute attention for each K[j], V[j] pair and sum the results
    // Initialize result accumulator
    Matrix* accumulator = nullptr;

    for (size_t j = 0; j <= i; ++j) {
      // Make copies to avoid modifying originals
      Matrix* K_copy = matrix_memory_allocator.Allocate("K_copy_" + std::to_string(i) + "_" + std::to_string(j));
      Matrix* V_copy = matrix_memory_allocator.Allocate("V_copy_" + std::to_string(i) + "_" + std::to_string(j));

      // Copy K[j] and V[j] in HBM
      gpu_sim.Copy(keys[j], K_copy, sjtu::kInGpuHbm);
      gpu_sim.Copy(values[j], V_copy, sjtu::kInGpuHbm);

      // Transpose K_copy
      gpu_sim.Transpose(K_copy, sjtu::kInGpuHbm);

      // Move to SRAM for computation
      gpu_sim.MoveMatrixToSharedMem(current_query);
      gpu_sim.MoveMatrixToSharedMem(K_copy);
      gpu_sim.MoveMatrixToSharedMem(V_copy);
      gpu_sim.Run(false, &matrix_memory_allocator);

      // Compute Q * K[j]^T
      Matrix* QK = matrix_memory_allocator.Allocate("QK_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatMul(current_query, K_copy, QK);

      // Apply softmax row-wise
      Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatExp(QK, QK_exp);

      // Normalize each row
      Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax_result_" + std::to_string(i) + "_" + std::to_string(j));
      size_t num_rows = QK_exp->GetRowNum();

      for (size_t row = 0; row < num_rows; ++row) {
        Matrix* row_mat = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(row));
        gpu_sim.GetRow(QK_exp, row, row_mat, sjtu::kInSharedMemory);

        Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(row));
        gpu_sim.Sum(row_mat, row_sum);

        Matrix* normalized_row = matrix_memory_allocator.Allocate("normalized_row_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(row));
        gpu_sim.MatDiv(row_mat, row_sum, normalized_row);

        if (row == 0) {
          softmax_result = normalized_row;
        } else {
          Matrix* temp = matrix_memory_allocator.Allocate("softmax_temp_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(row));
          gpu_sim.Concat(softmax_result, normalized_row, temp, 0, sjtu::kInSharedMemory);
          gpu_sim.ReleaseMatrix(softmax_result);
          softmax_result = temp;
        }
      }

      // Compute softmax_result * V[j]
      Matrix* contribution = matrix_memory_allocator.Allocate("contribution_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatMul(softmax_result, V_copy, contribution);

      gpu_sim.Run(false, &matrix_memory_allocator);

      // Add to accumulator
      if (j == 0) {
        accumulator = contribution;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("acc_temp_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.MatAdd(accumulator, contribution, temp);
        gpu_sim.ReleaseMatrix(accumulator);
        accumulator = temp;
      }
    }

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(accumulator);
    gpu_sim.Run(false, &matrix_memory_allocator);

    rater.CommitAnswer(*accumulator);
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
