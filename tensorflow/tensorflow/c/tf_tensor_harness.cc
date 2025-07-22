#include "tensorflow/c/tf_tensor.h"
#include "klee/klee.h"
#include <stdlib.h>
#include <stdint.h>

void dummy_deallocator(void* data, size_t len, void* arg) {}

int main() {
  int64_t dims[2] = {2, 2};
  int num_dims = 2;

  size_t len = 4 * sizeof(float);
  float* data = (float*)malloc(len);
  klee_make_symbolic(data, len, "tensor_data");

  TF_Tensor* t = TF_NewTensor(
      TF_FLOAT, dims, num_dims,
      data, len,
      dummy_deallocator, NULL
  );

  // Optional: Add KLEE assertions
  klee_assert(t != NULL);

  return 0;
}
