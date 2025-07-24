#include "tensorflow/c/tf_tensor.h"
#include "klee/klee.h"
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

// Needed for klee_assert to compile correctly
extern "C" void __assert_fail(const char *assertion, const char *file,
                              unsigned int line, const char *function);

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
