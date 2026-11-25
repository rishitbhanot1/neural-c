# Multilayer Perceptron in C

An implementation of a two-layer dense feedforward neural network written in C using only the C Standard Library (`stdio.h`, `stdlib.h`, `math.h`, `string.h`, `time.h`). The program parses $28 \times 28$ grayscale digit images from CSV files, trains a $784 \rightarrow 300 \rightarrow 10$ network via online stochastic gradient descent (SGD) and backpropagation, evaluates classification accuracy on a held-out test set, and serializes weights and diagnostics to disk.

## Module Breakdown

```
.
├── matrix.h / matrix.c         # Matrix struct, heap allocation, flattening, argmax, initialization
├── operations.h / operations.c # Matrix multiplication, element-wise arithmetic, transpose, apply
├── activation.h / activation.c # Scalar sigmoid, post-activation derivative, stabilized softmax
├── img.h / img.c               # CSV parser and 28x28 image lifecycle management
├── network.h / network.c       # Network state, forward/backward passes, serialization, exports
├── main.c                      # Training loop, learning rate schedule, evaluation driver
├── mnist_train.csv             # Training dataset (10,000 rows loaded)
└── mnist_test.csv              # Evaluation dataset (3,000 rows loaded)
```

## Dataset

* **Source:** [MNIST in CSV (Kaggle)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
* **Format and Schema:** Plain-text CSV files (`mnist_train.csv` and `mnist_test.csv`) containing a single header row followed by 785 comma-separated integer fields per record: column `0` holds the ground-truth digit class in $[0, 9]$, and columns `1` through `784` hold row-major $28 \times 28$ grayscale pixel intensities in $[0, 255]$.
* **Subset Configuration:** `main.c` ingests the first `10,000` records from `mnist_train.csv` for training and the first `3,000` records from `mnist_test.csv` for testing.

## Data Structures and Memory Model

### `Matrix` (`matrix.h`)

```c
typedef struct {
    double **data;
    int rows;
    int cols;
} Matrix;
```

* **Heap Layout:** `mtx_create(row, col)` allocates a `Matrix` struct, an array of `row` row-pointers (`double**`), and `row` separate contiguous blocks of `col` `double` values. Elements are indexed as `m->data[i][j]`.
* **Allocation Semantics:** Every function in `operations.c` (`dot`, `add`, `subtract`, `multiply`, `scale`, `transpose`, `apply`) allocates and returns a new `Matrix*` rather than mutating inputs in place. Callers are responsible for explicitly releasing intermediate matrices with `mtx_free`.

### `Img` (`img.h`)

```c
typedef struct {
    Matrix* img_data;
    int label;
} Img;
```

* Datasets are stored as heap-allocated pointer arrays (`Img**`). Because each image is referenced by pointer, epoch shuffling (`shuffle_imgs`) swaps 8-byte pointers in $O(n)$ time without copying the underlying $28 \times 28 \times 8 = 6{,}272$-byte pixel buffers.

### `NeuralNetwork` (`network.h`)

```c
typedef struct {
    int input;
    int hidden;
    int output;
    double learning_rate;
    Matrix* hidden_weights;
    Matrix* output_weights;
} NeuralNetwork;
```

* Parameter storage consists of two dense weight matrices without bias vectors:
  * `hidden_weights`: $300 \times 784$ ($235{,}200$ `double` parameters)
  * `output_weights`: $10 \times 300$ ($3{,}000$ `double` parameters)
  * **Total:** $238{,}200$ parameters ($1{,}905{,}600$ bytes in IEEE-754 64-bit floating-point representation).

## Processing Pipeline

### 1. CSV Ingestion and Normalization (`img.c`)

`csv_to_imgs` opens the target CSV file, discards the first header line via `fgets`, and reads up to `number_of_imgs` lines into a fixed stack buffer (`char row[10000]`). Each line is tokenized on `","` using `strtok`:

* Token index `0` is converted via `atoi` and assigned to `img->label`.
* Token indices `1` through `784` are mapped onto the $28 \times 28$ matrix using integer division and modulo arithmetic:

  $$
  \text{row} = \left\lfloor \frac{j - 1}{28} \right\rfloor, \qquad \text{col} = (j - 1) \bmod 28
  $$

* Raw integer pixel intensities in $[0, 255]$ are divided by `255.0` to normalize inputs into $[0.0, 1.0]$.

### 2. Weight Initialization (`matrix.c`)

`mtx_rnd(m, n)` populates weight matrices with uniform pseudo-random values bounded by $\left[-\frac{1}{\sqrt{n}}, +\frac{1}{\sqrt{n}}\right]$, where $n$ is the row dimension (output neuron count) of the layer ($n = 300$ for `hidden_weights`, $n = 10$ for `output_weights`).

Sampling is performed by `udiff(low, high)`, which discretizes the interval $(\text{high} - \text{low})$ by a scale factor of `10000`, takes `rand() % scaled_difference`, and rescales back to `double`, yielding step increments of $10^{-4}$.

### 3. Forward Pass and Inference (`network.c`, `activation.c`)

Before entering the network, `mtx_flatten(img_data, 0)` reshapes the $28 \times 28$ image into a $784 \times 1$ column vector $X$ in row-major order (`i * cols + j`).

Signal propagation proceeds through two matrix-vector products and element-wise Sigmoid activations:

$$
Z_1 = W_{\text{hidden}} \cdot X \in \mathbb{R}^{300 \times 1}, \qquad A_1 = \sigma(Z_1) = \frac{1}{1 + e^{-Z_1}}
$$

$$
Z_2 = W_{\text{output}} \cdot A_1 \in \mathbb{R}^{10 \times 1}, \qquad A_2 = \sigma(Z_2) = \frac{1}{1 + e^{-Z_2}}
$$

* **Training (`network_train`):** Execution stops at $A_2$ to compute the error against the one-hot target vector.
* **Inference (`network_predict`):** $A_2$ is passed through `softmax`. To prevent `exp()` overflow, `softmax` first finds $m = \max_{i}(A_{2,i})$ and computes:

  $$
  P(y = i \mid X) = \frac{e^{A_{2,i} - m}}{\sum_{k=0}^{9} e^{A_{2,k} - m}}
  $$

  `mtx_argmax` then scans the $10 \times 1$ probability vector and returns the row index with the maximum value.

### 4. Backpropagation and Weight Updates (`network.c`)

`network_train_batch_imgs` iterates sequentially through the dataset ($\text{batch size} = 1$). For each sample, it allocates a $10 \times 1$ target vector $Y$, initializes all entries to `0` via `mtx_fill`, and sets $Y_{\text{label}, 0} = 1$.

1. **Output Error and Local Gradient:**

   $$
   e_{\text{out}} = Y - A_2
   $$

   `sigmoidPrime(A_2)` computes $\sigma'(Z_2) = A_2 \odot (1 - A_2)$ directly from the cached post-activation matrix $A_2$, avoiding redundant `exp()` evaluations:

   $$
   \delta_{\text{out}} = e_{\text{out}} \odot \big(A_2 \odot (1 - A_2)\big) \in \mathbb{R}^{10 \times 1}
   $$

2. **Hidden Error and Local Gradient:**

   $$
   e_{\text{hidden}} = W_{\text{output}}^T \cdot \delta_{\text{out}} \in \mathbb{R}^{300 \times 1}
   $$

   $$
   \delta_{\text{hidden}} = e_{\text{hidden}} \odot \big(A_1 \odot (1 - A_1)\big) \in \mathbb{R}^{300 \times 1}
   $$

3. **Parameter Update:**
   Outer products of the local gradients and transposed layer inputs yield the weight deltas, scaled by learning rate $\eta$:

   $$
   W_{\text{output}} \leftarrow W_{\text{output}} + \eta \big(\delta_{\text{out}} \cdot A_1^T\big)
   $$

   $$
   W_{\text{hidden}} \leftarrow W_{\text{hidden}} + \eta \big(\delta_{\text{hidden}} \cdot X^T\big)
   $$

   Each weight update allocates a new weight matrix, frees the previous weight matrix pointer, and reassigns `net->output_weights` and `net->hidden_weights`. A single call to `network_train` performs 18 intermediate `Matrix*` allocations and 18 corresponding `mtx_free` calls.

### 5. Training Schedule (`main.c`)

* **Dataset Split:** `10,000` training records (`mnist_train.csv`) and `3,000` test records (`mnist_test.csv`).
* **Epoch Loop:** Runs for `15` epochs (`150,000` total gradient updates). At the start of each epoch, `shuffle_imgs` applies an in-place Fisher-Yates shuffle to the `train_imgs` pointer array.
* **Learning Rate Decay:** Initialized at $\eta_0 = 0.1$ and multiplied by `0.92` at the end of each epoch ($\eta_t = 0.1 \times 0.92^t$, reaching $\approx 0.0286$ on epoch 15).

## Output File Specifications

Running the compiled binary generates four files in the working directory:

1. **`mlpneuralc.nn` (Binary Model Archive — $1{,}905{,}616$ bytes)**
   * **Bytes `0..3`:** 4-byte ASCII signature `{'N', 'N', '0', '1'}` verified by `memcmp` in `network_load`.
   * **Bytes `4..15`:** Three 32-bit integers (`input`, `hidden`, `output`).
   * **Bytes `16..1,881,615`:** `hidden_weights` written row-by-row as raw IEEE-754 `double` values ($300 \times 784 \times 8$ bytes).
   * **Bytes `1,881,616..1,905,615`:** `output_weights` written row-by-row as raw IEEE-754 `double` values ($10 \times 300 \times 8$ bytes).

2. **`trained_weights.h` (C Source Header)**
   * Defines preprocessor constants `NN_INPUT`, `NN_HIDDEN`, and `NN_OUTPUT` alongside `static const double HIDDEN_WEIGHTS[300][784]` and `static const double OUTPUT_WEIGHTS[10][300]` formatted with `%.12g` precision for direct inclusion in standalone C inference builds.

3. **`features.pgm` (ASCII Portable GrayMap — $580 \times 435$ pixels)**
   * Arranges all `300` rows of `hidden_weights` into a $20 \times 15$ grid of $29 \times 29$ tiles ($28 \times 28$ weight pixels plus a 1-pixel `0`-valued border on the right and bottom edges). Each neuron's 784 weights are independently min-max normalized to integers in `[0, 255]` under the `P2` header.

4. **`errors.csv` (Misclassification Log)**
   * Writes a header (`image_index,true_label,predicted_label,confidence`) followed by one row for each test sample where `mtx_argmax(prediction) != img->label`, recording the post-softmax probability of the predicted class.

## Compilation and Execution

### Windows (PowerShell)

```powershell
gcc -O3 -march=native -Wall matrix.c operations.c activation.c img.c network.c main.c -lm -o mlpneuralc.exe
.\mlpneuralc.exe
```

### Linux / macOS

```bash
gcc -O3 -march=native -Wall matrix.c operations.c activation.c img.c network.c main.c -lm -o mlpneuralc
./mlpneuralc
```

## Architectural Constraints and Trade-Offs

* **Absence of Bias Vectors:** Transformations compute $W \cdot X$ rather than $W \cdot X + b$, restricting every neuron's separating hyperplane to pass through the origin in input/hidden space.
* **Sigmoid Gradient Attenuation:** Using Sigmoid in hidden layers bounds local derivatives to $\sigma'(z) \le 0.25$. Adding further hidden layers under this activation scheme attenuates gradients exponentially during backpropagation.
* **Jagged Array Memory Layout:** Storing `Matrix->data` as `double**` places each matrix row in a separate heap allocation rather than a single contiguous buffer, reducing spatial cache locality during matrix multiplication (`dot`).
* **Per-Step Heap Churn:** Non-mutating operations in `operations.c` trigger 18 `malloc`/`free` cycles per training sample rather than reusing pre-allocated scratchpad buffers.