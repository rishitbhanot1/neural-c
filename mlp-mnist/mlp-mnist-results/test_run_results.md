# Test Run Results

Evaluation log and quantitative breakdown for the $784 \rightarrow 300 \rightarrow 10$ multilayer perceptron trained on `mnist_train.csv` and evaluated on `mnist_test.csv`.

## 1. Run Configuration

| Parameter | Value |
| :--- | :--- |
| **Topology** | $784 \rightarrow 300 \rightarrow 10$ (no bias vectors) |
| **Trainable Parameters** | $238{,}200$ (`double`, 64-bit IEEE-754) |
| **Training Set** | $10{,}000$ samples (`mnist_train.csv`) |
| **Test Set** | $3{,}000$ samples (`mnist_test.csv`) |
| **Epochs** | $15$ ($150{,}000$ total SGD updates) |
| **Batch Size** | $1$ (online SGD with per-epoch Fisher-Yates shuffle) |
| **Learning Rate Schedule** | $\eta_0 = 0.1$, exponential decay $\gamma = 0.92$ per epoch ($\eta_{15} \approx 0.0311$) |
| **Compiler Flags** | `gcc -O3 -march=native -Wall` |

## 2. Quantitative Evaluation

| Metric | Baseline (1 Epoch, Unshuffled) | Current Run (15 Epochs, Shuffled) | Delta |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | $0.90000$ ($90.00\%$) | $0.94567$ ($94.57\%$) | $+4.57\%$ |
| **Correct Predictions** | $2{,}700 / 3{,}000$ | $2{,}837 / 3{,}000$ | $+137$ samples |
| **Misclassifications** | $300 / 3{,}000$ | $163 / 3{,}000$ | $-137$ samples ($-45.67\%$ relative error) |

## 3. Sample Inference Analysis

The first five sequential records in `mnist_test.csv` were evaluated and rendered at $28 \times 14$ character resolution (vertical stride of $2$ rows to preserve terminal aspect ratio):

| Sample Index | Ground Truth | Predicted Class | Softmax Confidence | Result |
| :--- | :--- | :--- | :--- | :--- |
| `#0` | `7` | `7` | $23.2\%$ | Correct |
| `#1` | `2` | `2` | $22.6\%$ | Correct |
| `#2` | `1` | `1` | $22.6\%$ | Correct |
| `#3` | `0` | `0` | $23.0\%$ | Correct |
| `#4` | `4` | `4` | $23.1\%$ | Correct |

### Mathematical Note on the $\approx 23.2\%$ Confidence Ceiling

All five confident predictions exhibit post-softmax probabilities tightly clustered between $22.6\%$ and $23.2\%$. This upper bound is an expected consequence of applying Sigmoid ($\sigma(z) \in (0, 1)$) to `final_inputs` prior to `softmax` in `network_predict`:

* Because each output activation $A_{2,i}$ is bounded within $(0, 1)$, the input logits to `softmax` cannot exceed $1.0$ or drop below $0.0$.
* In the theoretical limit where the target neuron saturates at $A_{2,\text{target}} \rightarrow 1.0$ and all $9$ remaining neurons approach $A_{2,k} \rightarrow 0.0$, the maximum attainable Softmax probability is:

$$P_{\max} = \frac{e^{1}}{e^{1} + 9e^{0}} = \frac{e}{e + 9} \approx 0.23197 \quad (23.20\%)$$

Sample `#0` ($23.2\%$) operates within $0.03\%$ of this mathematical ceiling, indicating strong class separation despite the compressed probability scale.

### Captured Terminal Output

```text
Sample #0 | True: 7 | Predicted: 7 (23.2%)
                            
                            
                            
                            
      #%%%%%********+.      
            : :::: %%-      
                  %@:       
                :%%:        
                #%:         
              :%%:          
             #%#.           
           .#%=             
          :%%%.             
          =%#               

Sample #1 | True: 2 | Predicted: 2 (22.6%)
                            
                            
         +%%%%%%#.          
       .%%#.   #%=          
        .     #%%:          
            :%%%:           
           *%%+             
          *%%=              
         #%%=               
        %%%                 
        %%%%%%%%+++%%%%%%%= 
         ====+%%%+==.       
                            
                            

Sample #2 | True: 1 | Predicted: 1 (22.6%)
                            
                            
                .%-         
                =%          
               :%:          
              .%#           
              =%:           
             =@+            
             %#             
            +%+             
           :%%:             
           #%=              
                            
                            

Sample #3 | True: 0 | Predicted: 0 (23.0%)
                            
                            
             +%#.           
            *%%%-           
          %%%%%%%%#.        
        :#%%%%+::=%%-       
       .%%%*      -%%%.     
       .%%        .#@%+     
       #%%       :%%%%.     
       #%%     *#%%%*       
       -%%%%%%%%%%%+        
        :+%%%%%%-           
                            
                            

Sample #4 | True: 4 | Predicted: 4 (23.1%)
                            
                            
                            
          =%       ++       
         :%=       =%       
        +%.        @%       
       +%-       .%%:       
       %+        +%+        
       +%#::::=+*%%-        
         .::::  +%%         
                =%%         
                =%%         
                *:          
```

## 4. Generated Artifacts

| File | Format | Size / Dimensions | Contents |
| :--- | :--- | :--- | :--- |
| `mnist_model.nn` | Custom Binary (`NN01`) | $1{,}905{,}616$ bytes | 16-byte header + raw IEEE-754 `double` weight arrays |
| `trained_weights.h` | C99 Header | $238{,}200$ literals | `static const double` `HIDDEN_WEIGHTS` and `OUTPUT_WEIGHTS` |
| `features.bmp` | 24-bit Uncompressed BMP | $580 \times 435$ px ($757{,}034$ bytes) | $20 \times 15$ grid of $28 \times 28$ min-max normalized hidden neuron receptive fields |
| `errors.csv` | Plain-Text CSV | $164$ lines ($1$ header + $163$ rows) | Index, true label, predicted label, and confidence for every misclassified test sample |

## 5. Diagnostic Takeaways for Next Iteration

1. **Uncalibrated Output Probabilities:** Removing the redundant Sigmoid activation on `final_inputs` and computing $\text{Softmax}(Z_2)$ directly in both training (with cross-entropy gradient $Y - P$) and inference will allow logits to span $(-\infty, +\infty)$, spreading predicted confidences across $[0, 1]$.
2. **Error Distribution Audit:** Analyzing `errors.csv` ($163$ rows) by `(true_label, predicted_label)` pairs will identify specific digit confusions (such as `4`/`9` or `3`/`8`) and reveal whether high-confidence errors stem from ambiguous handwriting or missing bias offsets.