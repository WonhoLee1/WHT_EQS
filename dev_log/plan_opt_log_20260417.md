# Optimization Log Output Alignment Plan (2026-04-17)

## 1. Goal
Improve the readability of the optimization loop output by aligning columns and ensuring consistent widths for values.

## 2. Current Status
The current output lacks proper alignment, making it difficult to track changes across iterations:
```
🔄 [Iter 0141/150] 1.27s | Loss: 7.15761e+00 | dt:0.002406(avg:0.0001748) | dz:0.0040(avg:0.00054)
```

## 3. Improvements
1. **Consistency in Widths**: Define fixed widths for iteration index, time, loss, and sensitivity parameters.
2. **Proper Formatting**: Use f-string alignment features (e.g., `:8.6f`, `:12.5e`).
3. **Structured Physical Stats**: Align the "Physical Stats" line that appears periodically.

## 4. Task List
- [ ] Modify `sens_info` construction to use fixed widths.
- [ ] Modify the main iteration `print` statements to use fixed widths.
- [ ] Modify "Physical Stats" print statement for alignment.

## 5. Execution
I will apply these changes to `main_shell_opt.py`.
