# Parallel Canny edge detector

Final project for the Parallel Programming course at Konkuk University (October to December 2025): make a Canny edge detector fast with OpenMP while keeping its output within 1% of a reference implementation.

The course gave us `main.cpp`, which handles PGM input and output, runs the single-threaded reference, times both versions and checks the result. Everything in `student_kernel.hpp` is my work.

## Results

2000×2000 input on my home server (NucBox K8 Plus), built with `g++ -fopenmp -march=native`:

| Threads | Time |
|---|---|
| 1 | 97.43 ms |
| 8 | 17.46 ms |
| 18 | 22.22 ms |

That is about 5.6 times faster at 8 threads, and the output passed the 1% mismatch check. With more than 8 threads it got slower again. On a 10000×10000 input, 16 threads took 559.97 ms where the single-threaded reference took 15,745.7 ms. All of these are single runs copied from the terminal, not averages.

## What the kernel does

All stages are parallelized with OpenMP.

1. Gaussian blur, split into a horizontal and a vertical 5-tap pass. The kernel `[1 4 6 4 1]` is symmetric, so mirrored pixels are added before multiplying, and the loaded pixels shift through registers as the window moves along the row.
2. Sobel gradient, magnitude and direction in a single loop, so the `gx` and `gy` images are never stored. Direction is binned into four angles by comparing against tan(22.5°) and tan(67.5°) instead of calling `atan2`.
3. Non-maximum suppression with the two neighbor offsets looked up from a table by direction, which removes the branch on direction for interior pixels. Border pixels take a separate, bounds-checked path.
4. Double threshold, with high = mean + standard deviation of the thinned magnitudes and low = 0.4 × high. Mean and variance come from an OpenMP reduction.
5. Hysteresis. The reference follows weak edges with a serial stack. I changed it to a breadth-first search done in waves: strong pixels form the first wave, each thread expands its part of the wave into its own list, and a weak neighbor is taken with a compare-and-swap from 1 to 255. Only the thread whose swap succeeds adds the pixel, so nothing is added twice. The lists are merged into the next wave until no wave is left.

## Build and run

```bash
g++ -O2 -fopenmp -march=native main.cpp -o canny
./canny input.pgm output.pgm 8      # input, output, thread count
```

Input and output are 8-bit binary PGM (P5).
