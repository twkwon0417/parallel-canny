# Parallel Canny Edge Detector

An OpenMP implementation of the Canny edge detector in C++, written as the final project of the Parallel Programming course at Konkuk University (Oct – Dec 2025).

The course provided `main.cpp`: PGM input/output, a single-threaded reference implementation, timing, and a PASS/FAIL check against the reference. My work is the optimized kernel in `student_kernel.hpp`.

## Result

On a 2000×2000 input, measured on my home server (NucBox K8 Plus, `g++ -fopenmp -march=native`):

| Threads | Time |
|---|---|
| 1 | 97.43 ms |
| 8 | **17.46 ms** (about 5.6× faster) |
| 18 | 22.22 ms |

The output passed the course check (at most 1% of pixels differing from the reference). Past 8 threads the run got slower again.

On a 10000×10000 input, 16 threads took 559.97 ms against 15,745.7 ms for the single-threaded reference.

These are single runs recorded from the terminal, not averages.

## Pipeline and optimizations

Every stage runs in parallel with OpenMP.

1. **Gaussian blur (5×5).** Split into a horizontal and a vertical 5-tap pass. The symmetric kernel `[1 4 6 4 1]` lets each pass add mirrored pixels before multiplying, and a sliding window reuses loaded values across neighboring outputs.
2. **Sobel gradient, magnitude, and direction in one pass.** The three steps are fused, so the separate `gx` and `gy` images are never written to memory. Direction is quantized to four bins with a tan(22.5°) comparison instead of `atan2`.
3. **Non-maximum suppression.** The inner region looks up neighbor offsets in a table indexed by direction, which removes the per-pixel branch on direction. Image borders go through a separate path with bounds checks.
4. **Double threshold.** Thresholds adapt to the image: high = mean + standard deviation of the thinned magnitudes, low = 0.4 × high. The mean and variance come from an OpenMP reduction.
5. **Hysteresis as a parallel wavefront BFS.** The reference follows weak edges with a serial stack. I replaced it with a breadth-first search in waves:
   - strong pixels form the first wave
   - each thread expands its share of the current wave into a private list for the next wave
   - a weak pixel joins the edge only if a compare-and-swap changes it from 1 to 255, so each weak pixel is claimed by exactly one thread and is never added twice
   - the private lists are merged and the next wave starts, until no wave is left

## Build and run

```bash
g++ -O2 -fopenmp -march=native main.cpp -o canny
./canny input.pgm output.pgm 8      # input, output, number of threads
```

Input and output are 8-bit binary PGM (P5) images.

## Files

| File | Contents |
|---|---|
| `main.cpp` | Course-provided driver: PGM I/O, reference implementation, timing, verification |
| `student_kernel.hpp` | My optimized Canny implementation |
