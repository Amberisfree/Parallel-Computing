# Parallel-Computing


## Part1-OpenMP:
The problem of parallelization primary comes with overhead, poor memory access patterns, and data race. To address these problem, we apply loop optimization to improve the use of memory, specify data environment to avoid data race, and add nowait to decrease the unnecessary time spent by threads waiting at implicit barrier.

## Part2-MPI:

To optimize the performance of MPI, it is crucial to minimize communication in parallel programs. This involves reducing data transfer among processes and optimizing communication patterns to minimize global synchronization while promoting local communication.
Our primary parallel strategy focuses on reducing unnecessary data transfer through techniques such as data partition and halo exchange communication pattern. This approach has significantly decreased the communication cost, especially in sharing boundary values between neighboring processes.
