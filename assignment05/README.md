# Assignment 5

In the usual cpsc445 repo, in branch "assignment05".
All programs in this assignment MUST use CUDA.

1) Implement a sqrt.cu that does the following:

    The program should read from a "input.csv" an arbitrary table of
    floating point numbers, copy it to the device, which should
    compute the sqrt each number, send back to the host, and write the result
    in a csv table "output.csv".

    (+2 points)

2) Implement a sqrt_shared.cu that does the following:

    The same task as 1) but the device copies input to shared memory
    before computing the sqrt in place and then copy the result from
    shared memory to global device memory.

    (+2 points)


3)  Implement a extreme.cu that does the following:

    The program should read from a "input.csv" an arbitrary table of
    floating point numbers p, copy it to the device, and the device would
    find all points i,j where p(i,j) is greater or smaller than its 8 neighbors.
    The program would then return the set of points in a csv table "output.csv".
    The points in the output should be sorted according to i * N + j.
    The sorting can be done on host.

    (+2 points)

4) Modify the above program so that each block returns a set of points i,j already sorted
   (only sorting points processed by the block). Use the merge function on device to find
   the total sorted sequence.

   (+2 points)