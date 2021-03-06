Assignment 1
============

## Due Date

September 26, 2021 at 11:59pm (Pacific Time)

Late assignment not accepted.

## Assignment 1

- Fill form https://forms.gle/9k74SUd66FSL2pidA
- Create a private github repo with name "cpsc445" lower case
- Give me private access to the repo by going to [settings][manage access][invite collaborator] and invite me: "profmdipierro"
- Create a branch calls "assignment01"
- In the above branch, in the root folder of the repo, write a C++ called "life.cpp" that implements Conway's Game of Life using threads to parallelize execution.

- The program must take 4 command line arguments: the input filename, the output filename, the number of steps, the number of threads.

- The input file will be a test files with rows of 0 and 1, each row ending in newline. The output must follow the same format.
- For number of steps == 0 the output should be the same as the input.
- You can assume 1 <= number of threads <= number of rows.
- The program must use a double buffer: at each step compute a new board and then switch them.
- Must work with this example:

```
g++ life.cpp -lpthread -o life.exe
cat > input.txt <<EOF
0000000001
0000010001
0000101001
0000000000
0000000000
EOF
./file.exe input.txt output.txt 3 5
cat output.txt
```

## About Grading

This assignment counts for a total of 8 class points allocated as follows:
- 0 if the program does not compile
- +1 if the program compiles and runs the above example and produces correct output for num threads = 1
- +1 if the program compiles and runs the above example and produces correct output for num threads = 1 and arbitrary number of steps
- +1 if the program compiles and runs the above example and can detect and report invaild input (should output "invalid input") and invalid command line arguments (should output "invalid arguments")
- +1 if the program compiles and runs the above example and produces correct output of num threads = 2 and abitrary number of steps.
- +1 if the program compiles and runs the above example and produces correct output of num threads = number of columns and arbitrary number of steps.
- +1 if the program compiles and runs the above example and produces correct output for arbitrary input and arbitrary number of threads and arbitrary number of steps.
- +1 if the program is documented
- +1 has no memory leaks or other memory issues.
- +1 (extra credit) if your program uses a state machine and is well organized.

## About collaboration

You cannot share the program with each other but you are encouraged to share input and output files and give each other coding advice.
