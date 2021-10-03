Assignment 2
============

## Due Date

September 12, 2021 at 11:59pm (Pacific Time)

Late assignment not accepted.

## Assignment 2

As for assignment 01, your work must be in your private repo cpsc445 and in branch "assignment02".

Write a program called search.cpp that take four command line arguments:
- filename 1 pointing to a list of keywords (one per line)
- filename 2 pointing to some text (arbirarily long text)
- filename 3 which will be the output file created by your program
- the number of threads to be used.

Your program must count how many times each of the keywords in filename 1 is mentioned in filename 2. The output must contain the same list of keywords, one per row, sorted alphabetically, followed by a space and the relative count.

filename 1 contains multiple lines of text. Parallelize the code so that line k is processed by thread rank = k % p.

Example:

```
g++ search.cpp -lpthread
cat > keywords.txt <<EOF
dog
mouse
hippo
EOF
cat > sometext.txt <<EOF
The mouse was a friend of the dog.
The mouse was in the zoo with the hippo.
The hippo was a very big hippo.
EOF
./a.out keywords.txt sometext.txt output.txt 2
cat output.txt
dog 1
hippo 3
mouse 2
```

## About Grading

This assignment counts for a total of 8 class points allocated as follows:
- 0 if the program does not compile or cannot be retrieved
- +2 if the program works with the above example
- +2 if the program works with arbitrary generated examples
- +2 if the program is parallelized as described.
- +1 if the program has no concurrency issues/race conditions/memory leaks.
- +1 if the program is well organized (uses classes, no computations in main, separation of input logic from computation logic, functions are commented)

## About collaboration

You cannot share the program with each other but you are encouraged to share input and output files and give each other coding advice.
