# Assignment 3

In this assignment you will be using MPI, not threads.

In the usual cpsc445 repo, in branch "assignment03".

1) Install a local copy of MPI (sudo apt install mpich) or just use CI
2) Copy the file main.cpp (attached) and try comile it and run it:

   mpiCC main.cpp
   mpirun -n 5 ./a.out

   (+1 point)

3) Fix the condition (sum != n*p) so that the program returns successfully
   (+1 point)

4) Copy main.cpp into dna_count.cpp and modify as follows:

    The program should read from "dna.txt" a string containing ATGC chars.
    It should scatter the string over the parallel processes,
    and it should count, in parallel, the number of As, Ts, Gs, Cs.
    It should output in file "output.txt" the frequency of each char.
    Only node rank==0 should perform input output.
    Exaxmple dna.txt
    GATTACA
    Example output.txt
    A 3
    T 2
    G 1
    C 1

    (+1 points)

5) In a file "README.txt" derive the Speedup formula for the above code as function of tB and tL (assume the fastest network topology for each step)

   (+1 points)

6) Copy dna_count.cpp into dna_invert.cpp and modify the code as follows:

    The program should read from "dna.txt" a string containing ATGC chars.
    It should scatter the string over the parallel processes
    and it should invert the dna sequence (replace A->T, T->A, C->G, G->C)
    It should output in file "output.txt" the inverted string.
    Only node rank==0 should perform input output.
    Exaxmple dna.txt
    GATTACA
    Example output.txt
    CTAATGT

    (+1 points)

7) In the file "README.txt" derive the Speedup formula for the above code as function of tB and tL (assume the fastest network topology for each step)

   (+1 points)

8) Copy the dna_invert into dna_parse.cpp and modify the code as follows:

    The program should read from "dna.txt" a string containing ATGC chars.
    It should break the sequence into triplets (ATGTGATAC.. -> [ATG, TGA, TAC])
    and count the occurrence of each sequence, in parallel.
    It should output in file "output.txt" the frequency of each triplet found
    (in alphabetical order)
    Only node rank==0 should perform input output.
    Example dna.txt
    ATGTGATACATG
    Exmaple.output.txt
    TAC 1
    TGA 1
    ATG 2
    (+2 points)

9) In the file "README.txt" derive the Speedup formula for the above code as function of tB and tL (assume the fastest network topology for each step)

   (+1 points)

10) Extra credit. Copy the dna_parse into dna_genes.cpp and modify the code as follows:

    The program should read from "dna.txt" a string containing ATGC chars.
    It should break the sequence into triplets (ATGTGATAC.. -> [ATG, TGA, TAC])
    and find the start and the end of each gene where a gene starts at ATG
    and stops at TGA, TAG or TAA. For example
    [0] [1] [2] [3] [4] [5] [6] [7] [8]
    ATT ATT ATG TGA TAG ATG TTA TAG ATG
    ... ... ^   ... ... ... ... ^   ... => start at 2, stop at 7
    (caveat ATG can also appear inside a gene, as in [5], ignore it)
    It should output in file "output.txt" the start and stop of each gene.
    Each line should contain the start and stop indexes of a gene,
    separated by a space.
    Only node rank==0 should perform input output.

    (+2 points)
