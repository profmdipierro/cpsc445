# Assignment 4

In the usual cpsc445 repo, in branch "assignment04".
All programs in this assignment MUST use CUDA.

1) Implement a dna_invert.cu that does the following:

    The program should read from "dna.txt" a string containing ATGC chars.
    It should scatter the string over the parallel processes
    and it should invert the dna sequence (replace A->T, T->A, C->G, G->C)
    It should output in file "output.txt" the inverted string.
    Only node rank==0 should perform input output.
    Exaxmple dna.txt
    GATTACA
    Example output.txt
    CTAATGT

    (+2 points)

2)  Implement a dna_count.cu that does the following:

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

    (+3 points)

3) Implment a dna_parse.cu that does the following:

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

   (+3 points)

4) Extra credit. Implement a dna_genes.cu that does the following:

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

    (+2 points)
