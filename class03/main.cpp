#include <iostream>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <thread>

#define MSGSIZE 16
char msg1[] = "hello, world #1";
char msg2[] = "hello, world #2";
char msg3[] = "hello, world #3";
  
int main()
{
    char inbuf[MSGSIZE];
    int p[2], i;
  
    if (pipe(p) < 0)
        exit(1);

    int child_pid = fork();
    if (child_pid > 0) {
      // we are the parent
      write(p[1], msg1, MSGSIZE);
      write(p[1], msg2, MSGSIZE);
      write(p[1], msg3, MSGSIZE);      
    } else {
      // we are the child
      for (i = 0; i < 3; i++) {
        /* read pipe */
        read(p[0], inbuf, MSGSIZE);
        printf("%s\n", inbuf);
      }
    }  
    return 0;
}
