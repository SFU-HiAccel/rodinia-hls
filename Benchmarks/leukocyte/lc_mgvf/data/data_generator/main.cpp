#include<stdlib.h>
#include<stdio.h>


int main()
{

FILE* fid = fopen("input.data","w");

fprintf(fid, "%%%%\n");

for (int i = 0; i<1024*1024; i++){
fprintf(fid, "%.18f\n", (float)i / (float)(1024*1024));
}

fprintf(fid, "%%%%\n");

for (int i = 0; i<1024*1024; i++){
fprintf(fid, "%.18f\n", (float)i / (float)(1024*1024));
}

return 0;

}


