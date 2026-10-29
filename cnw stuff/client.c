#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 12345

int main(){

    int sock;
    struct sockaddr_in serv_addr;
    char* message="Hello from client";
    char buffer[1024]={0};

    sock=socket(AF_INET,SOCK_STREAM,0);
    serv_addr.sin_family=AF_INET;
    serv_addr.sin_port=htons(PORT);
    serv_addr.sin_addr.s_addr=inet_addr("127.0.0.1");

    if (connect(sock,(struct sockaddr*)&serv_addr,sizeof(serv_addr))<0){
        perror("CONNECT");
        exit(EXIT_FAILURE);
    }
    send(sock,message,strlen(message),0);
    



    return 0;
}