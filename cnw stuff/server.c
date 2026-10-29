#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>

#define PORT 12345

int main()
{
    int server_fd,new_socket;
    struct sockaddr_in address;
    int addrlen=sizeof(address);
    char buffer[1024]={0};
    char *message="Hello from server";

    server_fd=socket(AF_INET,SOCK_STREAM,0);
    address.sin_family=AF_INET;
    address.sin_addr.s_addr=INADDR_ANY;
    address.sin_port=htons(PORT);

    if( bind(server_fd,(struct sockaddr *)&address,sizeof(address))<0)
    {
        perror("BINDING ERROR");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd,3)<0)
    {
        perror("Listen");
        exit(EXIT_FAILURE);
    }

    printf("SERVER IS LISTENING.....")


    new_socket=accept(server_fd,(struct sockaddr*)&addr,(socklen_t*) &addrlen);
    if (new_socket<0)
    {
        perror("Socket Accept");
        exit(EXIT_FAILURE);

    }
    read(new_socket,buffer,1024);
    printf("CLIENT:%s\n",buffer);

    send(new_socket,message,strlen(message),0);
    printf("Message sent to client\n");

    close(new_socket);
    close(server_fd);

    return 0;
}