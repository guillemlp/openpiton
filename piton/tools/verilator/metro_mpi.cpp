#include <iostream>
#include <mpi.h>

using namespace std;

unsigned long long message_async;
MPI_Status status_async;
MPI_Request request_async;

const int nitems=2;
int          blocklengths[2] = {1,1};
MPI_Datatype types[2] = {MPI_UNSIGNED_SHORT, MPI_UNSIGNED_LONG_LONG};
MPI_Datatype mpi_data_type;
MPI_Aint     offsets[2];

const int nitems_noc=6;
int          blocklengths_noc[6] = {1,1,1,1,1,1};
MPI_Datatype types_noc[6] = {MPI_UNSIGNED_SHORT, MPI_UNSIGNED_SHORT, MPI_UNSIGNED_SHORT,
                         MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};
MPI_Datatype mpi_noc_type;
MPI_Aint     offsets_noc[6];



typedef struct {
    unsigned short valid;
    unsigned long long data;
} mpi_data_t;

typedef struct {
    unsigned long long data_0;
    unsigned long long data_1;
    unsigned long long data_2;
    unsigned short valid_0;
    unsigned short valid_1;
    unsigned short valid_2;
} mpi_noc_t;

void initialize(){
    MPI_Init(NULL, NULL);
    cout << "initializing" << endl;
    
    // Initialize the struct data&valid
    offsets[0] = offsetof(mpi_data_t, valid);
    offsets[1] = offsetof(mpi_data_t, data);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_data_type);
    MPI_Type_commit(&mpi_data_type);

}

// MPI finish functions
unsigned short mpi_receive_finish(){
    unsigned short message;
    int message_len = 1;
    //cout << "[DPI CPP] Block Receive finish from rank: " << origin << endl << std::flush;
    MPI_Bcast(&message, message_len, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
    /*if (short(message)) {
        cout << "[DPI CPP] finish received: " << std::hex << (short)message << endl << std::flush;
    }*/
    return message;
}

void mpi_send_finish(unsigned short message, int rank){
    int message_len = 1;
    if (message) {
        cout << "[DPI CPP] Sending finish " << std::hex << (int)message << " to All" << endl << std::flush;
    }
    MPI_Bcast(&message, message_len, MPI_UNSIGNED_SHORT, rank, MPI_COMM_WORLD);
}

// MPI Yummy functions
unsigned short mpi_receive_yummy(int origin, int flag){
    unsigned short message;
    int message_len = 1;
    MPI_Status status;
    //cout << "[DPI CPP] Block Receive YUMMY from rank: " << origin << endl << std::flush;
    MPI_Recv(&message, message_len, MPI_UNSIGNED_SHORT, origin, flag, MPI_COMM_WORLD, &status);
    /*if (short(message)) {
        cout << "[DPI CPP] Yummy received: " << std::hex << (short)message << endl << std::flush;
    }*/
    return message;
}

void mpi_send_yummy(unsigned short message, int dest, int rank, int flag){
    int message_len = 1;
    /*if (message) {
        cout << "[DPI CPP] Sending YUMMY " << std::hex << (int)message << " to " << dest << endl << std::flush;
    }*/
    MPI_Send(&message, message_len, MPI_UNSIGNED_SHORT, dest, flag, MPI_COMM_WORLD);
}

// MPI data&Valid functions
void mpi_send_data(unsigned long long data, unsigned char valid, int dest, int rank, int flag){
    int message_len = 1;
    mpi_data_t message;
    //cout << "valid: " << std::hex << valid << std::endl;
    message.valid = valid;
    message.data  = data;
    /*if (valid) {
        cout << flag << " [DPI CPP] Sending DATA valid: " << flag << " " << std::hex << (int)message.valid << " data: " << std::hex << message.data << " to " << dest << endl;
    }*/
    MPI_Send(&message, message_len, mpi_data_type, dest, flag, MPI_COMM_WORLD);
}

unsigned long long mpi_receive_data(int origin, unsigned short* valid, int flag){
    int message_len = 1;
    MPI_Status status;
    mpi_data_t message;
    //cout << flag << " [DPI CPP] Blocking Receive data rank: " << origin << endl << std::flush;
    MPI_Recv(&message, message_len, mpi_data_type, origin, flag, MPI_COMM_WORLD, &status);
    /*if (message.valid) {
        cout << flag << " [DPI CPP] Data Message received: " << (short) message.valid << " " << std::hex << message.data << endl << std::flush;
    }*/
    *valid = message.valid;
    return message.data;
}

// MPI Send 3 NoC messages
void mpi_send_noc(unsigned long long data_0, unsigned char valid_0, unsigned long long data_1, unsigned char valid_1, unsigned long long data_2, unsigned char valid_2, int dest, int rank, int flag){
    int message_len = 1;
    mpi_noc_t message;
    //cout << "valid: " << std::hex << valid << std::endl;
    message.valid_0 = valid_0;
    message.data_0  = data_0;
    message.valid_1 = valid_1;
    message.data_1  = data_1;
    message.valid_2 = valid_2;
    message.data_2  = data_2;
    /*if (valid) {
        cout << flag << " [DPI CPP] Sending DATA valid: " << flag << " " << std::hex << (int)message.valid << " data: " << std::hex << message.data << " to " << dest << endl;
    }*/
    MPI_Send(&message, message_len, mpi_noc_type, dest, flag, MPI_COMM_WORLD);
}

unsigned long long mpi_receive_noc(int origin, unsigned short* valid, int flag){
    int message_len = 1;
    MPI_Status status;
    mpi_noc_t message;
    //cout << flag << " [DPI CPP] Blocking Receive data rank: " << origin << endl << std::flush;
    MPI_Recv(&message, message_len, mpi_noc_type, origin, flag, MPI_COMM_WORLD, &status);
    /*if (message.valid) {
        cout << flag << " [DPI CPP] Data Message received: " << (short) message.valid << " " << std::hex << message.data << endl << std::flush;
    }*/
    *valid = message.valid_0;
    return message.data_0;
}

int getRank(){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int getSize(){
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &size);
    return size;
}

void finalize(){
    cout << "[DPI CPP] Finalizing" << endl;
    MPI_Finalize();
}

