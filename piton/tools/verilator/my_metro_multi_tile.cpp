/*
Copyright (c) 2019 Princeton University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Princeton University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY PRINCETON UNIVERSITY "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL PRINCETON UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "Vmetro_tile.h"
#include "verilated.h"
#include <iostream>
#define VERILATOR_VCD 1
//#define MPI_OPT_3 1
#define MPI_OPT_MULTI 1
#ifdef VERILATOR_VCD
#include "verilated_vcd_c.h"
#endif
#include <iomanip>
#include <map>

const int YUMMY_NOC_1   = 0;
const int DATA_NOC_1    = 1;
const int YUMMY_NOC_2   = 2;
const int DATA_NOC_2    = 3;
const int YUMMY_NOC_3   = 4;
const int DATA_NOC_3    = 5;
const int TEST_FINISH   = 6;
const int DATA_ALL_NOC  = 7;
const int ALL_YUMMY     = 8;
const int ALL_NOC       = 9;
const int ALL_MULTI_NOC = 10;

// Compilation flags parameters
const int PITON_X_TILES = X_TILES;
const int PITON_Y_TILES = Y_TILES;
// multi_tile parameter
const int PITON_X_MULTI_TILE = X_MULTI_TILE;
const int PITON_Y_MULTI_TILE = Y_MULTI_TILE;

// Get number of multi_tiles per row X
const int MULTI_TILES_PER_ROW = PITON_X_TILES/PITON_X_MULTI_TILE;
const int MULTI_TILES_PER_COL = PITON_Y_TILES/PITON_Y_MULTI_TILE;

// num cores in the multi_tile
const int NUM_CORES_MULTI_TILE = X_MULTI_TILE * Y_MULTI_TILE;

uint64_t main_time = 0; // Current simulation time
uint64_t clk = 0;
Vmetro_tile* top;
int rank, dest, size;
int rankN, rankS, rankW, rankE;
int tile_x, tile_y;//, PITON_X_TILES, PITON_Y_TILES;
Vmetro_tile* multi_tile_cores[NUM_CORES_MULTI_TILE];
unsigned multi_tile_ids[NUM_CORES_MULTI_TILE];
unsigned multi_tile_idX[NUM_CORES_MULTI_TILE];
unsigned multi_tile_idY[NUM_CORES_MULTI_TILE];
int idNearN[NUM_CORES_MULTI_TILE];
int idNearS[NUM_CORES_MULTI_TILE];
int idNearW[NUM_CORES_MULTI_TILE];
int idNearE[NUM_CORES_MULTI_TILE];
int nearN[NUM_CORES_MULTI_TILE];
int nearS[NUM_CORES_MULTI_TILE];
int nearW[NUM_CORES_MULTI_TILE];
int nearE[NUM_CORES_MULTI_TILE];
// aux map to get index of array from ID
std::map<int,int> idToIndex;

#ifdef VERILATOR_VCD
VerilatedVcdC* tfp[NUM_CORES_MULTI_TILE];
//VerilatedVcdC* tfp;
#endif

// MPI functions
void initialize();
int getRank();
int getSize();
void finalize();

unsigned short mpi_receive_finish();

void mpi_send_finish(unsigned short message, int rank);

typedef struct {
    unsigned long long data_0;
    unsigned long long data_1;
    unsigned long long data_2;
    unsigned short valid_0;
    unsigned short valid_1;
    unsigned short valid_2;
    unsigned short yummy_0;
    unsigned short yummy_1;
    unsigned short yummy_2;
    unsigned short dest_id;    
} mpi_multi_all_t;

void mpi_send_multi_all(mpi_multi_all_t message, int dest, int rank, int flag);
mpi_multi_all_t mpi_receive_multi_all(int origin, int flag);

// This is a 64-bit integer to reduce wrap over issues and
// // allow modulus. You can also use a double, if you wish.
double sc_time_stamp () { // Called by $time in Verilog
    return main_time; // converts to double, to match
    // what SystemC does
}

int get_rank_fromXY(int x, int y) {
    // before there was a +1
    return ((x)+((PITON_X_TILES)*y));
}

// MPI ID funcitons
int getDimX (int id) {
    if (id==0) // Should never happen
        return 0;
    else
        return (id)%PITON_X_TILES;
}

int getDimY (int id) {
    if (id==0) // Should never happen
        return 0;
    else
        return (id)/PITON_X_TILES;
}

int getDimXMultiTile (int id) {
    return (id) % MULTI_TILES_PER_ROW;
}

int getDimYMultiTile (int id) {
    return (id) / MULTI_TILES_PER_ROW;
}

int getIdN (int tile_x, int tile_y) {
    if (tile_y == 0)
        return -1;
    else
        return get_rank_fromXY(tile_x, tile_y-1);
}

int getIdS (int tile_x, int tile_y) {
    if (tile_y+1 == PITON_Y_TILES)
        return -1;
    else
        return get_rank_fromXY(tile_x, tile_y+1);
}

int getIdE (int tile_x, int tile_y) {
    if (tile_x+1 == PITON_X_TILES)
        return -1;
    else
        return get_rank_fromXY(tile_x+1, tile_y);
}

int getIdW (int tile_x, int tile_y) {
    if (rank==1 and tile_x==0 and tile_y==0) { // go to chipset
        return 0;
    }
    else if (tile_x == 0) {
        return -1;
    }
    else {
        return get_rank_fromXY(tile_x-1, tile_y);
    }
}

int getRankN () {
    if (rank <= MULTI_TILES_PER_ROW)
        return -1;
    else
        return rank-MULTI_TILES_PER_ROW;
}

int getRankS () {
    if (((rank-1)/MULTI_TILES_PER_ROW)+1 == MULTI_TILES_PER_COL )
        return -1;
    else
        return rank+MULTI_TILES_PER_ROW;
}

int getRankE () {
    int x_rank = (rank-1) % MULTI_TILES_PER_ROW;
    std::cout << "x_rank: " << x_rank << std::endl;
    std::cout << "MULTI_TILES_PER_ROW: " << MULTI_TILES_PER_ROW << std::endl;
    if (x_rank+1 == MULTI_TILES_PER_ROW)
        return -1;
    else
        return rank+1;
}

int getRankW () {
    int x_rank = (rank-1) % MULTI_TILES_PER_ROW;
    if (rank==1) { // go to chipset
        return 0;
    }
    else if (x_rank == 0) {
        return -1;
    }
    else {
        return rank-1;
    }
}

int getCoreIndex(int id) {
    return idToIndex[id];
}

void tick() {
    top->core_ref_clk = !top->core_ref_clk;
    main_time += 250;
    top->eval();
//#ifdef VERILATOR_VCD
//    tfp->dump(main_time);
//#endif
    top->core_ref_clk = !top->core_ref_clk;
    main_time += 250;
    top->eval();
//#ifdef VERILATOR_VCD
//    tfp->dump(main_time);
//#endif
}

void tick(Vmetro_tile* aux_top, int i) {
    aux_top->core_ref_clk = !aux_top->core_ref_clk;
    main_time += 250;
    aux_top->eval();
    #ifdef VERILATOR_VCD
        tfp[i]->dump(main_time);
    #endif
    aux_top->core_ref_clk = !aux_top->core_ref_clk;
    main_time += 250;
    aux_top->eval();
    #ifdef VERILATOR_VCD
        tfp[i]->dump(main_time);
    #endif
}

void tick_multi_tile(int i) {
    multi_tile_cores[i]->core_ref_clk = !multi_tile_cores[i]->core_ref_clk;
    main_time += 250;
    multi_tile_cores[i]->eval();

    multi_tile_cores[i]->core_ref_clk = !multi_tile_cores[i]->core_ref_clk;
    main_time += 250;
    multi_tile_cores[i]->eval();
}

void mpi_send_multi_N(Vmetro_tile* core, unsigned int dest_id, bool enable_mpi) {

    mpi_multi_all_t message;
    message.data_0  = core->out_N_noc1_data;
    message.valid_0 = core->out_N_noc1_valid;
    message.data_1  = core->out_N_noc2_data;
    message.valid_1 = core->out_N_noc2_valid;
    message.data_2  = core->out_N_noc3_data;
    message.valid_2 = core->out_N_noc3_valid,
    message.yummy_0 = core->out_N_noc1_yummy;
    message.yummy_1 = core->out_N_noc2_yummy;
    message.yummy_2 = core->out_N_noc3_yummy;
    message.dest_id = dest_id;

    if (enable_mpi) {
        // send data over MPI
        mpi_send_multi_all(message, rankN, rank, ALL_MULTI_NOC); 
    } else {
        // we are sending directly inside the MPI
        unsigned index = getCoreIndex(dest_id);
        multi_tile_cores[index]->in_S_noc1_data  = message.data_0; 
        multi_tile_cores[index]->in_S_noc1_valid = message.valid_0;
        multi_tile_cores[index]->in_S_noc2_data  = message.data_1; 
        multi_tile_cores[index]->in_S_noc2_valid = message.valid_1;
        multi_tile_cores[index]->in_S_noc3_data  = message.data_2; 
        multi_tile_cores[index]->in_S_noc3_valid = message.valid_2;
        multi_tile_cores[index]->in_S_noc1_yummy = message.yummy_0;
        multi_tile_cores[index]->in_S_noc2_yummy = message.yummy_1;
        multi_tile_cores[index]->in_S_noc3_yummy = message.yummy_2;
    }
       
}

void mpi_receive_multi_N() {


    //for (int i=0; i < PITON_X_MULTI_TILE; i++) {
        mpi_multi_all_t all_response = mpi_receive_multi_all(rankN, ALL_MULTI_NOC);
        unsigned id = all_response.dest_id;
        unsigned index = getCoreIndex(id);
        multi_tile_cores[index]->in_N_noc1_data  = all_response.data_0; 
        multi_tile_cores[index]->in_N_noc1_valid = all_response.valid_0;
        multi_tile_cores[index]->in_N_noc2_data  = all_response.data_1; 
        multi_tile_cores[index]->in_N_noc2_valid = all_response.valid_1;
        multi_tile_cores[index]->in_N_noc3_data  = all_response.data_2; 
        multi_tile_cores[index]->in_N_noc3_valid = all_response.valid_2;
        multi_tile_cores[index]->in_N_noc1_yummy = all_response.yummy_0;
        multi_tile_cores[index]->in_N_noc2_yummy = all_response.yummy_1;
        multi_tile_cores[index]->in_N_noc3_yummy = all_response.yummy_2;
    //}
       
}

// RECEIVE
/*
// receive data
    mpi_multi_all_t all_response = mpi_receive_multi_all(rankN, ALL_MULTI_NOC);

    core->in_N_noc1_data  = all_response.data_0; 
    core->in_N_noc1_valid = all_response.valid_0;
    core->in_N_noc2_data  = all_response.data_1; 
    core->in_N_noc2_valid = all_response.valid_1;
    core->in_N_noc3_data  = all_response.data_2; 
    core->in_N_noc3_valid = all_response.valid_2;
    core->in_N_noc1_yummy = all_response.yummy_0;
    core->in_N_noc2_yummy = all_response.yummy_1;
    core->in_N_noc3_yummy = all_response.yummy_2;
*/

void mpi_send_multi_S(Vmetro_tile* core, unsigned int dest_id, bool enable_mpi) {

    mpi_multi_all_t message;
    message.data_0  = core->out_S_noc1_data;
    message.valid_0 = core->out_S_noc1_valid;
    message.data_1  = core->out_S_noc2_data;
    message.valid_1 = core->out_S_noc2_valid;
    message.data_2  = core->out_S_noc3_data;
    message.valid_2 = core->out_S_noc3_valid,
    message.yummy_0 = core->out_S_noc1_yummy;
    message.yummy_1 = core->out_S_noc2_yummy;
    message.yummy_2 = core->out_S_noc3_yummy;
    message.dest_id = dest_id;

    if (enable_mpi) {
        // send data over MPI
        mpi_send_multi_all(message, rankS, rank, ALL_MULTI_NOC);     
    } else {
        unsigned index = getCoreIndex(dest_id);
        multi_tile_cores[index]->in_N_noc1_data  = message.data_0; 
        multi_tile_cores[index]->in_N_noc1_valid = message.valid_0;
        multi_tile_cores[index]->in_N_noc2_data  = message.data_1; 
        multi_tile_cores[index]->in_N_noc2_valid = message.valid_1;
        multi_tile_cores[index]->in_N_noc3_data  = message.data_2; 
        multi_tile_cores[index]->in_N_noc3_valid = message.valid_2;
        multi_tile_cores[index]->in_N_noc1_yummy = message.yummy_0;
        multi_tile_cores[index]->in_N_noc2_yummy = message.yummy_1;
        multi_tile_cores[index]->in_N_noc3_yummy = message.yummy_2;
    }
}

void mpi_receive_multi_S() {

    //for (int i=0; i < PITON_X_MULTI_TILE; i++) {
        mpi_multi_all_t all_response = mpi_receive_multi_all(rankS, ALL_MULTI_NOC);
        unsigned id = all_response.dest_id;
        unsigned index = getCoreIndex(id);
        multi_tile_cores[index]->in_S_noc1_data  = all_response.data_0; 
        multi_tile_cores[index]->in_S_noc1_valid = all_response.valid_0;
        multi_tile_cores[index]->in_S_noc2_data  = all_response.data_1; 
        multi_tile_cores[index]->in_S_noc2_valid = all_response.valid_1;
        multi_tile_cores[index]->in_S_noc3_data  = all_response.data_2; 
        multi_tile_cores[index]->in_S_noc3_valid = all_response.valid_2;
        multi_tile_cores[index]->in_S_noc1_yummy = all_response.yummy_0;
        multi_tile_cores[index]->in_S_noc2_yummy = all_response.yummy_1;
        multi_tile_cores[index]->in_S_noc3_yummy = all_response.yummy_2;
    //}
}

void mpi_send_multi_E(Vmetro_tile* core, unsigned int dest_id, bool enable_mpi) {

    mpi_multi_all_t message;
    message.data_0  = core->out_E_noc1_data;
    message.valid_0 = core->out_E_noc1_valid;
    message.data_1  = core->out_E_noc2_data;
    message.valid_1 = core->out_E_noc2_valid;
    message.data_2  = core->out_E_noc3_data;
    message.valid_2 = core->out_E_noc3_valid,
    message.yummy_0 = core->out_E_noc1_yummy;
    message.yummy_1 = core->out_E_noc2_yummy;
    message.yummy_2 = core->out_E_noc3_yummy;
    message.dest_id = dest_id;

    if (enable_mpi) {
        // send data over MPI
        mpi_send_multi_all(message, rankE, rank, ALL_MULTI_NOC);      
    } else {
        unsigned index = getCoreIndex(dest_id);
        //std::cout << "Inside send index: " << index << std::endl;
        multi_tile_cores[index]->in_W_noc1_data  = message.data_0; 
        multi_tile_cores[index]->in_W_noc1_valid = message.valid_0;
        multi_tile_cores[index]->in_W_noc2_data  = message.data_1; 
        multi_tile_cores[index]->in_W_noc2_valid = message.valid_1;
        multi_tile_cores[index]->in_W_noc3_data  = message.data_2; 
        multi_tile_cores[index]->in_W_noc3_valid = message.valid_2;
        multi_tile_cores[index]->in_W_noc1_yummy = message.yummy_0;
        multi_tile_cores[index]->in_W_noc2_yummy = message.yummy_1;
        multi_tile_cores[index]->in_W_noc3_yummy = message.yummy_2;
        if (message.valid_0 or message.valid_1 or message.valid_2) {
            std::cout << " send multi E-> " << " dest_id: " << dest_id << " " << " index: " << index << " " << std::endl << std::flush;
            std::cout << core->flat_tileid << " ";
            std::cout << std::setprecision(10) << sc_time_stamp();
            std::cout << " " << message.valid_0 << " " << message.data_0;
            std::cout << " " << message.valid_1 << " " << message.data_1;
            std::cout << " " << message.valid_2 << " " << message.data_2;
            std::cout << " " << message.yummy_0 << " " << message.yummy_1 << " " << message.yummy_2;
            std::cout << std::endl << std::flush;
        }
    }
      
}

void mpi_receive_multi_E() {

    //for (int i=0; i < PITON_Y_MULTI_TILE; i++) {
        mpi_multi_all_t all_response = mpi_receive_multi_all(rankE, ALL_MULTI_NOC);
        unsigned id = all_response.dest_id;
        unsigned index = getCoreIndex(id);
        multi_tile_cores[index]->in_E_noc1_data  = all_response.data_0; 
        multi_tile_cores[index]->in_E_noc1_valid = all_response.valid_0;
        multi_tile_cores[index]->in_E_noc2_data  = all_response.data_1; 
        multi_tile_cores[index]->in_E_noc2_valid = all_response.valid_1;
        multi_tile_cores[index]->in_E_noc3_data  = all_response.data_2; 
        multi_tile_cores[index]->in_E_noc3_valid = all_response.valid_2;
        multi_tile_cores[index]->in_E_noc1_yummy = all_response.yummy_0;
        multi_tile_cores[index]->in_E_noc2_yummy = all_response.yummy_1;
        multi_tile_cores[index]->in_E_noc3_yummy = all_response.yummy_2;
    //}
}

void mpi_send_multi_W(Vmetro_tile* core, unsigned int dest_id, bool enable_mpi) {

    mpi_multi_all_t message;
    message.data_0  = core->out_W_noc1_data;
    message.valid_0 = core->out_W_noc1_valid;
    message.data_1  = core->out_W_noc2_data;
    message.valid_1 = core->out_W_noc2_valid;
    message.data_2  = core->out_W_noc3_data;
    message.valid_2 = core->out_W_noc3_valid,
    message.yummy_0 = core->out_W_noc1_yummy;
    message.yummy_1 = core->out_W_noc2_yummy;
    message.yummy_2 = core->out_W_noc3_yummy;
    message.dest_id = dest_id;
    //std::cout << message.data_0 << " "  << message.valid_0  << std::endl;
    //std::cout << message.data_1 << " "  << message.valid_1  << std::endl;
    //std::cout << message.data_2 << " "  << message.valid_2  << std::endl;
    //std::cout << message.yummy_0 << " " << message.yummy_1 << " " << message.yummy_2 << std::endl << std::flush;
//
    //std::cout << "enable_mpi: " << enable_mpi << " dest_id: " << dest_id << " " << std::endl << std::flush;

    if (enable_mpi) {
        //std::cout << "enable_mpi: " << enable_mpi << " dest_id: " << dest_id << " " << std::endl << std::flush;
        // send data over MPI
        mpi_send_multi_all(message, rankW, rank, ALL_MULTI_NOC);   
    } else {
        unsigned index = getCoreIndex(dest_id);
                
        multi_tile_cores[index]->in_E_noc1_data  = message.data_0; 
        multi_tile_cores[index]->in_E_noc1_valid = message.valid_0;
        multi_tile_cores[index]->in_E_noc2_data  = message.data_1; 
        multi_tile_cores[index]->in_E_noc2_valid = message.valid_1;
        multi_tile_cores[index]->in_E_noc3_data  = message.data_2; 
        multi_tile_cores[index]->in_E_noc3_valid = message.valid_2;
        multi_tile_cores[index]->in_E_noc1_yummy = message.yummy_0;
        multi_tile_cores[index]->in_E_noc2_yummy = message.yummy_1;
        multi_tile_cores[index]->in_E_noc3_yummy = message.yummy_2;

        if (message.valid_0 or message.valid_1 or message.valid_2) {
            std::cout << " send multi W-> " << " dest_id: " << dest_id << " " << " index: " << index << " " << std::endl << std::flush;
            std::cout << core->flat_tileid << " ";
            std::cout << std::setprecision(10) << sc_time_stamp();
            std::cout << " " << message.valid_0 << " " << message.data_0;
            std::cout << " " << message.valid_1 << " " << message.data_1;
            std::cout << " " << message.valid_2 << " " << message.data_2;
            std::cout << " " << message.yummy_0 << " " << message.yummy_1 << " " << message.yummy_2;
            std::cout << std::endl << std::flush;
        }

        /*if (core->out_W_noc1_valid) {
            std::cout << "Out W valid time: " << std::setprecision(10) << sc_time_stamp() << std::endl << std::flush;
            std::cout << "dest id: " << dest_id << " index " << index << std::endl << std::flush;
            std::cout << "out_W_noc1_valid 1, message.valid_0 " << message.valid_0 << " data " << message.data_0 << std::endl << std::flush;
            std::cout << "in_W_noc1_valid " << multi_tile_cores[index]->in_E_noc1_valid << " data " << multi_tile_cores[index]->in_E_noc1_data  << std::endl << std::flush;
        }*/
    }
       
}

void mpi_receive_multi_W() {

    //for (int i=0; i < PITON_Y_MULTI_TILE; i++) {
        mpi_multi_all_t all_response = mpi_receive_multi_all(rankW, ALL_MULTI_NOC);
        unsigned id = all_response.dest_id;
        unsigned index = getCoreIndex(id);
        //std::cout << index << " " << id << std::endl;
        multi_tile_cores[index]->in_W_noc1_data  = all_response.data_0; 
        multi_tile_cores[index]->in_W_noc1_valid = all_response.valid_0;
        multi_tile_cores[index]->in_W_noc2_data  = all_response.data_1; 
        multi_tile_cores[index]->in_W_noc2_valid = all_response.valid_1;
        multi_tile_cores[index]->in_W_noc3_data  = all_response.data_2; 
        multi_tile_cores[index]->in_W_noc3_valid = all_response.valid_2;
        multi_tile_cores[index]->in_W_noc1_yummy = all_response.yummy_0;
        multi_tile_cores[index]->in_W_noc2_yummy = all_response.yummy_1;
        multi_tile_cores[index]->in_W_noc3_yummy = all_response.yummy_2;
    //}
}

void mpi_tick() {
    // clock the cores
    //top->core_ref_clk = !top->core_ref_clk;
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        multi_tile_cores[i]->core_ref_clk = !multi_tile_cores[i]->core_ref_clk;
    }
    // Advance time
    main_time += 250;
    // first eval of the cores
    // top->eval();
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        multi_tile_cores[i]->eval();
    }
    
    // SEND FIRST ALL MESSAGES
    // Send all North
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        // Send over MPI
        if (nearN[i] == 1) {
            mpi_send_multi_N(multi_tile_cores[i], idNearN[i], true);
        }
        // Send over hand inside multi-tile
        else if (nearN[i] == 0) {
            mpi_send_multi_N(multi_tile_cores[i], idNearN[i], false);
        } //else { // -1 DO NOT SEND
            // Do nothing
        //}
    }

    // Send all South
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        // Send over MPI
        if (nearS[i] == 1) {
            mpi_send_multi_S(multi_tile_cores[i], idNearS[i], true);
        }
        // Send over hand inside multi-tile
        else if (nearS[i] == 0) {
            mpi_send_multi_S(multi_tile_cores[i], idNearS[i], false);
        } //else { // -1 DO NOT SEND
            // Do nothing
        //}
    }

    // Send all East
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        // Send over MPI
        if (nearE[i] == 1) {
            mpi_send_multi_E(multi_tile_cores[i], idNearE[i], true);
        }
        // Send over hand inside multi-tile
        else if (nearE[i] == 0) {
            mpi_send_multi_E(multi_tile_cores[i], idNearE[i], false);
        } //else { // -1 DO NOT SEND
            // Do nothing
        //}
    }

    // Send all West
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        // Send over MPI
        if (nearW[i] == 1) {
            mpi_send_multi_W(multi_tile_cores[i], idNearW[i], true);
        }
        // Send over hand inside multi-tile
        else if (nearW[i] == 0) {
            mpi_send_multi_W(multi_tile_cores[i], idNearW[i], false);
        } //else { // -1 DO NOT SEND
            // Do nothing
        //}
    }

    // Receive ALL NOW
    // Receive North 
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        // Receive over MPI
        if (nearN[i] == 1) {
            mpi_receive_multi_N();
        }
        // Send over hand inside multi-tile
        /*else if (nearN[multi_tile_ids[i]] == 0) {
            receive_multi_N(multi_tile_cores[i], idNearN[i]);
        } //else { // -1 DO NOT SEND*/
            // Do nothing
        //}
    }

    // Receive all South
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        // Send over MPI
        if (nearS[i] == 1) {
            mpi_receive_multi_S();
        }
        // Send over hand inside multi-tile
        /*else if (nearS[multi_tile_ids[i]] == 0) {
            mpi_receive_multi_N(multi_tile_cores[i], idNearS[i]);
        } //else { // -1 DO NOT SEND*/
            // Do nothing
        //}
    }

    // Receive all East
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        // Send over MPI
        if (nearE[i] == 1) {
            mpi_receive_multi_E();
        }
        // Send over hand inside multi-tile
        /*else if (nearE[multi_tile_ids[i]] == 0) {
            mpi_receive_multi_E(multi_tile_cores[i], idNearE[i]);
        } //else { // -1 DO NOT SEND*/
            // Do nothing
        //}
    }

    // Receive all West
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        // Send over MPI
        if (nearW[i] == 1) {
            mpi_receive_multi_W();
        }
        // Send over hand inside multi-tile
        /*else if (nearW[multi_tile_ids[i]] == 0) {
            mpi_receive_multi_W(multi_tile_cores[i], idNearW[i]);
        } //else { // -1 DO NOT SEND*/
            // Do nothing
        //}
    }
    
    // top->eval();
    // second eval
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        multi_tile_cores[i]->eval();
    }
#ifdef VERILATOR_VCD
    //tfp->dump(main_time);
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        tfp[i]->dump(main_time);
    }
#endif
    // clock the cores
    //top->core_ref_clk = !top->core_ref_clk;
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        multi_tile_cores[i]->core_ref_clk = !multi_tile_cores[i]->core_ref_clk;
    }
    main_time += 250;
    // top->eval();
    // second eval
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        multi_tile_cores[i]->eval();
    }
#ifdef VERILATOR_VCD
    //tfp->dump(main_time);
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        tfp[i]->dump(main_time);
    }
#endif
}

void reset_and_init_multi() {

    for (int n=0; n < NUM_CORES_MULTI_TILE;n++) {
        // Hack to not modify global time
        main_time = 0;
        //multi_tile_cores[] = multi_tile_cores[n];
    
    //    fail_flag = 1'b0;
    //    stub_done = 4'b0;
    //    stub_pass = 4'b0;

    //    // Clocks initial value
        multi_tile_cores[n]->core_ref_clk = 0;

    //    // Resets are held low at start of boot
        multi_tile_cores[n]->sys_rst_n = 0;
        multi_tile_cores[n]->pll_rst_n = 0;

        multi_tile_cores[n]->ok_iob = 0;

    //    // Mostly DC signals set at start of boot
    //    clk_en = 1'b0;
        multi_tile_cores[n]->pll_bypass = 1; // trin: pll_bypass is a switch in the pll; not reliable
        multi_tile_cores[n]->clk_mux_sel = 0; // selecting ref clock
    //    // rangeA = x10 ? 5'b1 : x5 ? 5'b11110 : x2 ? 5'b10100 : x1 ? 5'b10010 : x20 ? 5'b0 : 5'b1;
        multi_tile_cores[n]->pll_rangea = 1; // 10x ref clock
    //    // pll_rangea = 5'b11110; // 5x ref clock
    //    // pll_rangea = 5'b00000; // 20x ref clock
        
    //    // JTAG simulation currently not supported here
    //    jtag_modesel = 1'b1;
    //    jtag_datain = 1'b0;

        multi_tile_cores[n]->async_mux = 0;

        multi_tile_cores[n]->in_N_noc1_data  = 0;
        multi_tile_cores[n]->in_E_noc1_data  = 0;
        multi_tile_cores[n]->in_W_noc1_data  = 0;
        multi_tile_cores[n]->in_S_noc1_data  = 0;
        multi_tile_cores[n]->in_N_noc1_valid = 0;
        multi_tile_cores[n]->in_E_noc1_valid = 0;
        multi_tile_cores[n]->in_W_noc1_valid = 0;
        multi_tile_cores[n]->in_S_noc1_valid = 0;
        multi_tile_cores[n]->in_N_noc1_yummy = 0;
        multi_tile_cores[n]->in_E_noc1_yummy = 0;
        multi_tile_cores[n]->in_W_noc1_yummy = 0;
        multi_tile_cores[n]->in_S_noc1_yummy = 0;

        multi_tile_cores[n]->in_N_noc2_data  = 0;
        multi_tile_cores[n]->in_E_noc2_data  = 0;
        multi_tile_cores[n]->in_W_noc2_data  = 0;
        multi_tile_cores[n]->in_S_noc2_data  = 0;
        multi_tile_cores[n]->in_N_noc2_valid = 0;
        multi_tile_cores[n]->in_E_noc2_valid = 0;
        multi_tile_cores[n]->in_W_noc2_valid = 0;
        multi_tile_cores[n]->in_S_noc2_valid = 0;
        multi_tile_cores[n]->in_N_noc2_yummy = 0;
        multi_tile_cores[n]->in_E_noc2_yummy = 0;
        multi_tile_cores[n]->in_W_noc2_yummy = 0;
        multi_tile_cores[n]->in_S_noc2_yummy = 0;

        multi_tile_cores[n]->in_N_noc3_data  = 0;
        multi_tile_cores[n]->in_E_noc3_data  = 0;
        multi_tile_cores[n]->in_W_noc3_data  = 0;
        multi_tile_cores[n]->in_S_noc3_data  = 0;
        multi_tile_cores[n]->in_N_noc3_valid = 0;
        multi_tile_cores[n]->in_E_noc3_valid = 0;
        multi_tile_cores[n]->in_W_noc3_valid = 0;
        multi_tile_cores[n]->in_S_noc3_valid = 0;
        multi_tile_cores[n]->in_N_noc3_yummy = 0;
        multi_tile_cores[n]->in_E_noc3_yummy = 0;
        multi_tile_cores[n]->in_W_noc3_yummy = 0;
        multi_tile_cores[n]->in_S_noc3_yummy = 0;

        //init_jbus_model_call((char *) "mem.image", 0);

        //std::cout << "Before first ticks" << std::endl << std::flush;
        tick(multi_tile_cores[n],n);
        //std::cout << "After very first tick" << std::endl << std::flush;
    //    // Reset PLL for 100 cycles
    //    repeat(100)@(posedge core_ref_clk);
    //    pll_rst_n = 1'b1;
        for (int i = 0; i < 100; i++) {
            tick(multi_tile_cores[n],n);
        }
        multi_tile_cores[n]->pll_rst_n = 1;

        //std::cout << "Before second ticks" << std::endl << std::flush;
        //    // Wait for PLL lock
        //    wait( pll_lock == 1'b1 );
        //    while (!multi_tile_cores[n]->pll_lock) {
        //        tick();
        //    }

        //std::cout << "Before third ticks" << std::endl << std::flush;
    //    // After 10 cycles turn on chip-level clock enable
    //    repeat(10)@(posedge `CHIP_INT_CLK);
    //    clk_en = 1'b1;
        for (int i = 0; i < 10; i++) {
            tick(multi_tile_cores[n],n);
        }
        multi_tile_cores[n]->clk_en = 1;

    //    // After 100 cycles release reset
    //    repeat(100)@(posedge `CHIP_INT_CLK);
    //    sys_rst_n = 1'b1;
    //    jtag_rst_l = 1'b1;
        for (int i = 0; i < 100; i++) {
            tick(multi_tile_cores[n],n);
        }
        multi_tile_cores[n]->sys_rst_n = 1;

    //    // Wait for SRAM init, trin: 5000 cycles is about the lowest
    //    repeat(5000)@(posedge `CHIP_INT_CLK);
        for (int i = 0; i < 5000; i++) {
            tick(multi_tile_cores[n],n);
        }

    //    multi_tile_cores[n]->diag_done = 1;

        //multi_tile_cores[n]->ciop_fake_iob.ok_iob = 1;
        multi_tile_cores[n]->ok_iob = 1;
        //std::cout << "Reset complete" << std::endl << std::flush;

    }
}

int main(int argc, char **argv, char **env) {
    //std::cout << "Started" << std::endl << std::flush;
    Verilated::commandArgs(argc, argv);

    //top = new Vmetro_tile;
    std::cout << "Vmetro_multi_tile created" << std::endl << std::flush;

    std::cout << "NUM_CORES_MULTI_TILE: " << NUM_CORES_MULTI_TILE << std::endl << std::flush;

    // MPI work 
    initialize();
    rank = getRank();
    size = getSize();

    std::cout << "rank: " << rank << " size: " << size << std::endl << std::flush; 
    
    // Get ID of first element in multi-tile
    // rank divided number of multi_tiles per row then we get position y --> multiply by number of PITON_X_TILES
    unsigned multi_tile_tid = ((rank-1)*PITON_X_MULTI_TILE) + ((rank-1)/MULTI_TILES_PER_ROW)*(PITON_X_TILES); 

    std::cout << "[" << rank << "] " << "multi_tile_tid: " << multi_tile_tid << std::endl << std::flush;
    
    // Create multi_tile_ids array
    // TODO Revisit
    for (int i=0; i < PITON_Y_MULTI_TILE; i++) {
        for (int j=0; j < PITON_X_MULTI_TILE; j++) {
            int ptr=i+j;
            multi_tile_ids[ptr] = multi_tile_tid+ptr;
            idToIndex[multi_tile_tid+j]=ptr;
            multi_tile_idX[ptr] = getDimX(multi_tile_tid+ptr);
            multi_tile_idY[ptr] = getDimY(multi_tile_tid+ptr);
        }
        multi_tile_tid+=PITON_Y_TILES; // summ one entire row
    }
    ////////////////////////// PRINT /////////////////////////////////////////////////
    multi_tile_tid = ((rank-1)*PITON_X_MULTI_TILE) + ((rank-1)/MULTI_TILES_PER_ROW)*(PITON_X_TILES); 
    for (int i=0; i < PITON_Y_MULTI_TILE; i++) {
        for (int j=0; j < PITON_X_MULTI_TILE; j++) {
            int ptr=i+j;
            std::cout << "[" << multi_tile_tid+j << "] " << "ptr: " << ptr << std::endl << std::flush;
            std::cout << "[" << multi_tile_tid+j << "] " << "multi_tile_ids[ptr]: " << multi_tile_ids[ptr] << std::endl << std::flush;
            std::cout << "[" << multi_tile_tid+j << "] " << "idToIndex[multi_tile_tid+i]: " << idToIndex[multi_tile_tid+j] << std::endl << std::flush;
            std::cout << "[" << multi_tile_tid+j << "] " << "multi_tile_idX[ptr]: " << multi_tile_idX[ptr] << std::endl << std::flush;
            std::cout << "[" << multi_tile_tid+j << "] " << "multi_tile_idY[ptr]: " << multi_tile_idY[ptr] << std::endl << std::flush;
        }
        multi_tile_tid+=PITON_Y_TILES; // summ one entire row
    }
    ////////////////////////// PRINT /////////////////////////////////////////////////
    
   
    // Create idNear array
    for (int i=0; i < NUM_CORES_MULTI_TILE; i++) {
        idNearN[i] = getIdN(multi_tile_idX[i],multi_tile_idY[i]);
        idNearS[i] = getIdS(multi_tile_idX[i],multi_tile_idY[i]);
        idNearW[i] = getIdW(multi_tile_idX[i],multi_tile_idY[i]);
        idNearE[i] = getIdE(multi_tile_idX[i],multi_tile_idY[i]);
        // Set Near Vectors
        if (idNearN[i] == -1) {
            nearN[i] = -1;
        } else if (idToIndex.count(idNearN[i])>0) {
            nearN[i] = 0;
        } else {
            nearN[i] = 1;
        }
        if (idNearS[i] == -1) {
            nearS[i] = -1;
        } else if (idToIndex.count(idNearS[i])>0) {
            nearS[i] = 0;
        } else {
            nearS[i] = 1;
        }
        if (idNearW[i] == -1) {
            nearW[i] = -1;
        } else if (idToIndex.count(idNearW[i])>0) {
            if (rank==1 and idNearW[i]==0 and i==0) {
                nearW[i] = 1;
            } else {
                nearW[i] = 0;
            }
        } else {
            nearW[i] = 1;
        }
        if (idNearE[i] == -1) {
            nearE[i] = -1;
        } else if (idToIndex.count(idNearE[i])>0) {
            nearE[i] = 0;
        } else {
            nearE[i] = 1;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Create idNear array
    for (int i=0; i < NUM_CORES_MULTI_TILE; i++) {
        std::cout << "[" << i << "] " << "idNearN[i]: " << idNearN[i] << std::endl << std::flush;
        std::cout << "[" << i << "] " << "idNearS[i]: " << idNearS[i] << std::endl << std::flush;
        std::cout << "[" << i << "] " << "idNearW[i]: " << idNearW[i] << std::endl << std::flush;
        std::cout << "[" << i << "] " << "idNearE[i]: " << idNearE[i] << std::endl << std::flush;
        std::cout << "[" << i << "] " << "nearN[i]: " << nearN[i] << std::endl << std::flush;
        std::cout << "[" << i << "] " << "nearS[i]: " << nearS[i] << std::endl << std::flush;
        std::cout << "[" << i << "] " << "nearW[i]: " << nearW[i] << std::endl << std::flush;
        std::cout << "[" << i << "] " << "nearE[i]: " << nearE[i] << std::endl << std::flush;
    }
    /////////////////////////////////////////////////////////////////////////////////
    
    //tile_x = getDimX();
    //tile_y = getDimY();
    assert("RANK > 0" && rank>0);
    rankN  = getRankN();
    rankS  = getRankS();
    rankW  = getRankW();
    rankE  = getRankE();

    //#ifdef VERILATOR_VCD
    //std::cout << "TILE size: " << size << ", rank: " << rank <<  std::endl;
    //std::cout << "tile_y: " << tile_y << std::endl;
    //std::cout << "tile_x: " << tile_x << std::endl;
    std::cout << "rankN: " << rankN << std::endl;
    std::cout << "rankS: " << rankS << std::endl;
    std::cout << "rankW: " << rankW << std::endl;
    std::cout << "rankE: " << rankE << std::endl;
    //#endif

    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        multi_tile_cores[i] = new Vmetro_tile;
        multi_tile_cores[i]->default_chipid = 0;
        multi_tile_cores[i]->default_coreid_x = multi_tile_idX[i];
        multi_tile_cores[i]->default_coreid_y = multi_tile_idY[i];
        multi_tile_cores[i]->flat_tileid = multi_tile_ids[i];
    }

    #ifdef VERILATOR_VCD
        Verilated::traceEverOn(true);
        for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
            tfp[i] = new VerilatedVcdC;
            multi_tile_cores[i]->trace (tfp[i], 99);
            std::string tracename ("my_metro_tile"+std::to_string(i)+".vcd");
            const char *cstr = tracename.c_str();
            tfp[i]->open(cstr);
        }
        Verilated::debug(1);
    #endif

    //std::cout << "Before reset" << std::endl << std::flush;
    reset_and_init_multi();
    //std::cout << "After Vmetro_multi_tile reset" << std::endl << std::flush;

    bool test_exit = false;
    uint64_t cyclesToCheckEnd=std::stoi(argv[1]);
    uint64_t CyclesToCheckEndAfter=std::stoi(argv[2]);
    while (!Verilated::gotFinish() and !test_exit) { 
        mpi_tick();
        if (cyclesToCheckEnd==0) {
            //std::cout << "Checking Finish TILE" << std::endl;
            test_exit= mpi_receive_finish();
            cyclesToCheckEnd=CyclesToCheckEndAfter;
            //std::cout << "Finishing: " << test_exit << std::endl;
        }
        else {
            cyclesToCheckEnd--;
        }
    }
    std::cout << "ticks: " << std::setprecision(10) << sc_time_stamp() << " , cycles: " << sc_time_stamp()/500 << std::endl;

    #ifdef VERILATOR_VCD
    std::cout << "Trace done" << std::endl;
    //tfp->close();
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        tfp[i]->close();
    }
    #endif

    finalize();
    //top->final();
    for (int i=0; i < NUM_CORES_MULTI_TILE;i++) {
        multi_tile_cores[i]->final();
    }

    //delete top;
    exit(0);
}
