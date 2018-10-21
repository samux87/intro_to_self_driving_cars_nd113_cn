/**
    localizer.cpp

    Purpose: implements a 2-dimensional histogram filter
    for a robot living on a colored cyclical grid by 
    correctly implementing the "initialize_beliefs", 
    "sense", and "move" functions.

    This file is incomplete! Your job is to make these
    functions work. Feel free to look at localizer.py 
    for working implementations which are written in python.
*/



#include "helpers.cpp"
#include <stdlib.h>
#include "debugging_helpers.cpp"

using namespace std;
//#endif 

/**
    TODO - implement this function 
    
    Initializes a grid of beliefs to a uniform distribution. 

    @param grid - a two dimensional grid map (vector of vectors 
           of chars) representing the robot's world. For example:
           
           g g g
           g r g
           g g g
           
           would be a 3x3 world where every cell is green except 
           for the center, which is red.

    @return - a normalized two dimensional grid of floats. For 
           a 2x2 grid, for example, this would be:

           0.25 0.25
           0.25 0.25
*/

// 定义initialize_beliefs函数
vector< vector <float> > initialize_beliefs(vector< vector <char> > grid) {
    vector< vector <float> > newGrid;
    // your code here
    int height = grid.size();
    int width = grid[0].size();
    int area = height * width;
    float belief_per_cell = 1.0 / area;
    vector<float> row;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            row.push_back(belief_per_cell);
        }
        newGrid.push_back(row);
        row.clear();
    }
    return newGrid;
}

/**
    TODO - implement this function 
    
    Implements robot sensing by updating beliefs based on the 
    color of a sensor measurement 

    @param color - the color the robot has sensed at its location

    @param grid - the current map of the world, stored as a grid
           (vector of vectors of chars) where each char represents a 
           color. For example:

           g g g
           g r g
           g g g

    @param beliefs - a two dimensional grid of floats representing
           the robot's beliefs for each cell before sensing. For 
           example, a robot which has almost certainly localized 
           itself in a 2D world might have the following beliefs:

           0.01 0.98
           0.00 0.01

    @param p_hit - the RELATIVE probability that any "sense" is 
           correct. The ratio of p_hit / p_miss indicates how many
           times MORE likely it is to have a correct "sense" than
           an incorrect one.

    @param p_miss - the RELATIVE probability that any "sense" is 
           incorrect. The ratio of p_hit / p_miss indicates how many
           times MORE likely it is to have a correct "sense" than
           an incorrect one.

    @return - a normalized two dimensional grid of floats 
           representing the updated beliefs for the robot. 
*/



// 定义sense函数
vector< vector<float> > sense(char color, vector< vector<char> > grid, vector < vector<float> > beliefs, float p_hit, float p_miss) {
    vector < vector<float> > new_beliefs;
    vector <float> new_row;
    int hit;
    for (int i = 0; i < grid.size(); ++i) {
        for (int j = 0; j < grid[0].size(); ++j) {
            if (grid[i][j] == color) {
                hit = 1;
            }
            else {
                hit = 0;
            }
            new_row.push_back(hit*beliefs[i][j] * p_hit + (1 - hit)*beliefs[i][j] * p_miss);
        }
        new_beliefs.push_back(new_row);
        new_row.clear();
    }
    return  normalize(new_beliefs);
}



// 定义move函数
vector< vector <float> > move(int dy, int dx,
    vector< vector <float> > beliefs, float blurring) {
    int height = beliefs.size();
    int width = beliefs[0].size();
    vector< vector<float> > newGrid(beliefs.size(), vector<float>(beliefs[0].size(), 0.0));
    for (int i = 0; i < beliefs.size(); ++i) {
        for (int j = 0; j < beliefs[0].size(); ++j) {
            int new_i = (i + dy) % height;
            int new_j = (j + dx) % width;
            newGrid[new_j][new_i] = beliefs[i][j];
        }
    }
    return blur(newGrid, blurring);
}