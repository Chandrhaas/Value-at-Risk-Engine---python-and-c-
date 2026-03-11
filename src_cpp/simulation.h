//We are making a header file so that python can interact with c++ smoothly  
//and if we need more than one montecarlo engine we can just import this file instead of rewriting the whole code 
#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <iostream>
using namespace std;

struct Simresult 
{
    double var95;
    double var99;
    double meanloss;
};

class MonteCarloEngine
{   
    // We are using pass-by reference to save time which is taken to copy the data
    MonteCarloEngine(int sim, int assets,
                    const vector<double>&mean,
                    const vector<double>&covs,
                    const vector<double>& weights,
                    double portfolio_value);

    void run();

    Simresult get_result();
    
    private:
        int sim=0,assets=0;
        double portfolio_value=0.0;
        vector<double>mean,covs,weights;     
        Simresult result;
    
    vector<vector<double>>cholesky_decomp(vector<double>&mat,int n);

};

// c++ changes the names of classes. Python will not know the new name 
// extern tells to complie by the rules of c and not c++ so names are not changed and we can use the functions in python
extern "C"
{
    //python cannot create the object so here we do it
    MonteCarloEngine* MonteCarloEngine_new(int num_sims, int num_assets, 
                                           double* means, double* cov_matrix, 
                                           double* weights, double initial_portfolio_value);
    
    void MonteCarloEngine_run(MonteCarloEngine* engine);

   
    Simresult MonteCarloEngine_getResults(MonteCarloEngine* engine);

    
    void MonteCarloEngine_delete(MonteCarloEngine* engine);
}

#endif



