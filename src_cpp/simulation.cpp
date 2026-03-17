#include "simulation.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

namespace py = pybind11;
using namespace std; 

MonteCarloEngine::MonteCarloEngine(int assets, int sim) 
    : assets(assets), sim(sim) {}

// Vectors handle their own memory. No manual delete needed.
MonteCarloEngine::~MonteCarloEngine() {}

SimResult MonteCarloEngine::runSimulation(
    double initial_portfolio_value,
    const vector<double>& weights,
    const vector<double>& mean_returns,
    const vector<double>& cov_matrix) 
    {

    //Cholesky Decomposition , creating the L Matrix (in a flat vector)
    vector<double> L(assets * assets, 0.0);
    for (int i = 0; i < assets; i++) 
    {
        for (int j = 0; j <= i; j++) 
        {
            double sum = 0.0;
            if (j == i) 
            { 
                for (int k = 0; k < j; k++)
                    sum +=(L[j * assets + k])*(L[j * assets + k]);
                L[j * assets + j] = sqrt(cov_matrix[j * assets + j] - sum);
            } 
            else 
            {      
                for (int k = 0; k < j; k++)
                    sum += (L[i * assets + k] * L[j * assets + k]);
                L[i * assets + j] = (cov_matrix[i * assets + j] - sum) / L[j * assets + j];
            }
        }
    }
    //  Random Number Generator
    mt19937 generator(random_device{}());
    normal_distribution<double> distribution(0.0, 1.0);

    vector<double> final_portfolio_values;
    // resize fills the array with zeros and reserve keeps it empty saving us some work since we do not need those zeros
    final_portfolio_values.reserve(sim);

    vector<double> correlated_randoms(assets, 0.0);
    vector<double> uncorrelated_randoms(assets, 0.0);

    // The Monte Carlo Loop
    for(int i = 0; i < sim; i++) 
    {
        
        for(int j = 0; j < assets; j++)
            uncorrelated_randoms[j] = distribution(generator);

        fill(correlated_randoms.begin(), correlated_randoms.end(), 0.0);

        // Matrix Multiplication 
        for(int k = 0; k < assets; k++) 
        {
            for(int c = 0; c <= k; c++) 
            {
                correlated_randoms[k] += L[k * assets + c] * uncorrelated_randoms[c];
            }
        } 
        
        double current_value = 0.0;

        // Geometric Brownian Motion : Price_t = Price_0 * exp( (Mean - 0.5 * Var) + StdDev * Random )
        for (int a = 0; a < assets; a++) 
        {
            double volatility = sqrt(cov_matrix[a * assets + a]);
            double drift = mean_returns[a] - 0.5 * volatility * volatility;
            double diffusion =  correlated_randoms[a];
            
            double return_factor = exp(drift + diffusion);
            current_value += (initial_portfolio_value * weights[a]) * return_factor;
        }
        final_portfolio_values.push_back(current_value);
    }

   
    sort(final_portfolio_values.begin(), final_portfolio_values.end());

    int ind_95 = static_cast<int>(sim * 0.05);
    int ind_99=static_cast<int>(sim*0.01);
    
    SimResult result;
    result.var95 = initial_portfolio_value - final_portfolio_values[ind_95];
    result.var99 = initial_portfolio_value - final_portfolio_values[ind_99];

    //calculating mean loss
    double loss = 0.0;
    for (int i = 0; i < ind_95; i++) 
    {
        loss += (initial_portfolio_value - final_portfolio_values[i]);
    }
    result.mean_loss = (ind_95 > 0) ? (loss / ind_95) : 0.0; 

    return result;
}

PYBIND11_MODULE(riskengine, m) 
{
    // Bind the struct
    py::class_<SimResult>(m, "SimResult")
        .def_readwrite("var95", &SimResult::var95)
        .def_readwrite("var99", &SimResult::var99)
        .def_readwrite("mean_loss", &SimResult::mean_loss);

    // Bind the Engine class
    py::class_<MonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<int, int>(), py::arg("assets"), py::arg("sim"))
        .def("runSimulation", &MonteCarloEngine::runSimulation,
             py::arg("initial_portfolio_value"),
             py::arg("weights"),
             py::arg("mean_returns"),
             py::arg("cov_matrix"));
}