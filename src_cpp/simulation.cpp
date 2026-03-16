#include "simulation.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
using namespace std;

MonteCarloEngine::MonteCarloEngine(int sim , int assets,
                                    const vector<double>&mean,
                                    const vector<double>&covs,
                                    const vector<double> &weights,
                                    double portfolio_value) :sim(sim), assets(assets),mean(mean),covs(covs),
                                    weights(weights),portfolio_value(portfolio_value){}

vector<vector<double>> MonteCarloEngine::cholesky_decomp(vector<double>&mat,int n)
{
    vector<vector<double>>L(n,vector<double>(n,0.0));
    //making mat into a lower tirangle matrix
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<=i;j++)
        {
            double sum=0;
            for(int k=0;k<j;k++)
            sum+=L[i][k]*L[j][k];

            if(i==j)
            L[i][j]=sqrt(mat[i*n+i]-sum);
            else
            L[i][j]=(1.0/L[j][j]*(mat[i*n+j]-sum));
        }
    }
    return L;
}
void MonteCarloEngine::run()
{
    mt19937 generator(random_device{}());
    normal_distribution<double>distribution(0.0,1.0);

    auto L = cholesky_decomp(covs,assets);

    vector<double>final_portfolio_values;
    // resize fills the array with zeros and reserve keeps it empty saving us some work since we do not need those zeros
    final_portfolio_values.reserve(sim);

    vector<double> correlated_randoms(assets, 0.0);
    vector<double> uncorrelated_randoms(assets, 0.0);
    for(int i=0;i<sim;i++)
    {
        //generating random number vector
        for(int j=0;j<assets;j++)
        uncorrelated_randoms[j]=distribution(generator);

        fill(correlated_randoms.begin(), correlated_randoms.end(), 0.0);

        // multiplying the random number vector with cholesky decomp matrix to get an array of correlated random numbers
        for(int k=0;k<assets;k++)
        {
            for(int c=0;c<=k;c++)
           correlated_randoms[k]+=L[k][c]*uncorrelated_randoms[c];
        }
        

        double current_value=0.0;

        //using geometric brownian motion : Price_t = Price_0 * exp( (Mean - 0.5 * Var) + StdDev * Random )
        for (int a=0;a<assets;a++)
        {
            // volatility = std deviation
            double volatility = sqrt(covs[a * assets + a]);

            // drift = mean - volatility drag 
            double drift = mean[a] - 0.5 * volatility * volatility;

            //the random shock
            double diffusion = volatility * correlated_randoms[a];

            double return_factor=exp(drift+diffusion);

            current_value+=(portfolio_value*weights[a])*return_factor;
        }
        final_portfolio_values.push_back(current_value);
    }


sort(final_portfolio_values.begin(),final_portfolio_values.end());

int ind_95= static_cast<int>(sim*0.05);
int ind_99=static_cast<int>(sim*0.01);

result.var95 = portfolio_value - final_portfolio_values[ind_95];
result.var99 = portfolio_value- final_portfolio_values[ind_99];

// calculating mean loss 
double loss=0.0;
for (int i=0;i<ind_95;i++)
{
    loss+= portfolio_value-final_portfolio_values[i];
}
result.meanloss=loss/ind_95;

}

Simresult MonteCarloEngine::get_result()
{
    return result;
}

extern "C"{
    // python sends the data here
    MonteCarloEngine* MonteCarloEngine_new(int sim, int assets, 
                                           double* mean, double* cov, 
                                           double* weights, double portfolio_value)
    {
        vector<double>v_means(mean,mean+assets);
        vector<double> v_cov(cov, cov + (assets * assets));
        vector<double> v_weights(weights, weights + assets);

        return new MonteCarloEngine(sim, assets, v_means, v_cov, v_weights, portfolio_value);
    }

    //  Run the math
    void MonteCarloEngine_run(MonteCarloEngine* engine) {
        if (engine != nullptr) {
            engine->run();
        }
    }

    // Get the results struct
    Simresult MonteCarloEngine_getresult(MonteCarloEngine* engine) {
        if (engine != nullptr) {
            return engine->get_result();
        }
        return {0.0, 0.0, 0.0};  
    }

    // Destroy the object
    void MonteCarloEngine_delete(MonteCarloEngine* engine) {
        if (engine != nullptr) {
            delete engine;
        }
    }
}



