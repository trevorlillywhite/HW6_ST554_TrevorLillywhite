# Homework 6 NCSU ST 554
# Author: Trevor Lillywhite
# Due Date: March 10, 2026 (extended to March 11)

# Part II - Messing with Classes
# Question 5

# Import relevant modules
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import numpy.typing as npt      # Will allow us to type cast "ArrayLike"
from sklearn import linear_model

# Create class with specified attributes
class SLR_slope_simulator:
    def __init__(self, beta_0: float, beta_1: float, 
                 x: npt.ArrayLike, sigma: float, seed: int):
        self.beta_0 = beta_0            # Directly passed as argument
        self.beta_1 = beta_1            # Directly passed as argument
        self.sigma = sigma              # Directly passed as argument
        self.x = x                      # Directly passed as argument
        self.n = len(x)                 # Determined from argument
        self.rng = default_rng(seed)    # Determined from argument
        self.slopes = []                # Empty list until we append to it
    
    def _generate_data(self):               # Private method: Generate y values
        y = self.beta_0 + \
                 self.beta_1*self.x + \
                 self.rng.normal(loc=0.0, scale=self.sigma, size=self.n)
        return self.x, y
    
    def _fit_slope(self, x, y):               # Private method: Fit slope using SLR
        reg = linear_model.LinearRegression()   # Create a reg object
        fit = reg.fit(x.reshape(-1, 1), y)      # Fit SLR model to data
        return fit.coef_[0]                     # Ignores fitted intercept value

    def run_simulations(self, m: int = 10000):  # Non-private method to fit multiple slopes
        for _ in range(m):                      # Repeat m times
            x_temp, y_temp = self._generate_data()      # Temporarily store x, y
            slope_i = self._fit_slope(x_temp, y_temp)   # Fit generated data
            self.slopes.append(slope_i)
        return None

    def plot_sampling_distribution(self):
        if len(self.slopes) == 0:               # Check for empty list
            print("You can't do that! Call run_simulations() first.")
        else:                                   # Plot histogram
            plt.hist(self.slopes, edgecolor='white')    
            plt.title('Histogram of Slope Samples')
            plt.xlabel('Slope Value (Bin)')
            plt.ylabel('Frequency')
            plt.show()
        return None

    def find_prob(self, value: float, sided: str):
        if len(self.slopes) == 0:               # Check for empty list
            print("You can't do that! Call run_simulations() first.")
        else:                                   # Calculate probability
            array_slopes = np.array(self.slopes)
            if sided == 'above':                # Calculate one-tailed upper prob
                bool_prob = array_slopes > value
                prob = bool_prob.mean().item()
            elif sided == 'below':              # Calculate one-tailed lower prob
                bool_prob = array_slopes < value
                prob = bool_prob.mean().item()
            elif sided == 'two-sided':          # Calculate two-tailed probability
                if value > np.median(array_slopes):
                    bool_prob = array_slopes > value
                    prob = 2 * bool_prob.mean().item()
                else:
                    bool_prob = array_slopes < value
                    prob = 2 * bool_prob.mean().item()
            else:                               # Error - print warning
                print('Invalid "sided" argument. Must be "above", "below", or "two-sided".')
                return                          # End method early (erroneous arg)
            return round(prob, 4)   # Return probability rounded to 4 decimals
        
# Test functionality    
def main():
    print('Create an instance with specified arguments')
    test1 = SLR_slope_simulator(beta_0 = 12, 
                       beta_1 = 2,
                       x = np.array(list(np.linspace(start=0, stop=10, num=11))*3),
                       sigma = 1, 
                       seed = 10)
    print('\n\n')

    print('Call plot_sampling_distribution() - should return an error message')
    test1.plot_sampling_distribution()
    print('\n\n')

    print('Run 10,000 simulations')
    test1.run_simulations(m=10000)
    print('\n\n')

    print('Plot the sampling distribution')
    test1.plot_sampling_distribution()
    print('\n\n')

    print('Approximate the two-sided probability of being larger than 2.1')
    print(test1.find_prob(value=2.1, sided='two-sided'))
    print('\n\n')

    print('Print out simulated slopes using the object attribute')
    print(test1.slopes)

if __name__ == "__main__":
    main()