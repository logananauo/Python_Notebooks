
import random 
import math

random.seed(42)


### function to run monte-carlo
def estimate_pi(num_samples):
    inside_count = 0
    for _ in range(num_samples):
        x = random.random() # random x in [0,1]
        y = random.random() # random y in [0,1]
        if x*x + y*y <= 1.0: # inside quarter circle
            inside_count += 1
    return 4 * inside_count / num_samples


### estimate pi with increasing sample sizes
for N in [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
    pi_est = estimate_pi(N)
    print(f'N={N:<8} -> pi_est = {pi_est:.5f}, error = {abs(pi_est - math.pi):.5f}')
