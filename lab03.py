
# call_count = 0
# def track_calls (call_count=None):
#     call_count += 1
#     return call_count
# print(track_calls(call_count))
def describe_stats (*args):
    #function to find the count of numbers, mean, minimum, and max and return those values
    count = len(args)
    mean = sum(args)/len(args)
    minimum = min(args)
    maximum = max(args)
    return count, mean, minimum, maximum

print(describe_stats(14,6,5,13,2,10))

import numpy as np
from scipy.stats import norm
# from statsmodels.genmod.families.links import sqrt


def confidence_interval(data, confidence=0.95):
    n = len(data) #gets the count of samples
    x_bar = np.mean(data)                                #mean
    s = np.std(data, ddof= 1)                            # sample standard deviation
    z = abs(norm.ppf((1-confidence)/2))                 #two-sided distribution bc divided by 2
    ci_lower = x_bar - (z * (s/np.sqrt(n)))
    ci_upper = x_bar + (z * (s/np.sqrt(n)))

    return float(ci_lower), float (ci_upper)            #typecase the upper and lower CIs as float

scores = np. array ([72 , 85 , 90 , 65 , 78 , 88])
print(confidence_interval (scores))
#  Expected  ( rounded ):  (71.82 ,  87.52 )
print(confidence_interval (scores , confidence = 0.99)) # Expected : a wider interval than the 95% case
