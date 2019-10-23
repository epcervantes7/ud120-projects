#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    res=abs(net_worths-predictions)
    err, a,n = zip(*sorted(zip(res, ages,net_worths)))
    cleaned_data=(a[:81],n[:81],err[:81])

    ### your code goes here

    
    return cleaned_data

