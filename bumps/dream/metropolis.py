"""
MCMC step acceptance test.
"""
from __future__ import with_statement

__all__ = ["metropolis", "metropolis_dr"]

from numpy import exp, sqrt, minimum, where, cov, eye, array, dot, errstate
from numpy.linalg import norm, cholesky, inv
from . import util




def paccept(logp_old, logp_try):
    """
    Returns the probability of taking a metropolis step given two
    log density values.
    """
    return exp(minimum(logp_try-logp_old, 0))


def metropolis(xtry, logp_try, xold, logp_old, step_alpha, jamiefile, jamiefile1, crossover, amt, amt_Needed):
    """
    Metropolis rule for acceptance or rejection

    Generates the next generation, *newgen* from::

        x_new[k] = x[k]     if U > alpha
                 = x_old[k] if U <= alpha

    where alpha is p/p_old and accept is U > alpha.

    Returns x_new, logp_new, alpha, accept
    """
    with errstate(under='ignore'):
        alpha = paccept(logp_try=logp_try, logp_old=logp_old)
        alpha *= step_alpha
    accept = alpha > util.rng.rand(*alpha.shape)
    logp_new = where(accept, logp_try, logp_old)
    ## The following only works for vectors:
    # xnew = where(accept, xtry, xold)
    xnew = xtry+0
    count=0
    count1=0
    std_dev_count=0
    std_dev=[]
    for i in range(len(xold[0])):
        for j in range (len(xold[0]*10)):
            std_dev_count+=xold[j][j]**2
        std_dev.append((std_dev_count/(len(xold[0]*10)-1))**.5)


    for i, a in enumerate(accept):
        if amt>amt_Needed-4:
            if a:

                count1=count1 +1
                xdiff_accept=[]
                for j in range(len(xnew[0])):
                    xdiff_accept.append(abs((xtry[i][j] - xold[i][j]) / std_dev[j]))


                for j in range(len(xdiff_accept)):
                    jamiefile1.write(str(xdiff_accept[j]) + ",\t")
                jamiefile1.write("\n")



        if not a:

            xnew[i] = xold[i]
            if amt>amt_Needed-4:
                count = count + 1
                xdiff_reject=[]

                for j in range(len(crossover[0])):
                    crossover[i][j]*=2
                for j in range(len(xnew[0])):
                    if xold[i][j]!=0: xdiff_reject.append(abs((xtry[i][j]-xold[i][j])/std_dev[j]))
                    else: xdiff_reject.append(0)


                for j in range(len(xdiff_reject)):
                    jamiefile.write(str(xdiff_reject[j])+",\t")



                #jamiefile.write(str(xold[i])+"\n"+str(xtry[i]))
                jamiefile.write("\n")

    if amt>amt_Needed-4:
        jamiefile.write("\n")
        jamiefile1.write("\n")
        jamiefile.write(str(count))
        jamiefile1.write(str(count1))
        jamiefile1.write("\n\n\n")
        jamiefile.write("\n\n\n")
        #assert False

    return xnew, logp_new, alpha, accept


def dr_step(x, scale):
    """
    Delayed rejection step.
    """

    # Compute the Cholesky Decomposition of X
    nchains, npars = x.shape
    r = (2.38/sqrt(npars)) * cholesky(cov(x.T) + 1e-5*eye(npars))

    # Now do a delayed rejection step for each chain
    delta_x = dot(util.rng.randn(*x.shape), r)/scale

    # Generate ergodicity term
    eps = 1e-6 * util.rng.randn(*x.shape)

    # Update x_old with delta_x and eps;
    return x + delta_x + eps, r


def metropolis_dr(xtry, logp_try, x, logp, xold, logp_old, alpha12, R):
    """
    Delayed rejection metropolis
    """

    # Compute alpha32 (note we turned x and xtry around!)
    alpha32 = paccept(logp_try=logp, logp_old=logp_try)

    # Calculate alpha for each chain
    l2 = paccept(logp_try=logp_try, logp_old=logp_old)
    iR = inv(R)
    q1 = array([exp(-0.5*(norm(dot(x2-x1, iR))**2 - norm(dot(x1-x0, iR))**2))
                for x0, x1, x2 in zip(xold, x, xtry)])
    alpha13 = l2*q1*(1-alpha32)/(1-alpha12)

    accept = alpha13 > util.rng.rand(*alpha13.shape)
    logp_new = where(accept, logp_try, logp)
    ## The following only works for vectors:
    # xnew = where(accept, xtry, x)
    xnew = xtry+0
    for i, a in enumerate(accept):
        if not a:
            xnew[i] = x[i]

    return xnew, logp_new, alpha13, accept
