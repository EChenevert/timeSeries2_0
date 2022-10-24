import numpy as np

def getgamma(eigvals, a):
    tosum = []
    for e in eigvals:
        tosum.append(e/(a+e))
    return np.sum(tosum)


def getB(gamma, mN, t, phi):
    N = len(phi)
    tosum = []
    for i in range(len(t)):
        tosum.append((t[i] - mN.T@phi[i, :])**2)
    scalar = np.sum(tosum)
    return (N - gamma)/scalar


def iterative_prog_wPrior(phi, t, prior_vect):
    """
    I believe this is the hyperparameter tuning using the train set
    :param B: random initial beta hyperparameter
    :param a: random initial alpha hyperparameter
    :param phi: training data matrix
    :param t: training target vector
    :return: ideal beta, alpha, and effective lambda
    """
    B = np.random.uniform(0, 10)
    a = np.random.uniform(0, 10)
    Blist = [B]
    alist = [a]
    itr = 0
    switch = 'off'
    while switch == 'off':
        I = np.identity(len(phi[0, :]))  # make identity matrix the length of the input data's columns
        std = np.std(prior_vect)
        I = std*I
        m0 = np.mean(prior_vect)

        SN = np.linalg.inv(a*I + B*(phi.T@phi))  # stays the same --> using infinitely broad prior
        # S0 = 1  # is that right? for standard normal distribution.....????
        mNfromData = B*(SN@(phi.T@t))
        priormeanVect = np.full(shape=len(mNfromData), fill_value=m0, dtype=float)
        mN = np.add(priormeanVect, mNfromData)

        # aI = a*I
        # aIm0 = aI@m0
        # Bphit = np.expand_dims(B*(phi.T@t), axis=1)
        # aIplusBphit = np.add(aIm0, Bphit)
        # mN = SN@aIplusBphit

        eigs_logLHessian = np.linalg.eigvals(B*(phi.T@phi))  # derived by taking logL of Hessian M to max evidence
        gamma = getgamma(eigs_logLHessian, a)
        B = getB(gamma, mN, t, phi)
        Blist.append(B)
        a = gamma/(mN.T@mN)
        alist.append(a)
        itr += 1
        if abs(Blist[itr] - Blist[itr-1]) < 0.000001 and abs(alist[itr] - alist[itr-1]) < 0.000001:  # DAMN this seems crazy
            switch = 'on'
        if itr > 1000:
            switch = 'on'
    return B, a, a/B, itr



def iterative_prog(phi, t):
    """
    I believe this is the hyperparameter tuning using the train set
    :param B: random initial beta hyperparameter
    :param a: random initial alpha hyperparameter
    :param phi: training data matrix
    :param t: training target vector
    :return: ideal beta, alpha, and effective lambda
    """
    B = np.random.uniform(0, 10)
    a = np.random.uniform(0, 10)
    Blist = [B]
    alist = [a]
    itr = 0
    switch = 'off'
    while switch == 'off':
        I = np.identity(len(phi[0, :]))  # make identity matrix the length of the input data's columns
        SN = np.linalg.inv(a*I + B*(phi.T@phi))  # the dimensions of the covariance matrix should be the same # as the data's columns.Right?
        mN = B*(SN@(phi.T@t))
        eigs_logLHessian = np.linalg.eigvals(B*(phi.T@phi))  # derived by taking logL of Hessian M to max evidence
        gamma = getgamma(eigs_logLHessian, a)
        B = getB(gamma, mN, t, phi)
        Blist.append(B)
        a = gamma/(mN.T@mN)
        alist.append(a)
        itr += 1
        if abs(Blist[itr] - Blist[itr-1]) < 0.000001 and abs(alist[itr] - alist[itr-1]) < 0.000001:  # DAMN this seems crazy
            switch = 'on'
        if itr > 1000:
            switch = 'on'
    return B, a, a/B, itr


def leastSquares(lam, phi, t):
    I = np.identity(len(phi[0, :]))  # dim of I are equal to the # of features in phi
    whatdis = np.linalg.inv(lam*I + phi.T@phi)
    w = whatdis@phi.T@t
    return w


def samplePrior(mean, sigma, size):
    """Jus a function to sample a mean vector to then input """
    return np.random.normal(mean, sigma, size=(size, 1))


def returnMSE(phi, w, t):
    N = len(phi)
    tosum = []
    for i in range(len(t)):
        tosum.append((phi[i, :]@w - t[i])**2)
    summed = np.sum(tosum)
    return summed/N


def returnMAE(phi, w, t):
    N = len(phi)
    tosum = []
    for i in range(len(t)):
        tosum.append(phi[i, :]@w - t[i])
    summed = np.sum(tosum)
    return abs(summed/N)

def calculate_log_evidence(phi, a, B, t):
    """  """
    # t = np.expand_dims(t, axis=1)
    I = np.identity(len(phi[0, :]))  # make identity matrix the length of the input data's columns
    SN = np.linalg.inv(a*I + B*(phi.T@phi))
    mN = B*(SN@(phi.T@t))
    M = len(mN)
    N = len(phi)
    term_ls = []

    # I changed this chunk a bit
    for i in range(N):
        term_ls.append(t[i] - (phi[i, :]@mN))
    EmNi = B/2 * (np.sum(term_ls)**2) + a/2 * (mN.T@mN)
    A = a*np.identity(len(phi[0, :])) + B*(phi.T@phi)
    detA = np.linalg.det(A)
    return M/2 * np.log(a) + N/2 * np.log(B) - EmNi - 1/2 * np.log(detA) - N/2 * np.log(2*np.pi)




