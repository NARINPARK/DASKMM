import numpy as np
import pandas as pd
import time
import math
import warnings
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as normal
from scipy.stats import mvn
from sklearn.cluster import KMeans

def update_params(ksi, sigma, lambd) :     
    dim=len(ksi)
    omega = sigma + lambd @ lambd.T 
    omega_inv = np.linalg.pinv(omega) 
    delta = np.eye(dim) - lambd.T @ omega_inv @ lambd
    return omega, omega_inv, delta


def sknm_pdf(z, w, ksi, sigma, lambd) :       
    n_comp = z.shape[1]
    dim = len(ksi)
    omega, omega_inv, delta = update_params(ksi, sigma, lambd)   
    probs = 2 ** dim * normal.pdf(z, mean=ksi, cov=omega)
    probs *= normal.cdf((lambd.T @ omega_inv @ (z - ksi).T).T, cov=delta)
    probs *= w
    return probs


def sknm_logpdf(z, w, ksi, sigma, lambd):             
    dim = z.shape[1]
    n_comp = len(ksi)
    log_prob = [] 
    for g in range(n_comp):
        omega, omega_inv, delta = update_params(ksi[g], sigma[g], lambd[g])  
        probs = np.log(2) * dim
        probs += normal.logpdf(z, mean=ksi[g], cov=omega, allow_singular=True)
        probs += normal.logcdf((lambd[g].T @ omega_inv @ (z - ksi[g]).T).T, cov=delta, allow_singular=True)
        probs += np.log(w[g])
        log_prob.append(probs)        
    log_prob=np.array(log_prob).T
    return log_prob
    
    
def sknm_rvs(n, w, ksi, sigma, lambd) :   
    dim = len(w)
    num = (n * w).astype(np.int32)
    val= []
    for i in range(dim):
        tau = np.abs(normal.rvs(cov=np.eye(dim), size= num[i]))
        value = normal.rvs(mean=ksi[i], cov=sigma[i], size=num[i]) + (lambd[i] @ tau.T).T
        val.append(value)
    val = np.concatenate(val)     
    return val


def sknm_initial(z, n_comp): #확인
    kmeans = KMeans(n_clusters=n_comp)
    kmeans.fit(z)
    cluster_labels = kmeans.labels_
    p = z.shape[1]
    w = np.array([np.sum(cluster_labels == i) for i in range(n_comp)]) / len(cluster_labels)
    ksi = kmeans.cluster_centers_
    sigma = np.zeros((n_comp, p, p))
    for c in range(n_comp):
        cluster_data = z[cluster_labels == c]
        sigma[c] = np.cov(cluster_data, rowvar=False)
    lambd = np.array([np.eye(p)*3 for _ in range(n_comp)])
    return w, ksi, sigma, lambd


import numpy as np
from scipy.stats import multivariate_normal

def mtn_mar1(xn, n, mean, sigma, lower, upper=None):
    if upper is None:
        upper = np.full(len(mean), np.inf)

    C = sigma
    A = np.linalg.inv(sigma)
    A_1 = np.delete(np.delete(A, n, axis=0), n, axis=1)  # (n-1) x (n-1)
    A_1_inv = np.linalg.inv(A_1)
    C_1 = np.delete(np.delete(C, n, axis=0), n, axis=1)  # (n-1) x (n-1)
    c_nn = C[n, n]  #  1x1
    c = np.delete(C, n, axis=0)[:, n]  # (n-1) x 1

    mu = mean
    mu_1 = np.delete(mean, n, axis=0)
    mu_n = mean[n]

    p = mvn.mvnun(lower=lower, upper=upper, means=mu, covar=C)[0]
    f_xn = np.zeros(len(xn))

    valid_indices = np.logical_and(lower[n] <= xn, xn <= upper[n])
    valid_indices = np.logical_and(valid_indices, ~np.isinf(xn))

    m = mu_1 + (xn[valid_indices] - mu_n) * c / c_nn
    f_xn[valid_indices] = np.exp(-0.5 * (xn[valid_indices] - mu_n) ** 2 / c_nn) * mvn.mvnun(lower=np.delete(lower, n, axis=0), upper=np.delete(upper, n, axis=0), means=m, covar=A_1_inv)[0]
    density = 1 / p * 1 / np.sqrt(2 * np.pi * c_nn) * f_xn
    
    return density


def mtn_mar2(xq, xr,q, r, mean, sigma,
             lower, upper = None):
    if upper is None :
        upper = np.full(len(mean), np.inf) 
        
    n = sigma.shape[0]
    N = len(xq) 
    alpha = mvn.mvnun(lower=lower, upper=upper,means=mean,covar=sigma)[0]
    
    if n == 2 :
        density = np.zeros(len(xq))
        indOut = np.logical_or(xq < lower[q], xq > upper[q]) | np.logical_or(xr < lower[r], xr > upper[r]) | np.isinf(xq) | np.isinf(xr)
        density[~indOut] = normal.pdf(x= np.column_stack((xq,xr))[~indOut], mean=mean[np.array([q,r])], cov=sigma[[q,r], :][:,[q,r]]) 
        density[~indOut] /= alpha
        return density
    
    SD = np.sqrt(np.diag(sigma))

    lower_normalized = (lower - mean) / SD
    upper_normalized = (upper - mean) / SD
    xq_normalized = (xq - mean[q-1]) / SD[q]
    xr_normalized = (xr - mean[r-1]) / SD[r]

    D = np.zeros((n, n))
    np.fill_diagonal(D, 1 / np.sqrt(np.diag(sigma)))
    R = D @ sigma @ D

    RQR = np.zeros((n-2, n-2))
    RINV = np.linalg.inv(R)
    WW = np.zeros((n-2, n-2))
    M1 = 0
    for i in range(n):
        if i != q and i != r:
            M1 += 1
            M2 = 0
            for j in range(n):
                if j != q and j != r:
                    M2 += 1
                    WW[M1-1, M2-1] = RINV[i-1, j-1]
    WW = np.linalg.inv(WW[:n-1,:n-1])
    for i in range(n-2):
        for j in range(n-2):
            RQR[i, j] = WW[i, j] / np.sqrt(WW[i, i] * WW[j, j])
    AQR = np.zeros((N, n-2))
    BQR = np.zeros((N, n-2))
    M2 = 0
    for i in range(n):
        if i != q and i != r:
            M2 += 1
            BSQR = (R[q, i] - R[q, r] * R[r, i]) / (1 - R[q, r]**2)
            BSRQ = (R[r, i] - R[q, r] * R[q, i]) / (1 - R[q, r]**2)
            RSRQ = (1 - R[i, q]**2) * (1 - R[q-1, r-1]**2)
            RSRQ = (R[i, r] - R[i, q] * R[q, r]) / np.sqrt(RSRQ)  # partial correlation coefficient R[r,i] given q

            AQR[:, M2-1] = (lower_normalized[i] - BSQR * xq_normalized - BSRQ * xr_normalized) / np.sqrt((1 - R[i, q]**2) * (1 - RSRQ**2))
            AQR[:, M2-1] = np.where(np.isnan(AQR[:, M2-1]), -np.inf, AQR[:, M2-1])

            BQR[:, M2-1] = (upper_normalized[i] - BSQR * xq_normalized - BSRQ * xr_normalized) / np.sqrt((1 - R[i, q]**2) * (1 - RSRQ**2))
            BQR[:, M2-1] = np.where(np.isnan(BQR[:, M2-1]), np.inf, BQR[:, M2-1])

    R2 = np.array([[1, R[q, r]],
                   [R[q, r], 1]])
    
    sigma2 = sigma[[q, r]][:, [q, r]]
    density = np.zeros(N)
    for i in range(N):
        if (xq[i] < lower[q] or xq[i] > upper[q] or
            xr[i] < lower[r] or xr[i] > upper[r] or
            np.isinf(xq[i]) or np.isinf(xr[i])):
            density[i] = 0
        else:
            prob = np.zeros(N)
            if (n - 2) == 1:
                prob[i] = mvn.mvnun(lower=AQR[i, :], upper=BQR[i, :], means=np.zeros(len(RQR)), covar=RQR.reshape([-1]))[0]
            else:
                prob[i] = mvn.mvnun(lower=AQR[i, :], upper=BQR[i, :], means=np.zeros(len(RQR)), covar=RQR.reshape([-1]))[0]

            density[i] = normal.pdf(x=np.column_stack((xq[i], xr[i])), mean=mean[[q, r]], cov=sigma2) * prob[i] / alpha

    return density


def eta_psi(mean, sigma, lower, upper = None):
    if upper is None :
        upper = np.full(len(mean), np.inf) 
    
    N = len(mean)
    TMEAN = np.zeros(N)
    TVAR = np.full((N, N), np.nan)
    a = lower - mean
    b = upper - mean
    lower = lower - mean
    upper = upper - mean
    zero_mean = np.zeros(N)

    F_a = np.zeros(N)
    F_b = np.zeros(N)
    for q in range(N):
        tmp = mtn_mar1(xn=np.column_stack([a[q], b[q]]).reshape([-1]), n=q, 
                       mean=zero_mean, sigma=sigma, lower=lower, upper=upper)
        F_a[q] = tmp[0]
        F_b[q] = tmp[1]

    TMEAN = sigma @ (F_a-F_b) 

    F2 = np.zeros((N, N))
    for q in range(N):
        for s in range(N):
            if q != s:
                d = mtn_mar2(xq=np.column_stack([a[q], b[q], a[q], b[q]]).reshape([-1]),
                             xr=np.column_stack([a[s], a[s], b[s], b[s]]).reshape([-1]), q=q, r=s,
                             mean=zero_mean, sigma=sigma, lower=lower, upper=upper)
                F2[q, s] = (d[0] - d[1]) - (d[2] - d[3])

    F_a_q = np.where(np.isinf(a), 0, a * F_a)
    F_b_q = np.where(np.isinf(b), 0, b * F_b)

    for i in range(N):
        for j in range(N):
            sum_ = 0
            for q in range(N):
                sum_ += sigma[i, q] * sigma[j, q] * sigma[q, q]**(-1) * (F_a_q[q] - F_b_q[q])
                if j != q :
                    sum2 = 0
                    for s in range(N):
                        tt = sigma[j, s] - sigma[q, s] * sigma[j, q] * sigma[q, q]**(-1)
                        sum2 += tt * F2[q, s]
                    sum2 = sigma[i, q] * sum2
                    sum_ += sum2
            TVAR[i, j] = sigma[i, j] + sum_ 
            
    TVAR -= TMEAN[:, np.newaxis] @TMEAN[:, np.newaxis].T
    TMEAN += mean
    TVAR += TMEAN[:, np.newaxis] @TMEAN[:, np.newaxis].T     
    return TMEAN, TVAR


def mtmvn_vec(z, lambd, omega_inv, ksi, delta):
    mu = (lambd.T @ omega_inv @ (z- ksi).T).T 
    sigma = np.tile(delta, (len(mu), 1, 1))
    alpha = np.array(list(map(mvn.mvnun, np.zeros_like(mu), np.ones_like(mu) * np.inf, mu, sigma)))[:,0]
    results = list(map(lambda args: eta_psi(*args), zip(mu, sigma, np.zeros_like(mu))))
    exp_x = np.array([result[0] for result in results])
    exp_xx = np.array([result[1] for result in results])
    
    alpha = np.where(np.isnan(alpha), 0, alpha)   
    for i in range(len(z)):
        if alpha[i] < 1e-08 : 
            mu_erc = exp_x[i,].reshape((-1, 1))
            exp_xx[i,] = (mu_erc * exp_x[i,].T) + (exp_x[i,] * mu_erc.T) - (mu_erc * mu_erc.T) + sigma[i,]
            
    return exp_x, exp_xx


def cal_probs(z, w, ksi, sigma, lambd):
    post_probs = np.array(list(map(sknm_pdf, [z] * len(w), w, ksi, sigma, lambd)))
    probs = np.sum(post_probs, axis=0)
    prob_dists = post_probs / probs
    return prob_dists


def ksi_gen(z, deom, dist_probs, lambd, exp_x):
    ksi = (np.sum(dist_probs*z, axis=0) - lambd @ np.sum(dist_probs * exp_x, axis=0)) /deom
    #ksi = np.sum((z - (lambd @ exp_x.T).T)* dist_probs, 0) / denom
    return ksi


def lam_gen(z, dist_probs, ksi, exp_x, exp_xx):
    lambd_denom = np.linalg.pinv(np.sum(exp_xx * dist_probs.reshape([-1, 1, 1]), 0))
    lambd = ((z - ksi).T @ (exp_x * dist_probs)) @ lambd_denom
    return lambd


def lam_diag(z, dist_probs, ksi, sigma, exp_x, exp_xx, dim):
    lambd_denom = np.linalg.pinv(np.linalg.pinv(sigma) * np.sum(exp_xx * dist_probs.reshape([-1, 1, 1]), 0))
    diag_lambd = lambd_denom @ (np.linalg.pinv(sigma) * (( dist_probs *exp_x).T @ (z - ksi)) )  @ np.ones(dim)
    lambd = np.diag(diag_lambd)
    return lambd


def sig_gen(z, denom, dist_probs, ksi, lambd, exp_x, exp_xx):
    normalized = (z - ksi - (lambd @ exp_x.T).T)
    sigma = (normalized.T @ ( normalized * dist_probs ) +
             lambd @ ((exp_xx * dist_probs.reshape([-1, 1, 1])).sum(0) - 
                      exp_x.T @ (exp_x * dist_probs)) @ lambd.T) / denom
    sigma[np.triu_indices_from(sigma,k=1)] = sigma.T[np.triu_indices_from(sigma,k=1)]
    return sigma


def sknm_em(z, n_comp, tol=1e-4, max_iter=100000):
    n,dim = z.shape
    w, ksi, sigma, lambd = sknm_initial(z, n_comp)
    log_lik = []
    log_prob = np.log(np.sum(np.array(list(map(sknm_pdf, [z] * len(w), w, ksi, sigma, lambd))),axis=0)).sum()
    log_lik.append(log_prob)
    for number in range(max_iter):
        # E step
        result = zip(*np.array(list(map( update_params, ksi, sigma, lambd))))
        omega, omega_inv, delta = map(np.array,result)
        result = [mtmvn_vec(z, lambd_i, omega_inv_i, ksi_i, delta_i) for 
                  lambd_i, omega_inv_i, ksi_i, delta_i in zip(lambd,omega_inv,ksi,delta)]
        exp_x, exp_xx = zip(*result)
        dist_probs = cal_probs(z, w, ksi, sigma, lambd)
        dist_list = np.array([dist_probs[g].reshape([-1, 1]) for g in range(dist_probs.shape[0])])
        # M step
        w = np.mean(dist_probs, axis=1)
        denom = np.sum(dist_probs, axis=1)
        
        ksi= np.array([ksi_gen(z, denom_i, dist_i, lambd_i, exp_i) for 
                  denom_i, dist_i, lambd_i, exp_i in zip(denom, dist_list, lambd, exp_x)])

        lambd = np.array([lam_gen(z, dist_i, ksi_i, exp_i, exp_ii) for 
                  dist_i, ksi_i, exp_i, exp_ii in zip(dist_list, ksi, exp_x, exp_xx)])

        sigma = np.array([sig_gen(z, denom_i, dist_i, ksi_i, lam_i, exp_i, exp_ii) for 
                  denom_i,dist_i, ksi_i, lam_i, exp_i, exp_ii in zip(denom, dist_list, ksi, lambd, exp_x, exp_xx)])

        new_log = np.log(np.sum(np.array(list(map(sknm_pdf, [z] * len(w), w, ksi, sigma, lambd))),axis=0)).sum()
        
        log_lik.append(new_log)
        if np.abs(log_lik[-1] - log_lik[-2]) <= tol:
            break

    return ksi, sigma, lambd, log_lik


def sknm_em_diag(z, n_comp, tol=1e-4, max_iter=100000):
    n,dim = z.shape
    w, ksi, sigma, lambd = sknm_initial(z, n_comp)
    log_lik = []
    log_prob = np.log(np.sum(np.array(list(map(sknm_pdf, [z] * len(w), w, ksi, sigma, lambd))),axis=0)).sum()
    log_lik.append(log_prob)   
    for number in range(max_iter):
        # E step
        result = zip(*np.array(list(map( update_params, ksi, sigma, lambd))))
        omega, omega_inv, delta = map(np.array,result)
        result = [mtmvn_vec(z, lambd_i, omega_inv_i, ksi_i, delta_i) for 
                  lambd_i, omega_inv_i, ksi_i, delta_i in zip(lambd,omega_inv,ksi,delta)]
        exp_x, exp_xx = zip(*result)
        dist_probs = cal_probs(z, w, ksi, sigma, lambd)
        dist_list = np.array([dist_probs[g].reshape([-1, 1]) for g in range(dist_probs.shape[0])])
        # M step
        w = np.mean(dist_probs, axis=1)
        denom = np.sum(dist_probs, axis=1)
        
        ksi= np.array([ksi_gen(z, denom_i, dist_i, lambd_i, exp_i) for 
                  denom_i, dist_i, lambd_i, exp_i in zip(denom, dist_list, lambd, exp_x)])

        lambd=np.array([lam_diag(z, dist_i, ksi_i, sigma_i, exp_i, exp_ii, dim) for 
                dist_i, ksi_i, sigma_i, exp_i, exp_ii in zip(dist_list, ksi, sigma, exp_x, exp_xx)])

        sigma = np.array([sig_gen(z, denom_i, dist_i, ksi_i, lam_i, exp_i, exp_ii) for 
                  denom_i,dist_i, ksi_i, lam_i, exp_i, exp_ii in zip(denom, dist_list, ksi, lambd, exp_x, exp_xx)])

        new_log = np.log(np.sum(np.array(list(map(sknm_pdf, [z] * len(w), w, ksi, sigma, lambd))),axis=0)).sum()
        log_lik.append(new_log)
        if np.abs(log_lik[-1] - log_lik[-2]) <= tol:
            break
    return ksi, sigma, lambd, log_lik



