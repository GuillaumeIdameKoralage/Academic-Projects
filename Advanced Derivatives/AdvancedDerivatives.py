import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
from scipy import optimize


def implied_vol(price, S, r, b, t, T, K, CallPut='Call'):
    return optimize.root_scalar(lambda x: BlackScholes(S,r,b,x,t,T,K)-price, bracket=[0.001,10]).root


def geometric_asian_bsm(S,r,b,sigma, t,T,K,CallPut="Call", Greek="Price"):
    b_G = 0.5 * b - sigma*sigma / 12
    sigma_G = sigma / np.sqrt(3)
    return BlackScholes(S, r, b_G, sigma_G, t, T, K, CallPut="Call", Greek="Price")
    
#def arithmetic_asian_turnwake(S,r,b,sigma, t,T,K):
#   q = r - b
#    M1 = S * (np.exp((r-q)*T)-1((r-q)))
    
def trigger_forward_bsm(S,r,b,sigma,t,T,K,H,CallPit='Call',Greek='Price'):
    tau = T-t
    sqrtTau = np.sqrt(tau)
    vol = sigma * sqrtTau
    d1 = ( np.log(S/H) + (b+0.5*sigma*sigma)*tau ) / vol
    d2 = d1 - vol
    return S * np.exp((b-r)*tau) * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)

def forward_start_bsm(S,r,b,sigma,t,T,alpha,CallPit='Call',Greek='Price'):
    tau = T-t
    sqrtTau = np.sqrt(tau)
    vol = sigma * sqrtTau
    d1 = ( np.log(1/alpha) + (b+0.5*sigma*sigma)*tau ) / vol
    d2 = d1 - vol
    return S * np.exp((b-r)*t) *( np.exp((b-r)*tau) * norm.cdf(d1) - alpha * np.exp(-r*tau) * norm.cdf(d2))
    
    

def barrier_option_AB(S, r, b, vol, T, K, x, phi, eta):    
    return phi * S * np.exp((b-r)*T) * norm.cdf(phi*x) - phi*K*np.exp(-r*T)*norm.cdf(phi*x-phi*vol)   
def barrier_option_CD(S, r, b, vol, T, H, K, y, mu, phi, eta):
    return phi * S * np.exp((b-r)*T)* (H/S)**(2*(mu+1))*norm.cdf(eta*y) - phi*K*np.exp(-r*T)*(H/S)**(2*mu) * norm.cdf(eta*y-eta*vol)

def barrier_option_bsm(S, r, b, sigma, t, T, K, H, CallPut="DICall", Greek="Price"):
    tau = T-t
    sqrtTau = np.sqrt(tau)
    vol = sigma * sqrtTau
    sigma2 = sigma * sigma
    
    mu =  ( b - 0.5 * sigma2 ) / sigma2
    lam = np.sqrt( mu*mu + 2 * r / sigma2 )
    
    x1 = np.log(S/K) / vol + ( 1 + mu ) * vol
    x2 = np.log(S/H) / vol + ( 1 + mu ) * vol

    y1 = np.log(H*H/S/K) / vol + ( 1 + mu ) * vol
    y2 = np.log(H/S) / vol + ( 1 + mu ) * vol
    
    z = np.log(H/S) / vol + lam * vol

    if CallPut=='DICall':
        if K > H:
            C = barrier_option_CD(S, r, b, vol, T, H, K, y1, mu, 1, 1)
            return C
        else:
            A = barrier_option_AB(S, r, b, vol, T, K, x1, 1, 1)
            B = barrier_option_AB(S, r, b, vol, T, K, x2, 1, 1)
            D = barrier_option_CD(S, r, b, vol, T, H, K, y2, mu, 1, 1)
            return A - B + D
    if CallPut=='UOCall':
        if K > H:
            return 0
        else:
            eta = -1
            phi = 1
            A = barrier_option_AB(S, r, b, vol, T, K, x1, phi, eta)
            B = barrier_option_AB(S, r, b, vol, T, K, x2, phi, eta)
            C = barrier_option_CD(S, r, b, vol, T, H, K, y1, mu, phi, eta)          
            D = barrier_option_CD(S, r, b, vol, T, H, K, y2, mu, phi, eta)


            return A - B + C - D
        





def perpetual_american_bsm(S, H, r, sigma, greek='Price'):
    if S < H:
        if greek=='Price':
            return S / H
        if greek=='Delta':
            return 1 / H
        if greek=='Gamma':
            return 0
        if greek=='Vega':
            return 0
        if greek=='Theta':
            return 0
        if greek=='Rho':
            return 0
    # S > H
    else:
        if greek=='Price':
            return (S / H)**(-2*r/sigma/sigma) 
        if greek=='Delta':
            return -2*r/sigma/sigma / H * (S / H)**(-2*r/sigma/sigma-1)
        if greek=='Gamma':
            return (2*r/sigma/sigma+1) * 2*r/sigma/sigma / H**2 * (S / H)**(-2*r/sigma/sigma-1)
        if greek=='Theta':
            return 0
        if greek=='Rho':
            return -2/sigma**2*np.log(S/H) * perpetual_american_bsm(S, H, r, sigma, greek='Price')
        if greek=='Vega':
            return 2*r/sigma**3*np.log(S/H) * perpetual_american_bsm(S, H, r, sigma, greek='Price')


def BlackScholes(S, r, b, sigma, t, T, K, CallPut="Call", Greek="Price"):
    """
    Prices the BlackScholes Greeks
    """
    tau = T-t
    if tau == 0:
        if Greek=="Price":
            if CallPut=="Call":
                return max(S-K,0)
            else:
                return max(K-S,0)
        elif Greek=="Delta":
            if CallPut=="Call":
                return np.heaviside(S-K,0)
            else:
                return np.heaviside(K-S,0)
        else:
            return np.nan
    else:        
        sqrtTau = np.sqrt(tau)
        vol = sigma * sqrtTau
        d1 = ( np.log(S/K) + (b+0.5*sigma*sigma)*tau ) / vol
        d2 = d1 - vol
        if Greek=="Price":
            if CallPut=="Call":
                return S * np.exp((b-r)*tau) * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
            else:
                return -S * np.exp((b-r)*tau) * norm.cdf(-d1) + K * np.exp(-r*tau) * norm.cdf(-d2)
        elif Greek=="Delta":
            if CallPut=="Call":
                return np.exp((b-r)*tau) * norm.cdf(d1)
            else:
                return np.exp((b-r)*tau) * (norm.cdf(d1)-1)
        elif Greek=="Gamma":
            return np.exp((b-r)*tau) * norm.pdf(d1) / S / vol
        elif Greek=="Vega":
            return S * np.exp((b-r)*tau) * norm.pdf(d1) * sqrtTau
        elif Greek=="Volga":
            return S * np.exp((b-r)*tau) * norm.pdf(d1) * sqrtTau * d1 * d2 / sigma
        elif Greek=="Theta":
            temp = -0.5 * S * np.exp((b-r)*tau) * norm.pdf(d1) * sigma / sqrtTau
            if CallPut=="Call":
                return temp - (b-r) * S * np.exp((b-r)*tau) * norm.cdf(d1) - r * K * np.exp(-r*tau) * norm.cdf(d2)
            else:
                return temp + (b-r) * S * np.exp((b-r)*tau) * norm.cdf(-d1) + r * K * np.exp(-r*tau) * norm.cdf(-d2)
        elif Greek=="Rho":
            if CallPut=="Call":
                return tau * K * np.exp(-r * tau) * norm.cdf(d2)
            else:
                return -tau * K * np.exp(-r * tau) * norm.cdf(-d2)
        elif Greek=="Rho2":
            if CallPut=="Call":
                return tau * S * np.exp((b-r) * tau) * norm.cdf(d1)
            else:
                return -tau * S * np.exp((b-r) * tau) * norm.cdf(-d1)
        elif Greek=="Fwd":
            return S * np.exp(b*tau)



def black_scholes_digital(S, r, b, sigma, t, T, K, CallPut="Call", Greek="Price"):
    """
    Prices the BlackScholes Greeks
    """
    tau = T-t
    if CallPut == 'Call':
        phi = 1
    else:
        phi = -1
    if tau == 0:
        if Greek=="Price":
            return np.heaviside(phi*(S-K),0)
        elif Greek=="Delta":
            return 0
        else:
            return np.nan
    else:        
        sqrtTau = np.sqrt(tau)
        vol = sigma * sqrtTau
        d1 = ( np.log(S/K) + (b+0.5*sigma*sigma)*tau ) / vol
        d2 = d1 - vol
        if Greek=='Price':
            return np.exp(-r*tau) * norm.cdf(phi*d2)
        elif Greek=="Delta":
            return phi * np.exp(-r*tau) * norm.pdf(d2) / S / vol
        elif Greek=="Gamma":
            return -phi*np.exp(-r*tau) * norm.pdf(d2)* d1 / S / S / vol / vol
        elif Greek=="Vega":
            return -phi * np.exp(-r*tau) * norm.pdf(d2) * d1 / sigma
        elif Greek=="Theta":
            return np.exp(-r*tau) * ( 0.5 * phi * norm.pdf(d2) * (d1 - 2* b * sqrtTau / sigma) / tau + 
                 r * norm.cdf(phi*d2) )
        elif Greek=="Rho":
            return np.exp(-r * tau) * ( phi * sqrtTau * norm.pdf(d2) / sigma - tau * norm.cdf(phi*d2) )
        elif Greek=="Fwd":
            return S * np.exp(b*tau)


def black_scholes_onetouch(S, r, b, sigma, t, T, K, CallPut="Call", Greek="Price"):
    """
    Prices the BlackScholes Greeks
    """
    tau = T-t
    if CallPut == 'Call':
        phi = 1
    else:
        phi = -1
    if tau == 0:
        if Greek=="Price":
            return 0
        elif Greek=="Delta":
            return 0
        else:
            return np.nan
    else:        
        sqrtTau = np.sqrt(tau)
        vol = sigma * sqrtTau
        d1 = ( np.log(S/K) + (b+0.5*sigma*sigma)*tau ) / vol
        d2 = d1 - vol
        x1 = ( np.log(K/S) - (b-0.5*sigma*sigma)*tau ) / vol
        x2 = ( np.log(K/S) + (b-0.5*sigma*sigma)*tau ) / vol
        f = (K/S)**(2*b/sigma/sigma -1 )
        if Greek=='Price':
            return np.exp(-r*tau) * ( norm.cdf(x1) + f*norm.cdf(x2))
        elif Greek=="Delta":
            return phi * np.exp(-r*tau) * norm.pdf(d2) / S / vol
        elif Greek=="Gamma":
            return -phi*np.exp(-r*tau) * norm.pdf(d2)* d1 / S / S / vol / vol
        elif Greek=="Vega":
            return -phi * np.exp(-r*tau) * norm.pdf(d2) * d1 / sigma
        elif Greek=="Theta":
            return np.exp(-r*tau) * ( 0.5 * phi * norm.pdf(d2) * (d1 - 2* b * sqrtTau / sigma) / tau + 
                 r * norm.cdf(phi*d2) )
        elif Greek=="Rho":
            return np.exp(-r * tau) * ( phi * sqrtTau * norm.pdf(d2) / sigma - tau * norm.cdf(phi*d2) )
        elif Greek=="Fwd":
            return S * np.exp(b*tau)




def monte_carlo_step(p, rv, antithetic=False, importance_sampling=False):
    if importance_sampling:
         rv = np.abs(rv)       
    logS = p.m.log_spot + p.log_drift_tau + p.sigma_root_tau * rv    
    if antithetic:
        logS = np.concatenate((logS, monte_carlo_step(p,-rv,False)), axis=None)
    return logS

def MonteCarloGreeks(num_paths, p, sens_key, antithetic=False, importance_sampling=False):
    """
    Monte Carlo simulation stock price
    $\log S_i(T) = \log S(t) + (b-0.5*sigma^2)(T-t) \sigma \sqrt{T-t} Z_i
    Greeks are also calculated bumping the initial parameter and then using the random numbers
    """

    rv = np.random.normal(size=num_paths) 
    ST = np.exp(monte_carlo_step(p, rv, antithetic, importance_sampling))
    P = p.discount * p.d.payoff(ST)

   
    p_sens = {}
    S_sens={}
    P_sens = {}
    for k in sens_key:
        p_sens[k[0]] = copy.deepcopy(p)
        p_sens[k[0]].bump(k[0], k[1], k[2])
        S_sens[k[0]] = np.exp(monte_carlo_step(p_sens[k[0]], rv, antithetic))
        bump = p.get_bump(k[0], k[1], k[2])
        P_sens[k[0]] = p_sens[k[0]].sensitivity(S_sens[k[0]], P, bump)
    
    x = [ST, P]+[ P_sens[k[0]] for k in sens_key] 
    return x

    

#########################################################################################################
# Function to add teh option parameters to the plot
def display_option_params(ax, pricer, display_loc, x_label=None, y_label=None):
    if x_label != None:
        ax.set_xlabel(x_label)
    if y_label != None:
        ax.set_ylabel(y_label)
        
    yval = np.linspace(display_loc[1],display_loc[1]-0.25,5)
    ax.text(display_loc[0], yval[0], 'K='+str(pricer.d.data['Strike']), transform=ax.transAxes, fontsize=10)
    ax.text(display_loc[0], yval[1], '$\sigma$='+'%.3f' % pricer.m.data['Sigma'], transform=ax.transAxes, fontsize=10)
    ax.text(display_loc[0], yval[2], 'T='+'%.3f' % pricer.d.data['Expiry'], transform=ax.transAxes, fontsize=10)
    ax.text(display_loc[0], yval[3], 'r='+str(pricer.m.data['Rate']), transform=ax.transAxes, fontsize=10)
    ax.text(display_loc[0], yval[4], 'q='+str(pricer.m.data['Div']), transform=ax.transAxes, fontsize=10)
    

#########################################################################################################
# Function to plot a delta hedged portfoliio
def delta_hedged_portfolio(pricer, xmin, xmax, xpoints, save_plot=False, file1='DeltaHedge.pdf', file2='HedgePnL.pdf'):
    x = np.linspace(xmin, xmax, xpoints)
    dx = x - pricer.m.data['Spot']
    z = pricer.price(greek='Delta') * ( x - pricer.m.data['Spot']) + pricer.price(greek='Price')
    y = [ pricer.price_key('Spot',val) for val in x ]
    pricer.m.data['Today'] = 0.1
    y2 = [ pricer.price_key('Spot',val) for val in x ]  

    df = pd.DataFrame(index=x, data=y, columns=['Option'])
    df['Hedge']= z
    df['Decay'] = y2
    ax = df.plot(title='Option and Delta Hedge')
    display_option_params(ax, pricer, [0.8,0.3], x_label='Stock Price')
    if save_plot:
        plt.savefig(file1)
    plt.show()
    plt.close()
    
    df2 = pd.DataFrame(index=x, data=y-z, columns=['HedgedOption'])
    df2['Decay'] = y2 - z
#    df2['Zero'] = np.zeros(xpoints)
    ax = df2.plot(legend=True, title='Delta Hedged Option')
    ax.plot(x, np.zeros(xpoints))
    display_option_params(ax, pricer, [0.015,0.5], x_label='Stock Price', y_label='PnL')
    if save_plot:
        plt.savefig(file2)   
    plt.show()
    plt.close()

    df3 = pd.DataFrame(index=dx, data=y2-z, columns=['PnL'])
#    df3['Zero'] = np.zeros(xpoints)
    ax = df3.plot(legend=True, title='Hedging PnL')
    ax.set_xlabel('Stock Price Move')
    display_option_params(ax, pricer, [0.4,0.9])
    ax.plot(dx, np.zeros(xpoints))
    plt.show()
    plt.close()
    
#    ax.set_xlabel(xlabel)
#    ax.set_ylabel(ylabel)


def get_transaction_cost_vol(sigma, cost, delta_t, plus_minus):
    return sigma * np.sqrt(1+plus_minus * np.sqrt(2/np.pi/delta_t)*cost/sigma)    
    
    

#########################################################################################################
# Function to simulate delta hedging an option    
def delta_hedging( pricer, real_world_model, start_time, end_time, cost, num_steps, num_paths=1, num_samples=1, plus_minus = -1, write_file=False, filename='DeltaHedging' ):
    """

    Parameters
    ----------
    pricer : BlackScholesPricer
        Contains the model and derivative payoff that we are hedging
    real_world_model : BlackScholesModel
        Contains the model that we use to simulate the market moves
    start_time : float
        Start time of the simulation
    end_time : flat
        end time of the simulation
    cost : float
        the proportional bid-offer cost of trading
    num_steps : int
        number of time steps in a single path
    num_paths : TYPE, optional
        Number of paths we run. The default is 1.
    num_samples : TYPE, optional
        Number of sets of paths that we run. The default is 1.

    Returns
    -------
    None.

    """

    
#Generate the normal random variables that we need
    rv = np.random.normal(size=(num_samples,num_paths,num_steps))
    
# Set up the containers to hold the values.  
# We need one more data point than the number of steps in order to hold the initial value.
    log_asset = np.zeros( (num_samples, num_paths, num_steps+1) )
    asset = np.zeros( (num_samples, num_paths, num_steps+1) )
    deriv_price = np.zeros( (num_samples, num_paths, num_steps+1) )
    asset_weight = np.zeros( (num_samples, num_paths, num_steps+1) )
    bank_weight = np.zeros( (num_samples, num_paths, num_steps+1) )
    pv = np.zeros( (num_samples, num_paths, num_steps+1) )
    pnl = np.zeros( (num_samples, num_paths, num_steps+1) )
    trading_cost = np.zeros( (num_samples, num_paths, num_steps+1) )
    initial_hedge_cost = np.zeros( (num_samples, num_paths, num_steps+1) )
                   
# Set up the time variables
    time_step = np.linspace(start_time, end_time, num_steps+1)
    delta_t = (end_time - start_time) / num_steps
    drift_step = real_world_model.log_drift * delta_t
    vol_step = real_world_model.data['Sigma'] * np.sqrt(delta_t)
    bank_accrual = np.exp(pricer.m.data['Rate']*delta_t)
    div_accrual = np.exp(pricer.m.data['Div']*delta_t)       


# Record the initial values
    S_0 = pricer.m.data['Spot']
    logS_0 = pricer.m.log_spot
    t_0 = real_world_model.data['Today']
    C_0 = - plus_minus * pricer.price(greek='Price')
    weight_0 = plus_minus * pricer.price(greek='Delta')
    beta_0 = -C_0 - weight_0 * S_0
    cost_0 = -0.5 * abs(weight_0) * cost * S_0
    vega_0 = pricer.price(greek='Vega')
    df_list = []
    pv_list = []
      
# Loop over the random variables creating the paths     
    for sample in range(0,num_samples):
        for path in range(0,num_paths):
# Set the initial values
            log_asset[sample,path,0] = logS_0
            asset[sample,path,0] = S_0
            deriv_price[sample,path,0] = C_0
            asset_weight[sample,path,0] = weight_0
            bank_weight[sample,path,0] = beta_0
            initial_hedge_cost[sample, path, 0] = cost_0
            
            for step in range(1, num_steps+1):
# Move the asset forward in time
                log_asset[sample,path,step] = log_asset[sample,path,step-1] + drift_step + vol_step * rv[sample,path,step-1]
                pricer.update(log_asset[sample,path,step], time_step[step] )
                asset[sample,path,step] = pricer.m.data['Spot']
# Calculate the new derivative price and delta
                deriv_price[sample,path,step] = - plus_minus * pricer.price(greek='Price')
                asset_weight[sample,path,step] = plus_minus * pricer.price(greek='Delta')
# Update the bank account
                bank_weight[sample,path,step] = bank_accrual * bank_weight[sample,path,step-1] \
                    + (div_accrual * asset_weight[sample,path,step-1] - asset_weight[sample,path,step]) * pricer.m.data['Spot'] 
# Update the trading_cost account
                trading_cost[sample, path, step] = trading_cost[sample,path,step-1] * bank_accrual \
                    - 0.5*abs(asset_weight[sample,path,step] - asset_weight[sample,path,step-1]) * pricer.m.data['Spot'] * cost

                initial_hedge_cost[sample, path, step] = initial_hedge_cost[sample,path,step-1] * bank_accrual 


# Calculate the PV and the PnL of the portfolio
                pv[sample,path,step] = deriv_price[sample,path,step] + asset_weight[sample,path,step] * asset[sample,path,step] \
                    + bank_weight[sample,path,step]  
                pnl[sample,path,step] = pv[sample,path,step] - pv[sample,path,step-1]


#Process the data
            df_list.append(pd.DataFrame(index=time_step))
            df_list[path]['PV'] = pv[0,path,:]
            df_list[path]['PnL'] = pnl[0,path,:]
            df_list[path]['Deriv'] = deriv_price[0,path,:]
            df_list[path]['Asset'] = asset[0,path,:]
            df_list[path]['AssetWeight'] = asset_weight[0,path,:]
            df_list[path]['BankWeight'] = bank_weight[0,path,:]   
            df_list[path]['TradingCost'] = trading_cost[0,path,:]
            pv_list.append( pv[sample,path,num_steps] )
    
#    df_list[0]['PV'].plot(title='PnL')
#    plt.show()
#    df_list[0]['Asset'].plot(title='Asset')
#    plt.show()

# Plot of PnL distribution without trading costs
    df_pv = pd.DataFrame(data=pv[0,:,-1],columns=['PV'])
    pv_mean = df_pv.mean()[0]
    pv_std = df_pv.std()[0]
    ax = df_pv.plot.hist(legend=False, title='Impact of Discrete Hedging', bins=50, density=True )   
    ax.text(0.05, .9, 'mean='+'%.3f' % pv_mean, transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .85, 'std='+'%.3f' % pv_std, transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .80, 'Hedges='+str(num_steps), transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .75, 'Paths='+str(num_paths), transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .7, 'Option='+'%.3f' % C_0,transform=ax.transAxes, fontsize=10)
    ax.text(0.8, .9, 'Price='+str(real_world_model.data['Spot']), transform=ax.transAxes, fontsize=10)
    display_option_params(ax, pricer, [0.8,.85], x_label='Portfolio PnL')
    ax.text(0.8, .55, '$\mu$='+str(real_world_model.data['Rate']), transform=ax.transAxes, fontsize=10)
    ax.text(0.8, .5, '$\sigma_{RW}$='+str(real_world_model.data['Sigma']), transform=ax.transAxes, fontsize=10)
    if write_file:
        plt.savefig(filename+'DiscreteImpact.pdf')
    plt.show()
    plt.close()

# Plot of Trading Cost distribution
    df_cost = pd.DataFrame(data=trading_cost[0,:,-1],columns=['TradingCost'])
    tc_mean = df_cost.mean()[0]
    tc_std = df_cost.std()[0]
    ax = df_cost.plot.hist(legend=False, title='Transaction Costs', bins=50, density=True )   
    ax.text(0.05, .9, 'mean='+'%.3f' % tc_mean, transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .85, 'std='+'%.3f' % tc_std, transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .80, 'Hedges='+str(num_steps), transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .75, 'Paths='+str(num_paths), transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .7, 'Option='+'%.3f' % C_0,transform=ax.transAxes, fontsize=10)
    ax.text(0.8, .9, 'Price='+str(real_world_model.data['Spot']), transform=ax.transAxes, fontsize=10)
    display_option_params(ax, pricer, [0.8,.85], x_label='Portfolio PnL')
    ax.text(0.8, .55, '$\mu$='+str(real_world_model.data['Rate']), transform=ax.transAxes, fontsize=10)
    ax.text(0.8, .5, '$\sigma_{RW}$='+str(real_world_model.data['Sigma']), transform=ax.transAxes, fontsize=10)
    if write_file:
        plt.savefig(filename+'TransactionCost.pdf')
    plt.show()
    plt.close()



# Plot of PnL distribution with trading costs
    df_pv_cost = pd.DataFrame(data=trading_cost[0,:,-1]+pv[0,:,-1],columns=['Total'])
    pv_tc_std = df_pv_cost.std()[0]
    ax = df_pv_cost.plot.hist(legend=False, title='Discrete Hedging and Cost Impact', bins=50, density=True )
    ax.text(0.05, .9, 'mean='+'%.3f' % (tc_mean+pv_mean), transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .85, 'std='+'%.3f' % pv_tc_std, transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .80, 'Hedges='+str(num_steps), transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .75, 'Paths='+str(num_paths), transform=ax.transAxes, fontsize=10)
    ax.text(0.05, .7, 'Option='+'%.3f' % C_0,transform=ax.transAxes, fontsize=10)
    ax.text(0.8, .9, 'Price='+str(real_world_model.data['Spot']), transform=ax.transAxes, fontsize=10)
    display_option_params(ax, pricer, [0.8,.85], x_label='Portfolio PnL')
    ax.text(0.8, .55, '$\mu$='+str(real_world_model.data['Rate']), transform=ax.transAxes, fontsize=10)
    ax.text(0.8, .5, '$\sigma_{RW}$='+str(real_world_model.data['Sigma']), transform=ax.transAxes, fontsize=10)
    if write_file:
        plt.savefig(filename+'DiscreteAndCost.pdf')
    plt.show()
    plt.close()


 
# Estimate the std from the Derman and Kamal approximation
    transaction_sigma = real_world_model.data['Sigma']*np.sqrt(1+cost / real_world_model.data['Sigma'] *np.sqrt(2/np.pi/delta_t ))
    transaction_price = BlackScholes(S_0, pricer.m.data['Rate'], pricer.m.data['Rate'], transaction_sigma, 
                                     t_0, pricer.d.data['Expiry'], pricer.d.data['Strike'])
    std_approx = np.sqrt(np.pi/4/num_steps)*vega_0*pricer.m.data['Sigma']
    std_approx2 = np.sqrt(np.pi/4/num_steps)*C_0

    print('############### Simulation values with zero cost')
    print('mean='+str(pv_mean))
    print('std='+str(pv_std))
    print('approx std='+str(std_approx))
    print('approx2 std='+str(std_approx2))
    print('')
    print('############### Simulation values combined cost and PV')
    print('total pv='+str(pv_mean+tc_mean))
    print('total std='+str(pv_tc_std))
    print('')
    print('############### Simulation values with cost')
    print('mean cost='+str(tc_mean))
    print('std cost='+str(tc_std))
    print('Std error='+str(tc_std/np.sqrt(num_paths)))
    print('CallPrice='+str(C_0))
    print('Cost as % Prem='+str(tc_mean/C_0))
    print('')
    print('############### Formula values with cost')
    print('sigma_tc='+str(transaction_sigma))
    print('tc_price='+str(transaction_price))
    print('tc_cost='+str(transaction_price-C_0))
    print('')
    print('############### Misc')
    print('initial cost='+str(cost_0))
    print('vega='+str(vega_0))

#    print('tc_vega_cost='+str( (transaction_sigma-pricer.m.data['Sigma']) * vega_0))
    
    
#    print(df_list)
    


    return pv_std, std_approx

###############################
# Function for calculating delta hedging costs as a function of sigma
def delta_hedging_sigma_dependency(pricer, real_world_model, start_time, end_time, cost, num_steps, num_paths=1, num_samples=1, write_file=False, filename='DeltaHedging.pdf' ):
    model_0 = copy.deepcopy(pricer.m)            
    std_calc = []
    approx = []
    for s in np.linspace(0.01,0.5,10):
        model = copy.deepcopy(model_0)
        p = BlackScholesPricer(model,pricer.d)
        s1, s2 = delta_hedging( pricer=p, real_world_model=real_world_model, start_time=today, end_time=Expiry, cost=cost, num_steps=num_steps, num_paths=num_paths, num_samples=num_samples, write_file=True, filename=filename )
        std_calc.append(s1)
        approx.append(s2)
    
    df = pd.DataFrame(index=np.linspace(0.01,0.5,10), data=std_calc, columns=['Numeric'] )
    df['Approx'] = approx
    ax = df.plot(title='PnL Std Dev as a function of Sigma')                                                       
    ax.set_xlabel('Sigma')
    ax.set_ylabel('PnL Std Dev')


###############################
# Class for holding market data for a Black Scholes model        
class BlackScholesModel:
    def __init__(self, S=100, r=0.02, q=0.01, sigma=.2, t=0):
        self.data = {}
        self.data['Spot'] = S
        self.data['Rate'] = r
        self.data['Div'] = q
        self.data['Sigma'] = sigma
        self.data['Today'] = t
        self.update_derived_vals()

    def update_derived_vals(self):
        self.drift = self.data['Rate']-self.data['Div']
        self.log_spot = np.log(self.data['Spot']) 
        self.log_drift = self.drift - 0.5 * self.data['Sigma']**2        

       
###############################
# Base class for defining a derivative        
###############################
class Derivative:
    def __init__(self, T=1, K=100, Type="Call"):
        self.data = {}
        self.data['Expiry'] = T
        self.data["Strike"] = K
        self.data["Type"] = Type


###############################
# Returns the undiscounted payoff based on an asset value S        
    def payoff(self,S):
        return S

###############################
# Returns the undiscounted payoff based on an asset value S        
    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return S

###############################
# Derived class that implements a vanilla option        
###############################
class Vanilla(Derivative):       
    def payoff(self,S):       
        P = (S - self.data['Strike'])
        P[P<0] = 0
        return P        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return BlackScholes(S, r, b, sigma, t, T, K, Type, greek)
    

###############################
# Derived class that implements a digital option        
###############################
class Digital(Derivative):
    def __init__(self, T=1, K=100, Type="Call"):
        self.data = {}
        self.data["Expiry"] = T
        self.data["Strike"] = K
        self.data["Type"] = Type

    def payoff(self,S):       
        P = np.heaviside(S - self.data['Strike'],0)
        return P        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return black_scholes_digital(S, r, b, sigma, t, T, K, Type, greek)


###############################
# Derived class that implements a digital option        
###############################
class Onetouch(Derivative):
    def __init__(self, T=1, K=100, Type="Call"):
        self.data = {}
        self.data["Expiry"] = T
        self.data["Strike"] = K
        self.data["Type"] = Type

    def payoff(self,S):       
        P = np.heaviside(S - self.data['Strike'],0)
        return P        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return black_scholes_onetouch(S, r, b, sigma, t, T, K, Type, greek)


###############################
# Derived class that implements a perpetual American binary
###############################
class PerpetualAmerican(Derivative):
    def __init__(self, H=90):
        self.data = {}
        self.data['Barrier'] = H
        self.data['Expiry'] = 1.0
        self.data['Strike'] = H
        self.data['Type'] = 'Call'
        
    def payoff(self,S):       
        return 1        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return perpetual_american_bsm(S, K, r, sigma, greek)


###############################
# Derived class that implements a barrier option        
###############################
class BarrierOption(Derivative):
    def __init__(self, T=1, K=100, H=90, Type='DICall'):
        self.data = {}
        self.data['Barrier'] = H
        self.data['Expiry'] = T
        self.data['Strike'] = K
        self.data['Type'] = Type
        
    def payoff(self,S):       
        return 1        

    def black_scholes(self, S, r, b, sigma, t, T, K, Type, greek):
        return barrier_option_bsm(S, r, b, sigma, t, T, K, self.data['Barrier'], Type, greek)



###############################
# Class for combining a Black-Scholes model and a derivative        
###############################
class BlackScholesPricer():
    sens_dict = { 'Delta': ['Spot'], 'Vega': ['Sigma'], 'Rho': ['Rate'], 'Theta': ['Today']}
    
    def __init__(self, black_scholes_model, derivative):
        self.m = black_scholes_model
        self.d = derivative
        self.update_derived_vals()
        self.greek = "Price"

    def print_greeks(self):
        greeks = {}
        greeks['Fwd'] = self.price(greek='Fwd')
        greeks['Price'] = self.price()
        greeks['Delta'] = self.price(greek='Delta')
        greeks['Gamma'] = self.price(greek='Gamma')
        greeks['Theta'] = self.price(greek='Theta')
        greeks['Vega'] = self.price(greek='Vega')
        greeks['Rho'] = self.price(greek='Rho')
        for g in greeks:
            print(g+' = %.15f' % greeks[g])
        return greeks


    def update_derived_vals(self):
        self.m.update_derived_vals()
        self.tau = self.d.data['Expiry'] - self.m.data['Today']
        self.sigma_root_tau = self.m.data['Sigma'] * np.sqrt(self.tau)
        self.log_drift_tau = self.m.log_drift * self.tau
        self.discount = np.exp(-self.m.data['Rate'] * self.tau)

    def implied_vol(self,fixed_price):
        self.fixed_price = fixed_price
        imp_vol = optimize.root_scalar( self.implied_vol_pricer, bracket=[0.0001,10])
        return imp_vol.root
    
    def implied_vol_pricer(self,sigma):
        return self.price(sigma=sigma) - self.fixed_price        

    def sensitivity(self, S, P, bump):
        return ( self.discount * self.d.payoff(S) - P ) / bump

    def bump(self, key, bump, is_mult_bump=True):
        for x in self.sens_dict[key]:
            if is_mult_bump:
                self.m.data[x] *= bump
            else:
                self.m.data[x] += bump
        self.update_derived_vals()
        
    def get_bump(self, key, bump, is_mult_bump):
        if is_mult_bump:
            x = self.sens_dict[key][0]
            return self.m.data[x] * (bump-1)  
        else:
            return bump                
        
    def price_key(self, key, value):
        self.m.data[key] = value
        return self.price()
    
    def price(self, S=None, r=None, b=None, sigma=None, t=None, T=None, K=None, Type=None, greek=None):
        if S==None:
            S = self.m.data['Spot']
        if r==None:
            r = self.m.data['Rate']
        if b==None:
            b = self.m.drift
        if sigma==None:
            sigma = self.m.data['Sigma']
        if t==None:
            t = self.m.data['Today']
        if T==None:
            T = self.d.data['Expiry']
        if K==None:
            K = self.d.data['Strike']
        if Type==None:
            Type = self.d.data['Type']
        if greek==None:
            greek = self.greek
        return self.d.black_scholes(S, r, b, sigma, t, T, K, Type, greek)
    
    def plot(self, key, xmin, xmax, xpoints, greek="Price", title=None, xlabel=None, ylabel=None, display_vals=False, display_loc=[0.05,0.95], filename=None):
        if greek=='Spread':
            self.greek='Price'
        else:
            self.greek = greek
        if title==None:
            title=greek
        if xlabel==None:
            xlabel=key
        if ylabel==None:
            ylabel="Derivative Value"
        x = np.linspace(xmin, xmax, xpoints)
        y = [ self.price_key(key,val) for val in x ]
        if greek=="Spread":
            strike = self.d.data['Strike']
            self.d.data['Strike'] = strike * 0.95   
            y2 = [ self.price_key(key,val) for val in x ]
            self.d.data['Strike'] = strike
            for i in range(xpoints):
                y[i] = ( y2[i] - y[i] ) / 0.05 / strike            
        ax = pd.DataFrame(index=x, data=y).plot(legend=False, title=title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if display_vals:
            yval = np.linspace(display_loc[1],display_loc[1]-0.25,5)
            ax.text(0.05, yval[0], 'K='+str(self.d.data['Strike']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[1], 'sigma='+str(self.m.data['Sigma']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[2], 'T='+str(self.d.data['Expiry']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[3], 'r='+str(self.m.data['Rate']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[4], 'q='+str(self.m.data['Div']), transform=ax.transAxes, fontsize=10)
        if filename!=None:
            plt.savefig(filename+'.pdf')

    def plot2(self, key, xmin, xmax, xpoints, key2, xmin2, xmax2, xpoints2, greek="Price", title=None, xlabel=None, ylabel=None, display_vals=False, display_loc=[0.05,0.95], filename=None):
        if greek=='Spread':
            self.greek='Price'
        else:
            self.greek = greek
        if title==None:
            title=greek
        if xlabel==None:
            xlabel=key
        if ylabel==None:
            ylabel="Derivative Value"
        x = np.linspace(xmin, xmax, xpoints)
        ax = plt.subplot()
        for x2 in np.linspace(xmin2, xmax2, xpoints2):
            self.d.data[key2] = x2
            y = [ self.price_key(key,val) for val in x ]
            pd.DataFrame(index=x, data=y).plot(ax=ax, legend=False, title=title, label=key2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if display_vals:
            yval = np.linspace(display_loc[1],display_loc[1]-0.25,5)
            ax.text(0.05, yval[0], 'K='+str(self.d.data['Strike']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[1], 'sigma='+str(self.m.data['Sigma']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[2], 'T='+str(self.d.data['Expiry']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[3], 'r='+str(self.m.data['Rate']), transform=ax.transAxes, fontsize=10)
            ax.text(0.05, yval[4], 'q='+str(self.m.data['Div']), transform=ax.transAxes, fontsize=10)
        plt.show()
        if filename!=None:
            plt.savefig(filename+'.pdf')

        
# Function to update the pricer based on a change in logS and t
    def update(self, logS, t):
        self.m.data['Spot'] = np.exp(logS)
        self.m.data['Today'] = t
        self.update_derived_vals()
        
        