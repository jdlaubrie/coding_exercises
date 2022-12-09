# Mathematics libraries
using LinearAlgebra
# Residual function
function residual_func(time,Delta_t,Y,density,den_)
  # Elastin degradation [days]
  L_dam = 0.010
  t_dam = 40.0
  D_max = 0.5
  T_ela = 101.0*365.25
  density0 = 241.5
  # Residual function
  exponential = exp(-0.5*(Y/L_dam)^2 - time/t_dam)
  residual = -density + den_ + Delta_t*(-density/T_ela-(D_max/t_dam)*exponential*density0)
  return residual
end
# Ridders method for numeric derivatives
function df_ridders(time,Delta_t,Y,density,den_)
  """
  Returns the derivative of a function func at a point x by Ridders method of polynomial
  extrapolation. The value h is input as an estimated initial stepsize; it need not be small,
  but rather should be an increment in x over which func changes substantially. An estimate
  of the error in the derivative is returned as err .
  Parameters: Stepsize is decreased by CON at each iteration. Max size of tableau is set by
  NTAB. Return when error is SAFE worse than the best so far.
  """
  h=1.0e-3
  BIG=1.0e30
  NTAB=10
  CON=1.4
  CON2=CON*CON
  SAFE=2.0
  if h==0.0
    println("h must be nonzero in df_ridders")
    return
  end
  dfridr=0.0
  a = zeros(Float64,NTAB,NTAB)
  hh=h
  x_pos = density + hh
  x_neg = density - hh
  func_pos = residual_func(time,Delta_t,Y,x_pos,den_)
  func_neg = residual_func(time,Delta_t,Y,x_neg,den_)
  a[1,1] = (func_pos-func_neg)/(2.0*hh)
  err=BIG
  # Successive columns in the Neville tableau will go to smaller stepsizes and higher orders of extrapolation.
  for i in 2:NTAB
    hh=hh/CON
    # Try new, smaller stepsize.
    x_pos = density + hh
    x_neg = density - hh
    func_pos = residual_func(time,Delta_t,Y,x_pos,den_)
    func_neg = residual_func(time,Delta_t,Y,x_neg,den_)
    a[1,i] = (func_pos-func_neg)/(2.0*hh)
    fac=CON2
    # Compute extrapolations of various orders, requiring no new function evaluations.
    for j in 2:i
      a[j,i] = (a[j-1,i]*fac-a[j-1,i-1])/(fac-1.0)
      fac=CON2*fac
      errt=max(abs(a[j,i]-a[j-1,i]),abs(a[j,i]-a[j-1,i-1]))
      #The error strategy is to compare each new extrapolation to one order lower, both at
      #the present stepsize and the previous one.
      #If error is decreased, save the improved answer.
      if errt<=err
        err=errt
        dfridr=a[j,i]
      end
      #If higher order is worse by a significant factor SAFE, then quit early.
      if abs(a[i,i]-a[i-1,i-1])>=SAFE*err
        #print('Early quit in df_ridders function')
        return dfridr,err
      end
    end
  end
  return dfridr,err
end
# Evaluation of the function
function func_eval(time,Delta_t,Y,density,den_)
  """
  Function with the aim to get the values of the function and derivative at x.
  """
  h=1.0e-3
  func = residual_func(time,Delta_t,Y,density,den_)
  df,err = df_ridders(time,Delta_t,Y,density,den_)
  return func,df,err
end
# Newton-Raphson method
function newton_raphson(time,Delta_t,Y,density)
  """
  Using the Newton-Raphson method, find the root of a function known to lie close to x. 
  The root rtnewt will be refined until its accuracy is known within +/- xacc. funcd
  is a user-supplied subroutine that returns both the function value and the first derivative
  of the function at the point x.
  """
  # Set to maximum number of iterations.
  JMAX=20
  res = residual_func(time,Delta_t,Y,density,density)
  # Initial guess.
  rtnewt=density
  for j in 1:JMAX
    func,df,err = func_eval(time,Delta_t,Y,rtnewt,density)
    dx = func/df
    rtnewt -= dx
    # Convergence.
    if abs(func/res)<1.e-5
      #print('Iterations: '+str(j))
      return rtnewt
    end
  end
  println("newton_raphson exceeded maximum iterations")
  return rtnewt
end
# Test of the Solution method with derivatives from Ridders method
T_e = 101*365.25
Dm = 0.5
td = 40.0
Y=0.001
Ld=0.010
den_=241.5
dens=241.5
Delta_t=1.0
time=0.0
while time<3000.0
    global time += Delta_t
    global den_,dens
    backeuler = newton_raphson(time,Delta_t,Y,den_)
    func,df,err = func_eval(time,Delta_t,Y,backeuler,den_)
    density=241.5*exp(-time/T_e)+241.5*(Dm/td)*(T_e*td/(td-T_e))*exp(-0.5*(Y/Ld)^2)*(exp(-time/T_e)-exp(-time/td))
    exponential = exp(-0.5*(Y/0.010)^2 - time/40.0)
    forweuler = dens + Delta_t*(-dens/(101.0*365.25)-(0.5/40.0)*exponential*241.5)
    dens = forweuler
    den_=backeuler
    println("$time, $forweuler, $backeuler, $density")
end

