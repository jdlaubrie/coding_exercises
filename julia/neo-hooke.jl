# Mathematics libraries
using LinearAlgebra

# Calculate stresses in a thin-walled tube
function ThinWalled(pressure,stretch)
   """
   The stress in radial direction has to be zero to match with the condition
   of the axilsymmetryc model.
   stress_r = 0.0
   """
   stretch_t = stretch
   stretch_r = 1.0/stretch
   #geometry
   r=0.010*stretch_t
   h=0.00141*stretch_r
   #stresses in tube
   stress_t = pressure*r/h
   #stress_r = -pressure/2.0 
   #stress_z = pressure*r/(2.0*h)
   return stress_t #,stress_r
end

# Calculate stresses with anisotropic model
function NeoHookean(stretch,pressure)
   """
   The lagrangian multiplier has to be same than radial stress in thin-walled tube
   to match with results from axilsymmetric model.
   p = 0.0
   """
   # Variable to stretch
   stretch_t = stretch
   stretch_r = 1.0/stretch
   # parameter of hyperelastic model
   density = 1050.0
   mu = 144.0*density
   # stresses
   stress_t = mu*(stretch_t^2-stretch_r^2) #- pressure #/2.0
   #stress_r = mu*(stretch_r^2-I1/3.0)*J^(-5.0/3.0) + 2.0*kappa*(J-1.0)
   #stress_r = -pressure/2.0
   #stress_z = mu*(1.0-I1/3.0)*J^(-5.0/3.0) + 2.0*kappa*(J-1.0)
   return stress_t #,stress_r
end

# Compute circumferential residual function
function ResidualFunction(pressure,stretch)
   #call function to compute the stresses
   ThinStress_t = ThinWalled(pressure,stretch)
   HookStress_t = NeoHookean(stretch,pressure)
   #compute the residual
   Residual = HookStress_t - ThinStress_t
   #else
   #   Residual = HookStress_r - ThinStress_r
   #end
   return Residual
end

# Ridders method for numeric derivatives
function DRidders(pressure,stretch)
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
    println("h must be nonzero in DRidders")
    return
  end
  dfridr=0.0
  a = zeros(Float64,NTAB,NTAB)
  hh=h
  stretch_pos = stretch + hh
  stretch_neg = stretch - hh
  func_pos = ResidualFunction(pressure,stretch_pos)
  func_neg = ResidualFunction(pressure,stretch_neg)
  a[1,1] = (func_pos-func_neg)/(2.0*hh)
  err=BIG
  # Successive columns in the Neville tableau will go to smaller stepsizes and higher orders of extrapolation.
  for i in 2:NTAB
    hh=hh/CON
    # Try new, smaller stepsize.
    stretch_pos = stretch + hh
    stretch_neg = stretch - hh
    func_pos = ResidualFunction(pressure,stretch_pos)
    func_neg = ResidualFunction(pressure,stretch_neg)
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
        return dfridr
      end
    end
  end
  return dfridr
end

# Evaluation of the function
function FunctionEval(pressure,stretch)
  """
  Function with the aim to get the values of the function and derivative at x.
  """
  func = ResidualFunction(pressure,stretch)
  df = DRidders(pressure,stretch)
  return func,df
end

# Newton-Raphson method
function NewtonRaphson(pressure,stretch)
  """
  Using the Newton-Raphson method, find the root of a function known to lie close to x. 
  The root rtnewt will be refined until its accuracy is known within +/- xacc. funcd
  is a user-supplied subroutine that returns both the function value and the first derivative
  of the function at the point x.
  """
  # Set to maximum number of iterations.
  JMAX=20
  res,_ = FunctionEval(pressure,stretch)
  # Initial guess.
  rtnewt=stretch
  for j in 1:JMAX
    func,df = FunctionEval(pressure,rtnewt)
    dx = func/df
    rtnewt -= dx
    # Convergence.
    if abs(func/res)<1.e-5
      return rtnewt
    end
  end
  println("newton_raphson exceeded maximum iterations")
  return rtnewt
end

# Test of the Solution method with derivatives from Ridders method
number_of_increments=3
# initial guess
stretch = 1.0
# incremental solution
for Inc in 1:number_of_increments
   global stretch
   #global residual_norm
   factor=Inc/number_of_increments
   pressure=13.332e3*factor
   solution = NewtonRaphson(pressure,stretch)
   stretch = solution
   radius = 10.0*stretch
   println("Increment=$Inc, Pressure=$pressure, Stretch=$stretch, Radius=$radius")
end

