module MATH361Lectures

using LinearAlgebra

export forwardsub, backsub, rowopmat, lufact, luppfact, chfact

"""
    forwardsub(L,b)

Implements the forward substition algorithm to solve the linear system \$Ly=b\$, 
where \$L\$ is an \$n \\times n\$ lower triangular matrix and \$b\$ is a vector of length \$n\$.

# Example
```julia-repl
julia> L = [1. 0. 0.;2. -1. 0.; 1. -1. 1.];
julia> b = ones(3);
julia> y = forwardsub(L,b)
```

"""
function forwardsub(L,b)
    n = size(L)[1]; # number of rows
    y = zeros(n); # initialize solution vector
    y[1] = b[1]/L[1,1];
    for i=2:n
        y[i] = (b[i] - dot(L[i, 1:i-1],y[1:i-1])) / L[i,i];
    end
    return y
end

"""
    backsub(L,b)

Implements the back substition algorithm to solve the linear system \$Ux=y\$, 
where \$U\$ is an \$n \\times n\$ upper triangular matrix and \$y\$ is a vector of length \$n\$.

# Example
```julia-repl
julia> U = [-1.0 2.0 1.0;0. 3.0 -2.0;0.0 0.0 -1.0];
julia> y = ones(3);
julia> x = backsub(U,y)
```

"""
function backsub(U,y)
    n = size(U)[1]; # number of rows
    x = zeros(n); # initialize solution vector
    x[n] = y[n]/U[n,n];
    for i=n-1:-1:1
        x[i] = (y[i] - dot(U[i, i+1:n],x[i+1:n])) / U[i,i];
    end
    return x
end

"""
    rowmultmat(i,α,n)

Constructs an \$n \\times n\$ matrix that upon left multiplication replaces row 
i of a matrix \$A\$ with α times row i.

# Example
```julia-repl
julia> M = rowmultmat(2,2.0,4)
```

"""
function rowmultmat(i,α,n)
    In = Matrix{Float64}(I,n,n); # construct n by n identity matrix
    M = Matrix{Float64}(In);
    M[i,i] = α;
    return M
end

"""
    rowswapmat(i,j,n)

Constructs an \$n \\times n\$ matrix that upon left multiplication replaces row 
i of a matrix \$A\$ with row j.

# Example
```julia-repl
julia> M = rowswapmat(1,2,4)
```

"""
function rowswapmat(i,j,n)
    In = Matrix{Float64}(I,n,n); # construct n by n identity matrix
    M = Matrix{Float64}(In);
    M[i,:] = In[j,:];
    M[j,:] = In[i,:];
    return M
end

"""
    rowopmat(j,i,α,n)

Constructs an \$n \\times n\$ matrix that upon left multiplication replaces row 
j of a matrix \$A\$ with α times row i plus row j.

# Example
```julia-repl
julia> L12 = rowopmat(2,1,2.0,4)
```

"""
function rowopmat(j,i,α,n)
    In = Matrix{Float64}(I,n,n); # construct n by n identity matrix
    M = In + α*In[:,j]*In[:,i]'
    return M
end

"""
    lufact(A)

Constructs the LU factorization of a matrix \$A\$.

# Example
```julia-repl
julia> A = [-1.0 2.0 1.0;3.0 -2.0 2.0;-1.0 0.0 1.0]
julia> L,U = lufact(A)
```
 
"""
function lufact(A)
   n = size(A)[1];
   A = Matrix{Float64}(A);
   L = Matrix{Float64}(I,n,n); # initialize L
   for j=1:n-1
        for i=j+1:n
            L[i,j] = A[i,j] / A[j,j];
            A[i,j:n] = A[i,j:n] - L[i,j]*A[j,j:n];
        end
    end
    U = triu(A);
    return L, U
end

function luppfact(A)
    m,n = size(A);             # number of rows and columns
    P = Matrix{Float64}(I,n,n); # initialize P
    U = Matrix{Float64}(A);     # initialize U
    L = Matrix{Float64}(I,n,n); # initialize L
    for k=1:m-1
         ind = k;
         pivot=maximum(abs.(U[k:m,k]));
         for j=k:m
             if(abs(U[j,k])==pivot)
                 ind=j;
                 break
             end
         end
         U[[k,ind],k:m]=U[[ind,k],k:m];
         L[[k,ind],1:k-1]=L[[ind,k],1:k-1]
         P[[k,ind],:]=P[[ind,k],:]
         for j=k+1:m
             L[j,k]=U[j,k]/U[k,k];
             U[j,k:m]=U[j,k:m] - L[j,k].*U[k,k:m];
         end
     end
     return L, U, P
 end

 """
    chfact(A)

Constructs the Cholesky factorization of a SPD matrix \$A\$.

# Example
```julia-repl
julia> A = [5.0 7.0 1.0;7.0 14.0 -1.0;1.0 -1.0 2.0]
julia> L = chfact(A)
```
 
"""
function chfact(A)
   n = size(A)[1];
   A = Matrix{Float64}(A);
   L = Matrix{Float64}(I,n,n); # initialize L
   for k=1:n
     L[k,k] = sqrt(A[k,k] - dot(L[k,1:(k-1)],L[k,1:(k-1)]));
     for i=(k+1):n
       L[i,k] = (A[i,k] - dot(L[i,1:(k-1)],L[k,1:(k-1)]))/L[k,k];
     end
   end


   return L

end


end