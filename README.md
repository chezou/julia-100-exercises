100 Julia exercises
=============

This is Julia version of [100 numpy exercises](http://www.loria.fr/~rougier/teaching/numpy.100/)

Latest version of 100 numpy excercises are available at [this repository](https://github.com/rougier/numpy-100).

You can see executed results [here](http://nbviewer.ipython.org/github/chezou/julia-100-exercises/blob/master/julia-100-exercises.ipynb)


# Neophyte
## 1. Import the numpy package under the name `np`

```jl
# nothing to do
```

## 2. Print the Julia version

```jl
VERSION
```

## 3. Create a null vector of size 10

```jl
Z = zeros(10)
```

## 4. Create a null vector of size 10 and set the fifth value to 1

```jl
Z = zeros(10)
Z[5] = 1
Z
```

## 5. Create a vector with values ranging from 10 to 99

```jl
Z = [10:99]
```

## 6. Create a 3x3 matrix with values ranging from 0 to 8

```jl
Z = reshape(0:8, 3, 3)
```

## 7. Find indices of non-zero elements from [1,2,0,0,4,0]

```jl
nz = find([1,2,0,0,4,0])
```

## 8. Create a 3x3 identity matrix

```jl
Z = eye(3)
```

## 9. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal

```jl
Z = diagm(1:4, -1)
```

## 10. Create a 10x10x10 array with random values

```jl
rand(10, 10, 10)
```

# Novice
## 1. Create a 8x8 matrix and fill it with a checkerboard pattern

```jl
Z = zeros(Int64,8,8)
Z[1:2:end, 2:2:end] = 1
Z[2:2:end, 1:2:end] = 1
Z

# Another solution
# Author: harven
[(i+j)%2 for i=1:8, j=1:8]
```

## 2. Create a 10x10 array with random values and find the minimum and maximum values

```jl
Z = rand(10, 10)
Zmin, Zmax = minimum(Z), maximum(Z)

# It can also be written as follows.
# Author: hc_e
# http://qiita.com/chezou/items/d7ca4e95d25835a5cd01#comment-1c20073a44695c08f523
Zmin, Zmax = extrema(Z)
```

## 3. Create a checkerboard 8x8 matrix using the tile function

```jl
# numpy's tile equal to repmat
Z = repmat([0 1;1 0],4,4)
```

## 4. Normalize a 5x5 random matrix (between 0 and 1)

```jl
Z = rand(5, 5)
Zmin, Zmax = minimum(Z), maximum(Z)
Z = (Z .- Zmin)./(Zmax - Zmin)
```

## 5. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)

```jl
Z = ones(5,3) * ones(3,2)
```

## 6. Create a 10x10 matrix with row values ranging from 0 to 9

```jl
(zeros(Int64,10,10) .+ [0:9])'

# Alternate solution
# Author: Leah Hanson
[y for x in 1:10, y in 0:9]
```

## 7. Create a vector of size 1000 with values ranging from 0 to 1, both excluded

```jl
linspace(0,1, 1002)[2:end - 1]
```

## 8. Create a random vector of size 100 and sort it

```jl
Z = rand(100)
sort(Z) # returns a sorted copy of Z; leaves Z unchanged

# Alternate solution
# Author: Leah Hanson
Z = rand(100)
sort!(Z) # sorts Z in-place; returns Z
```

## 9. Consider two random matrices A anb B, check if they are equal.

```jl
A = rand(0:2, 2,2)
B = rand(0:2, 2,2)
A == B
```

## 10. Create a random vector of size 1000 and find the mean value

```jl
Z = rand(1000)
m = mean(Z)
```

# Apprentice
## 1. Make an array immutable (read-only)

```jl
julia> Pkg.add("ImmutableArrays")
julia> using ImmutableArrays
julia> Matrix4x4(rand(4,4))
4x4 ImmutableArrays.Matrix4x4{Float64}:
 0.0724154  0.0840244  0.291123  0.853076
 0.0994344  0.686174   0.214841  0.248117
 0.996963   0.680124   0.405399  0.180246
 0.232086   0.0424678  0.375087  0.799278

```

## 2. Consider a random 10x2 matrix representing Cartesian coordinates, convert them to polar coordinates

```jl
Z = rand(10,2)
X, Y = Z[:,1], Z[:,2]
R = sqrt(X.^2 + Y.^2)
T = atan2(Y,X)
```

## 3. Create random vector of size 100 and replace the maximum value by 0

```jl
Z = rand(100)
Z[indmax(Z)] = 0
```

## 4. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area.

```jl
# There is no official `meshgrid` function.
# See also: https://github.com/JuliaLang/julia/issues/4093
# assume using https://github.com/JuliaLang/julia/blob/master/examples/ndgrid.jl
include("/Applications/Julia-0.3.0-prerelease-547facf2c1.app/Contents/Resources/julia/share/julia/examples/ndgrid.jl")
X = linspace(0,1,10)
Zx, Zy = meshgrid(X, X)

# Another solution
# Author: Alireza Nejati
[(x,y) for x in linspace(0,1,10), y in linspace(0,1,10)]
```

## 5. Print the minimum and maximum representable value for each Julia scalar type

```jl
for dtype in (Int8, Int16, Int32, Int64)
    println(typemin(dtype))
    println(typemax(dtype))
end

# Another solution
# Author: harven
# typemin, typemax returns -Inf, Inf
print(map!(t -> (typemin(t),typemax(t)), subtypes(Signed)))

for dtype in (Float32, Float64)
    println(typemin(dtype))
    println(typemax(dtype))
    println(eps(dtype))
end
```

## 6. Create a structured array representing a position (x,y) and a color (r,g,b)


```jl
# Julia doesn't have StructArray
# see also: https://github.com/JuliaLang/julia/issues/1263
# use DataFrames
```

## 7. Consider a random vector with shape (100,2) representing coordinates, find point by point distances

```jl
Z = rand(10,2)
X,Y = Z[:,1], Z[:,2]
D = sqrt((X.-X').^2 + (Y .- Y').^2)
```

## 8. Generate a generic 2D Gaussian-like array

```jl
X, Y = meshgrid(linspace(-1,1,100),linspace(-1,1,100))
D = sqrtm(X*X + Y*Y)
sigma, mu = 1.0, 0.0
G = exp(-( (D.-mu)^2 / ( 2.0 * sigma^2 ) ) )

# Another solution
# Author: Billou Beilour
sigma, mu = 1.0, 0.0
G = [ exp(-(x-mu).^2/(2.0*sigma^2) -(y-mu).^2/(2.0*sigma^2) ) for x in linspace(-1,1,100), y in linspace(-1,1,100) ]

# It also written
# Author: Billou Beilour
sigma, mu = 1.0, 0.0
x,y = linspace(-1,1,100), linspace(-1,1,100)
G = zeros(length(x),length(y))

for i in 1:length(x), j in 1:length(y)
    G[i,j] = exp(-(x[i]-mu).^2/(2.0*sigma^2) -(y[j]-mu).^2/(2.0*sigma^2) )
end
```

## 9. Consider the vector [1, 2, 3, 4, 5]. How to build a new vector with 3 consecutive zeros interleaved between each value?

```jl
Z = [1,2,3,4,5]
nz = 3
Z0 = zeros(length(Z) + (length(Z)-1)*(nz))
Z0[1:nz+1:end] = Z
```

## 10. Find the nearest value from a given value in an array

```jl
Z = [3,6,9,12,15]
Z[indmin(abs(Z .- 10))]
```

# Journeyman
## 1. Consider the following file:

```
1,2,3,4,5
6,,,7,8
,,9,10,11
```

## How to read it?

```jl
using DataFrames
readtable("missing.dat")
```

## 2. Consider a generator function that generates 10 integers and use it to build an array


```jl
# I can't translate this question
```

## 3. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?

```jl
using StatsBase
Z = ones(10)
I = rand(0:length(Z), 20)
Z += counts(I, 1:length(Z))
```

## 4. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?

```jl
using StatsBase
X = WeightVec([1,2,3,4,5,6])
I = [1,3,9,3,4,1]
F = counts(I, maximum(I), X)
```

## 5. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors

```jl
w,h = 16,16
I = convert(Array{Uint8}, rand(0:2, (h,w,3)))
F = I[:,:,1] * 256 * 256 + I[:,:,2]*256 + I[:,:,3]
n = length(unique(F))
unique(I)
```

## 6. Considering a four dimensional array, how to get sum over the last two axis at once?

```jl
A = rand(0:10, (3,4,3,4))
x,y = size(A)[1:end-2]
z = prod(size(A)[end-1:end])
calc_sum = sum(reshape(A, (x,y,z)),3)
```

## 7. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices?

```jl
using StatsBase
D = WeightVec(rand(100))
S = rand(0:10,100)
D_sums = counts(S, maximum(S), D)
D_counts = counts(S, maximum(S))
D_means = D_sums ./ D_counts
```

# Craftsman
## 1. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])

```jl
# I don't find any function like stride_tricks.as_stride
function rolling(A, window)
       Z = zeros(length(A)-2, window)
       for i in 1:(length(A) - window +1)
           Z[i,:] = A[i:i+2]
       end
       return Z
end

rolling(0:100, 3)
```

## 2. Consider a set of 100 triplets describing 100 triangles (with shared vertices), find the set of unique line segments composing all the triangles.

```jl
faces = rand(0:100, 100, 3)
face2 = kron(faces,[1 1])

F = circshift(sortcols(face2),(0,1))
F = reshape(F, (convert(Int64,length(F)/2),2))
F = sort(F,2)
G = unique(F,1)
```

## 3. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?

```jl
using StatsBase
O = [1 1 2 3 4 4 6]
C = counts(O, maximum(O))
A = foldl(vcat,[kron(ones(Int64, C[i]), i) for i in 1:length(C)])
```

## 4. How to compute averages using a sliding window over an array?

```jl
function moving_average(A, n=3)
  ret = cumsum(A)
  ret[n+1:end] = ret[n+1:end] - ret[1:end-n]
  return ret[n:end-1] / n
end
Z = 0:20
moving_average(Z, 3)
```

# Artisan
## 1. Considering a 100x3 matrix, extract rows with unequal values (e.g. [2,2,3])

```jl
Z = rand(0:5,100,3)
E = prod(Z[:,2:end] .== Z[:,1:end-1],2)
U = Z[find(~E), :]
```

## 2. Convert a vector of ints into a matrix binary representation.

```jl
I = [0 1 2 3 15 16 32 64 128]
B = foldl(hcat,[reverse(int(bool(i & (2 .^ (0:8))))) for i in I])'
```

# Adept
## 1. Consider an arbitrary array, write a function that extracts a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary)

```jl
# Not solve yet
```

# Expert
## 1. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B?

```jl
# I can't execute numpy version...
```

## 2. Extract all the contiguous 3x3 blocks from a random 10x10 matrix.

```jl
# Not solve yet
```

## 3. Create a 2D array subclass such that Z[i,j] == Z[j,i]

```jl
# There is Symmetric class in julia but immutable
# https://github.com/JuliaLang/julia/blob/master/base/linalg/symmetric.jl
# See also: https://github.com/JuliaLang/julia/pull/1533
```

## 4. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of the p matrix products at once? (result has shape (n,1))


```jl
# Author: Alireza Nejati
p, n = 10, 20
M = ones(n,n,p)
V = ones(n,p)
S = reduce(+, [M[i,:,j]*V[i] for i = 1:n, j = 1:p])'
S
```

# Master
## 1. Given a two dimensional array, how to extract unique rows?

```jl
Z = rand(0:2, 6,3)
uZ = unique(Z,1)
```

# Archmaster
