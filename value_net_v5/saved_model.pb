??3
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8·.
?
chess_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_namechess_model/dense/kernel
?
,chess_model/dense/kernel/Read/ReadVariableOpReadVariableOpchess_model/dense/kernel* 
_output_shapes
:
??*
dtype0
?
chess_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_namechess_model/dense/bias
~
*chess_model/dense/bias/Read/ReadVariableOpReadVariableOpchess_model/dense/bias*
_output_shapes	
:?*
dtype0
?
chess_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_namechess_model/dense_1/kernel
?
.chess_model/dense_1/kernel/Read/ReadVariableOpReadVariableOpchess_model/dense_1/kernel* 
_output_shapes
:
??*
dtype0
?
chess_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namechess_model/dense_1/bias
?
,chess_model/dense_1/bias/Read/ReadVariableOpReadVariableOpchess_model/dense_1/bias*
_output_shapes	
:?*
dtype0
?
%chess_model/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%chess_model/batch_normalization/gamma
?
9chess_model/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp%chess_model/batch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
$chess_model/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$chess_model/batch_normalization/beta
?
8chess_model/batch_normalization/beta/Read/ReadVariableOpReadVariableOp$chess_model/batch_normalization/beta*
_output_shapes	
:?*
dtype0
?
+chess_model/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+chess_model/batch_normalization/moving_mean
?
?chess_model/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp+chess_model/batch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
/chess_model/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/chess_model/batch_normalization/moving_variance
?
Cchess_model/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp/chess_model/batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
'chess_model/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'chess_model/batch_normalization_1/gamma
?
;chess_model/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp'chess_model/batch_normalization_1/gamma*
_output_shapes	
:?*
dtype0
?
&chess_model/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&chess_model/batch_normalization_1/beta
?
:chess_model/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp&chess_model/batch_normalization_1/beta*
_output_shapes	
:?*
dtype0
?
-chess_model/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-chess_model/batch_normalization_1/moving_mean
?
Achess_model/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp-chess_model/batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0
?
1chess_model/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31chess_model/batch_normalization_1/moving_variance
?
Echess_model/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp1chess_model/batch_normalization_1/moving_variance*
_output_shapes	
:?*
dtype0
?
'chess_model/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'chess_model/batch_normalization_2/gamma
?
;chess_model/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp'chess_model/batch_normalization_2/gamma*
_output_shapes	
:?*
dtype0
?
&chess_model/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&chess_model/batch_normalization_2/beta
?
:chess_model/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp&chess_model/batch_normalization_2/beta*
_output_shapes	
:?*
dtype0
?
-chess_model/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-chess_model/batch_normalization_2/moving_mean
?
Achess_model/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp-chess_model/batch_normalization_2/moving_mean*
_output_shapes	
:?*
dtype0
?
1chess_model/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31chess_model/batch_normalization_2/moving_variance
?
Echess_model/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp1chess_model/batch_normalization_2/moving_variance*
_output_shapes	
:?*
dtype0
?
'chess_model/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'chess_model/batch_normalization_3/gamma
?
;chess_model/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp'chess_model/batch_normalization_3/gamma*
_output_shapes	
:?*
dtype0
?
&chess_model/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&chess_model/batch_normalization_3/beta
?
:chess_model/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp&chess_model/batch_normalization_3/beta*
_output_shapes	
:?*
dtype0
?
-chess_model/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-chess_model/batch_normalization_3/moving_mean
?
Achess_model/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp-chess_model/batch_normalization_3/moving_mean*
_output_shapes	
:?*
dtype0
?
1chess_model/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31chess_model/batch_normalization_3/moving_variance
?
Echess_model/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp1chess_model/batch_normalization_3/moving_variance*
_output_shapes	
:?*
dtype0
?
'chess_model/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'chess_model/batch_normalization_4/gamma
?
;chess_model/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp'chess_model/batch_normalization_4/gamma*
_output_shapes	
:?*
dtype0
?
&chess_model/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&chess_model/batch_normalization_4/beta
?
:chess_model/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp&chess_model/batch_normalization_4/beta*
_output_shapes	
:?*
dtype0
?
-chess_model/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-chess_model/batch_normalization_4/moving_mean
?
Achess_model/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp-chess_model/batch_normalization_4/moving_mean*
_output_shapes	
:?*
dtype0
?
1chess_model/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31chess_model/batch_normalization_4/moving_variance
?
Echess_model/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp1chess_model/batch_normalization_4/moving_variance*
_output_shapes	
:?*
dtype0
?
'chess_model/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'chess_model/batch_normalization_5/gamma
?
;chess_model/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp'chess_model/batch_normalization_5/gamma*
_output_shapes	
:?*
dtype0
?
&chess_model/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&chess_model/batch_normalization_5/beta
?
:chess_model/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp&chess_model/batch_normalization_5/beta*
_output_shapes	
:?*
dtype0
?
-chess_model/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-chess_model/batch_normalization_5/moving_mean
?
Achess_model/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp-chess_model/batch_normalization_5/moving_mean*
_output_shapes	
:?*
dtype0
?
1chess_model/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31chess_model/batch_normalization_5/moving_variance
?
Echess_model/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp1chess_model/batch_normalization_5/moving_variance*
_output_shapes	
:?*
dtype0
?
'chess_model/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'chess_model/batch_normalization_6/gamma
?
;chess_model/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp'chess_model/batch_normalization_6/gamma*
_output_shapes	
:?*
dtype0
?
&chess_model/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&chess_model/batch_normalization_6/beta
?
:chess_model/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp&chess_model/batch_normalization_6/beta*
_output_shapes	
:?*
dtype0
?
-chess_model/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-chess_model/batch_normalization_6/moving_mean
?
Achess_model/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp-chess_model/batch_normalization_6/moving_mean*
_output_shapes	
:?*
dtype0
?
1chess_model/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31chess_model/batch_normalization_6/moving_variance
?
Echess_model/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp1chess_model/batch_normalization_6/moving_variance*
_output_shapes	
:?*
dtype0
?
'chess_model/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'chess_model/batch_normalization_7/gamma
?
;chess_model/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp'chess_model/batch_normalization_7/gamma*
_output_shapes	
:?*
dtype0
?
&chess_model/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&chess_model/batch_normalization_7/beta
?
:chess_model/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp&chess_model/batch_normalization_7/beta*
_output_shapes	
:?*
dtype0
?
-chess_model/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-chess_model/batch_normalization_7/moving_mean
?
Achess_model/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp-chess_model/batch_normalization_7/moving_mean*
_output_shapes	
:?*
dtype0
?
1chess_model/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31chess_model/batch_normalization_7/moving_variance
?
Echess_model/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp1chess_model/batch_normalization_7/moving_variance*
_output_shapes	
:?*
dtype0
?
'chess_model/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'chess_model/batch_normalization_8/gamma
?
;chess_model/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp'chess_model/batch_normalization_8/gamma*
_output_shapes	
:?*
dtype0
?
&chess_model/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&chess_model/batch_normalization_8/beta
?
:chess_model/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp&chess_model/batch_normalization_8/beta*
_output_shapes	
:?*
dtype0
?
-chess_model/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-chess_model/batch_normalization_8/moving_mean
?
Achess_model/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp-chess_model/batch_normalization_8/moving_mean*
_output_shapes	
:?*
dtype0
?
1chess_model/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31chess_model/batch_normalization_8/moving_variance
?
Echess_model/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp1chess_model/batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d/kernel
?
-chess_model/conv2d/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d/kernel*'
_output_shapes
:?*
dtype0
?
chess_model/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namechess_model/conv2d/bias
?
+chess_model/conv2d/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d/bias*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namechess_model/conv2d_1/kernel
?
/chess_model/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d_1/kernel*(
_output_shapes
:??*
dtype0
?
chess_model/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d_1/bias
?
-chess_model/conv2d_1/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d_1/bias*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namechess_model/conv2d_2/kernel
?
/chess_model/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d_2/kernel*(
_output_shapes
:??*
dtype0
?
chess_model/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d_2/bias
?
-chess_model/conv2d_2/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d_2/bias*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namechess_model/conv2d_3/kernel
?
/chess_model/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d_3/kernel*(
_output_shapes
:??*
dtype0
?
chess_model/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d_3/bias
?
-chess_model/conv2d_3/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d_3/bias*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namechess_model/conv2d_4/kernel
?
/chess_model/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d_4/kernel*(
_output_shapes
:??*
dtype0
?
chess_model/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d_4/bias
?
-chess_model/conv2d_4/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d_4/bias*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namechess_model/conv2d_5/kernel
?
/chess_model/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d_5/kernel*(
_output_shapes
:??*
dtype0
?
chess_model/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d_5/bias
?
-chess_model/conv2d_5/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d_5/bias*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namechess_model/conv2d_6/kernel
?
/chess_model/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d_6/kernel*(
_output_shapes
:??*
dtype0
?
chess_model/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d_6/bias
?
-chess_model/conv2d_6/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d_6/bias*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namechess_model/conv2d_7/kernel
?
/chess_model/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d_7/kernel*(
_output_shapes
:??*
dtype0
?
chess_model/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d_7/bias
?
-chess_model/conv2d_7/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d_7/bias*
_output_shapes	
:?*
dtype0
?
chess_model/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namechess_model/conv2d_8/kernel
?
/chess_model/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpchess_model/conv2d_8/kernel*(
_output_shapes
:??*
dtype0
?
chess_model/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namechess_model/conv2d_8/bias
?
-chess_model/conv2d_8/bias/Read/ReadVariableOpReadVariableOpchess_model/conv2d_8/bias*
_output_shapes	
:?*
dtype0
?
chess_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_namechess_model/dense_2/kernel
?
.chess_model/dense_2/kernel/Read/ReadVariableOpReadVariableOpchess_model/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
chess_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namechess_model/dense_2/bias
?
,chess_model/dense_2/bias/Read/ReadVariableOpReadVariableOpchess_model/dense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/chess_model/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*0
shared_name!Adam/chess_model/dense/kernel/m
?
3Adam/chess_model/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/chess_model/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/chess_model/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameAdam/chess_model/dense/bias/m
?
1Adam/chess_model/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/chess_model/dense/bias/m*
_output_shapes	
:?*
dtype0
?
!Adam/chess_model/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!Adam/chess_model/dense_1/kernel/m
?
5Adam/chess_model/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/chess_model/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/chess_model/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/chess_model/dense_1/bias/m
?
3Adam/chess_model/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/chess_model/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
,Adam/chess_model/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/chess_model/batch_normalization/gamma/m
?
@Adam/chess_model/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp,Adam/chess_model/batch_normalization/gamma/m*
_output_shapes	
:?*
dtype0
?
+Adam/chess_model/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+Adam/chess_model/batch_normalization/beta/m
?
?Adam/chess_model/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp+Adam/chess_model/batch_normalization/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_1/gamma/m
?
BAdam/chess_model/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_1/gamma/m*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_1/beta/m
?
AAdam/chess_model/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_1/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_2/gamma/m
?
BAdam/chess_model/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_2/gamma/m*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_2/beta/m
?
AAdam/chess_model/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_2/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_3/gamma/m
?
BAdam/chess_model/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_3/gamma/m*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_3/beta/m
?
AAdam/chess_model/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_3/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_4/gamma/m
?
BAdam/chess_model/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_4/gamma/m*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_4/beta/m
?
AAdam/chess_model/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_4/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_5/gamma/m
?
BAdam/chess_model/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_5/gamma/m*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_5/beta/m
?
AAdam/chess_model/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_5/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_6/gamma/m
?
BAdam/chess_model/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_6/gamma/m*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_6/beta/m
?
AAdam/chess_model/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_6/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_7/gamma/m
?
BAdam/chess_model/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_7/gamma/m*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_7/beta/m
?
AAdam/chess_model/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_7/beta/m*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_8/gamma/m
?
BAdam/chess_model/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_8/gamma/m*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_8/beta/m
?
AAdam/chess_model/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_8/beta/m*
_output_shapes	
:?*
dtype0
?
 Adam/chess_model/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d/kernel/m
?
4Adam/chess_model/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d/kernel/m*'
_output_shapes
:?*
dtype0
?
Adam/chess_model/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/chess_model/conv2d/bias/m
?
2Adam/chess_model/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/chess_model/conv2d/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_1/kernel/m
?
6Adam/chess_model/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_1/kernel/m*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_1/bias/m
?
4Adam/chess_model/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_1/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_2/kernel/m
?
6Adam/chess_model/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_2/kernel/m*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_2/bias/m
?
4Adam/chess_model/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_2/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_3/kernel/m
?
6Adam/chess_model/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_3/kernel/m*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_3/bias/m
?
4Adam/chess_model/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_3/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_4/kernel/m
?
6Adam/chess_model/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_4/kernel/m*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_4/bias/m
?
4Adam/chess_model/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_4/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_5/kernel/m
?
6Adam/chess_model/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_5/kernel/m*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_5/bias/m
?
4Adam/chess_model/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_5/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_6/kernel/m
?
6Adam/chess_model/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_6/kernel/m*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_6/bias/m
?
4Adam/chess_model/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_6/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_7/kernel/m
?
6Adam/chess_model/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_7/kernel/m*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_7/bias/m
?
4Adam/chess_model/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_7/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_8/kernel/m
?
6Adam/chess_model/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_8/kernel/m*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_8/bias/m
?
4Adam/chess_model/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_8/bias/m*
_output_shapes	
:?*
dtype0
?
!Adam/chess_model/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!Adam/chess_model/dense_2/kernel/m
?
5Adam/chess_model/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/chess_model/dense_2/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/chess_model/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/chess_model/dense_2/bias/m
?
3Adam/chess_model/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/chess_model/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/chess_model/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*0
shared_name!Adam/chess_model/dense/kernel/v
?
3Adam/chess_model/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/chess_model/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/chess_model/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameAdam/chess_model/dense/bias/v
?
1Adam/chess_model/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/chess_model/dense/bias/v*
_output_shapes	
:?*
dtype0
?
!Adam/chess_model/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!Adam/chess_model/dense_1/kernel/v
?
5Adam/chess_model/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/chess_model/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/chess_model/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/chess_model/dense_1/bias/v
?
3Adam/chess_model/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/chess_model/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
,Adam/chess_model/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/chess_model/batch_normalization/gamma/v
?
@Adam/chess_model/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp,Adam/chess_model/batch_normalization/gamma/v*
_output_shapes	
:?*
dtype0
?
+Adam/chess_model/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*<
shared_name-+Adam/chess_model/batch_normalization/beta/v
?
?Adam/chess_model/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp+Adam/chess_model/batch_normalization/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_1/gamma/v
?
BAdam/chess_model/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_1/gamma/v*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_1/beta/v
?
AAdam/chess_model/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_1/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_2/gamma/v
?
BAdam/chess_model/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_2/gamma/v*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_2/beta/v
?
AAdam/chess_model/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_2/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_3/gamma/v
?
BAdam/chess_model/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_3/gamma/v*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_3/beta/v
?
AAdam/chess_model/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_3/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_4/gamma/v
?
BAdam/chess_model/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_4/gamma/v*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_4/beta/v
?
AAdam/chess_model/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_4/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_5/gamma/v
?
BAdam/chess_model/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_5/gamma/v*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_5/beta/v
?
AAdam/chess_model/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_5/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_6/gamma/v
?
BAdam/chess_model/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_6/gamma/v*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_6/beta/v
?
AAdam/chess_model/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_6/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_7/gamma/v
?
BAdam/chess_model/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_7/gamma/v*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_7/beta/v
?
AAdam/chess_model/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_7/beta/v*
_output_shapes	
:?*
dtype0
?
.Adam/chess_model/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/chess_model/batch_normalization_8/gamma/v
?
BAdam/chess_model/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/chess_model/batch_normalization_8/gamma/v*
_output_shapes	
:?*
dtype0
?
-Adam/chess_model/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adam/chess_model/batch_normalization_8/beta/v
?
AAdam/chess_model/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp-Adam/chess_model/batch_normalization_8/beta/v*
_output_shapes	
:?*
dtype0
?
 Adam/chess_model/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d/kernel/v
?
4Adam/chess_model/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d/kernel/v*'
_output_shapes
:?*
dtype0
?
Adam/chess_model/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/chess_model/conv2d/bias/v
?
2Adam/chess_model/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/chess_model/conv2d/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_1/kernel/v
?
6Adam/chess_model/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_1/kernel/v*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_1/bias/v
?
4Adam/chess_model/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_1/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_2/kernel/v
?
6Adam/chess_model/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_2/kernel/v*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_2/bias/v
?
4Adam/chess_model/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_2/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_3/kernel/v
?
6Adam/chess_model/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_3/kernel/v*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_3/bias/v
?
4Adam/chess_model/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_3/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_4/kernel/v
?
6Adam/chess_model/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_4/kernel/v*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_4/bias/v
?
4Adam/chess_model/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_4/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_5/kernel/v
?
6Adam/chess_model/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_5/kernel/v*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_5/bias/v
?
4Adam/chess_model/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_5/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_6/kernel/v
?
6Adam/chess_model/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_6/kernel/v*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_6/bias/v
?
4Adam/chess_model/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_6/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_7/kernel/v
?
6Adam/chess_model/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_7/kernel/v*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_7/bias/v
?
4Adam/chess_model/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_7/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/chess_model/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"Adam/chess_model/conv2d_8/kernel/v
?
6Adam/chess_model/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/chess_model/conv2d_8/kernel/v*(
_output_shapes
:??*
dtype0
?
 Adam/chess_model/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/chess_model/conv2d_8/bias/v
?
4Adam/chess_model/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOp Adam/chess_model/conv2d_8/bias/v*
_output_shapes	
:?*
dtype0
?
!Adam/chess_model/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!Adam/chess_model/dense_2/kernel/v
?
5Adam/chess_model/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/chess_model/dense_2/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/chess_model/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/chess_model/dense_2/bias/v
?
3Adam/chess_model/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/chess_model/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?

dense1

dense2

normalize1

normalize2

normalize3

normalize4

normalize5

normalize6
	
normalize7


normalize8

normalize9
convolution1
convolution2
convolution3
convolution4
convolution5
convolution6
convolution7
convolution8
convolution9
add
relu
	valueHead

signatures
	optimizer
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?

kernel
 bias
#!_self_saveable_object_factories
"trainable_variables
#	variables
$regularization_losses
%	keras_api
?

&kernel
'bias
#(_self_saveable_object_factories
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
#2_self_saveable_object_factories
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?
7axis
	8gamma
9beta
:moving_mean
;moving_variance
#<_self_saveable_object_factories
=trainable_variables
>	variables
?regularization_losses
@	keras_api
?
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
#F_self_saveable_object_factories
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
#P_self_saveable_object_factories
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
#Z_self_saveable_object_factories
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
#d_self_saveable_object_factories
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance
#n_self_saveable_object_factories
otrainable_variables
p	variables
qregularization_losses
r	keras_api
?
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
#x_self_saveable_object_factories
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
?
}axis
	~gamma
beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem? m?&m?'m?.m?/m?8m?9m?Bm?Cm?Lm?Mm?Vm?Wm?`m?am?jm?km?tm?um?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?v? v?&v?'v?.v?/v?8v?9v?Bv?Cv?Lv?Mv?Vv?Wv?`v?av?jv?kv?tv?uv?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
0
 1
&2
'3
.4
/5
86
97
B8
C9
L10
M11
V12
W13
`14
a15
j16
k17
t18
u19
~20
21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?
0
 1
&2
'3
.4
/5
06
17
88
99
:10
;11
B12
C13
D14
E15
L16
M17
N18
O19
V20
W21
X22
Y23
`24
a25
b26
c27
j28
k29
l30
m31
t32
u33
v34
w35
~36
37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
 
?
?non_trainable_variables
?layer_metrics
trainable_variables
?metrics
	variables
 ?layer_regularization_losses
regularization_losses
?layers
VT
VARIABLE_VALUEchess_model/dense/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEchess_model/dense/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
 
?
?non_trainable_variables
?layer_metrics
"trainable_variables
?metrics
#	variables
 ?layer_regularization_losses
$regularization_losses
?layers
XV
VARIABLE_VALUEchess_model/dense_1/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEchess_model/dense_1/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
 
?
?non_trainable_variables
?layer_metrics
)trainable_variables
?metrics
*	variables
 ?layer_regularization_losses
+regularization_losses
?layers
 
fd
VARIABLE_VALUE%chess_model/batch_normalization/gamma+normalize1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE$chess_model/batch_normalization/beta*normalize1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+chess_model/batch_normalization/moving_mean1normalize1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE/chess_model/batch_normalization/moving_variance5normalize1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
02
13
 
?
?non_trainable_variables
?layer_metrics
3trainable_variables
?metrics
4	variables
 ?layer_regularization_losses
5regularization_losses
?layers
 
hf
VARIABLE_VALUE'chess_model/batch_normalization_1/gamma+normalize2/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE&chess_model/batch_normalization_1/beta*normalize2/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-chess_model/batch_normalization_1/moving_mean1normalize2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE1chess_model/batch_normalization_1/moving_variance5normalize2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
:2
;3
 
?
?non_trainable_variables
?layer_metrics
=trainable_variables
?metrics
>	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
hf
VARIABLE_VALUE'chess_model/batch_normalization_2/gamma+normalize3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE&chess_model/batch_normalization_2/beta*normalize3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-chess_model/batch_normalization_2/moving_mean1normalize3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE1chess_model/batch_normalization_2/moving_variance5normalize3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
D2
E3
 
?
?non_trainable_variables
?layer_metrics
Gtrainable_variables
?metrics
H	variables
 ?layer_regularization_losses
Iregularization_losses
?layers
 
hf
VARIABLE_VALUE'chess_model/batch_normalization_3/gamma+normalize4/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE&chess_model/batch_normalization_3/beta*normalize4/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-chess_model/batch_normalization_3/moving_mean1normalize4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE1chess_model/batch_normalization_3/moving_variance5normalize4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
N2
O3
 
?
?non_trainable_variables
?layer_metrics
Qtrainable_variables
?metrics
R	variables
 ?layer_regularization_losses
Sregularization_losses
?layers
 
hf
VARIABLE_VALUE'chess_model/batch_normalization_4/gamma+normalize5/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE&chess_model/batch_normalization_4/beta*normalize5/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-chess_model/batch_normalization_4/moving_mean1normalize5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE1chess_model/batch_normalization_4/moving_variance5normalize5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
X2
Y3
 
?
?non_trainable_variables
?layer_metrics
[trainable_variables
?metrics
\	variables
 ?layer_regularization_losses
]regularization_losses
?layers
 
hf
VARIABLE_VALUE'chess_model/batch_normalization_5/gamma+normalize6/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE&chess_model/batch_normalization_5/beta*normalize6/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-chess_model/batch_normalization_5/moving_mean1normalize6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE1chess_model/batch_normalization_5/moving_variance5normalize6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1

`0
a1
b2
c3
 
?
?non_trainable_variables
?layer_metrics
etrainable_variables
?metrics
f	variables
 ?layer_regularization_losses
gregularization_losses
?layers
 
hf
VARIABLE_VALUE'chess_model/batch_normalization_6/gamma+normalize7/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE&chess_model/batch_normalization_6/beta*normalize7/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-chess_model/batch_normalization_6/moving_mean1normalize7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE1chess_model/batch_normalization_6/moving_variance5normalize7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
l2
m3
 
?
?non_trainable_variables
?layer_metrics
otrainable_variables
?metrics
p	variables
 ?layer_regularization_losses
qregularization_losses
?layers
 
hf
VARIABLE_VALUE'chess_model/batch_normalization_7/gamma+normalize8/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE&chess_model/batch_normalization_7/beta*normalize8/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-chess_model/batch_normalization_7/moving_mean1normalize8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE1chess_model/batch_normalization_7/moving_variance5normalize8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
v2
w3
 
?
?non_trainable_variables
?layer_metrics
ytrainable_variables
?metrics
z	variables
 ?layer_regularization_losses
{regularization_losses
?layers
 
hf
VARIABLE_VALUE'chess_model/batch_normalization_8/gamma+normalize9/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE&chess_model/batch_normalization_8/beta*normalize9/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-chess_model/batch_normalization_8/moving_mean1normalize9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE1chess_model/batch_normalization_8/moving_variance5normalize9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1

~0
1
?2
?3
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
][
VARIABLE_VALUEchess_model/conv2d/kernel.convolution1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEchess_model/conv2d/bias,convolution1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
_]
VARIABLE_VALUEchess_model/conv2d_1/kernel.convolution2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEchess_model/conv2d_1/bias,convolution2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
_]
VARIABLE_VALUEchess_model/conv2d_2/kernel.convolution3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEchess_model/conv2d_2/bias,convolution3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
_]
VARIABLE_VALUEchess_model/conv2d_3/kernel.convolution4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEchess_model/conv2d_3/bias,convolution4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
_]
VARIABLE_VALUEchess_model/conv2d_4/kernel.convolution5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEchess_model/conv2d_4/bias,convolution5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
_]
VARIABLE_VALUEchess_model/conv2d_5/kernel.convolution6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEchess_model/conv2d_5/bias,convolution6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
_]
VARIABLE_VALUEchess_model/conv2d_6/kernel.convolution7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEchess_model/conv2d_6/bias,convolution7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
_]
VARIABLE_VALUEchess_model/conv2d_7/kernel.convolution8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEchess_model/conv2d_7/bias,convolution8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
_]
VARIABLE_VALUEchess_model/conv2d_8/kernel.convolution9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEchess_model/conv2d_8/bias,convolution9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
[Y
VARIABLE_VALUEchess_model/dense_2/kernel+valueHead/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEchess_model/dense_2/bias)valueHead/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
00
11
:2
;3
D4
E5
N6
O7
X8
Y9
b10
c11
l12
m13
v14
w15
?16
?17
 

?0
?1
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
 
 
 
 
 
 
 
 
 
 

00
11
 
 
 
 

:0
;1
 
 
 
 

D0
E1
 
 
 
 

N0
O1
 
 
 
 

X0
Y1
 
 
 
 

b0
c1
 
 
 
 

l0
m1
 
 
 
 

v0
w1
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
yw
VARIABLE_VALUEAdam/chess_model/dense/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/chess_model/dense/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/chess_model/dense_1/kernel/mDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/chess_model/dense_1/bias/mBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/chess_model/batch_normalization/gamma/mGnormalize1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/chess_model/batch_normalization/beta/mFnormalize1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_1/gamma/mGnormalize2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_1/beta/mFnormalize2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_2/gamma/mGnormalize3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_2/beta/mFnormalize3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_3/gamma/mGnormalize4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_3/beta/mFnormalize4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_4/gamma/mGnormalize5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_4/beta/mFnormalize5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_5/gamma/mGnormalize6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_5/beta/mFnormalize6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_6/gamma/mGnormalize7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_6/beta/mFnormalize7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_7/gamma/mGnormalize8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_7/beta/mFnormalize8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_8/gamma/mGnormalize9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_8/beta/mFnormalize9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE Adam/chess_model/conv2d/kernel/mJconvolution1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/chess_model/conv2d/bias/mHconvolution1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_1/kernel/mJconvolution2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_1/bias/mHconvolution2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_2/kernel/mJconvolution3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_2/bias/mHconvolution3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_3/kernel/mJconvolution4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_3/bias/mHconvolution4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_4/kernel/mJconvolution5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_4/bias/mHconvolution5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_5/kernel/mJconvolution6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_5/bias/mHconvolution6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_6/kernel/mJconvolution7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_6/bias/mHconvolution7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_7/kernel/mJconvolution8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_7/bias/mHconvolution8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_8/kernel/mJconvolution9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_8/bias/mHconvolution9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE!Adam/chess_model/dense_2/kernel/mGvalueHead/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/chess_model/dense_2/bias/mEvalueHead/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/chess_model/dense/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/chess_model/dense/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/chess_model/dense_1/kernel/vDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/chess_model/dense_1/bias/vBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/chess_model/batch_normalization/gamma/vGnormalize1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/chess_model/batch_normalization/beta/vFnormalize1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_1/gamma/vGnormalize2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_1/beta/vFnormalize2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_2/gamma/vGnormalize3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_2/beta/vFnormalize3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_3/gamma/vGnormalize4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_3/beta/vFnormalize4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_4/gamma/vGnormalize5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_4/beta/vFnormalize5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_5/gamma/vGnormalize6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_5/beta/vFnormalize6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_6/gamma/vGnormalize7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_6/beta/vFnormalize7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_7/gamma/vGnormalize8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_7/beta/vFnormalize8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/chess_model/batch_normalization_8/gamma/vGnormalize9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/chess_model/batch_normalization_8/beta/vFnormalize9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE Adam/chess_model/conv2d/kernel/vJconvolution1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/chess_model/conv2d/bias/vHconvolution1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_1/kernel/vJconvolution2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_1/bias/vHconvolution2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_2/kernel/vJconvolution3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_2/bias/vHconvolution3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_3/kernel/vJconvolution4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_3/bias/vHconvolution4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_4/kernel/vJconvolution5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_4/bias/vHconvolution5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_5/kernel/vJconvolution6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_5/bias/vHconvolution6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_6/kernel/vJconvolution7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_6/bias/vHconvolution7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_7/kernel/vJconvolution8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_7/bias/vHconvolution8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/chess_model/conv2d_8/kernel/vJconvolution9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/chess_model/conv2d_8/bias/vHconvolution9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE!Adam/chess_model/dense_2/kernel/vGvalueHead/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/chess_model/dense_2/bias/vEvalueHead/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputsPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputschess_model/conv2d/kernelchess_model/conv2d/bias%chess_model/batch_normalization/gamma$chess_model/batch_normalization/beta+chess_model/batch_normalization/moving_mean/chess_model/batch_normalization/moving_variancechess_model/conv2d_1/kernelchess_model/conv2d_1/bias'chess_model/batch_normalization_1/gamma&chess_model/batch_normalization_1/beta-chess_model/batch_normalization_1/moving_mean1chess_model/batch_normalization_1/moving_variancechess_model/conv2d_2/kernelchess_model/conv2d_2/bias'chess_model/batch_normalization_2/gamma&chess_model/batch_normalization_2/beta-chess_model/batch_normalization_2/moving_mean1chess_model/batch_normalization_2/moving_variancechess_model/conv2d_3/kernelchess_model/conv2d_3/bias'chess_model/batch_normalization_3/gamma&chess_model/batch_normalization_3/beta-chess_model/batch_normalization_3/moving_mean1chess_model/batch_normalization_3/moving_variancechess_model/conv2d_4/kernelchess_model/conv2d_4/bias'chess_model/batch_normalization_4/gamma&chess_model/batch_normalization_4/beta-chess_model/batch_normalization_4/moving_mean1chess_model/batch_normalization_4/moving_variancechess_model/conv2d_5/kernelchess_model/conv2d_5/bias'chess_model/batch_normalization_5/gamma&chess_model/batch_normalization_5/beta-chess_model/batch_normalization_5/moving_mean1chess_model/batch_normalization_5/moving_variancechess_model/conv2d_6/kernelchess_model/conv2d_6/bias'chess_model/batch_normalization_6/gamma&chess_model/batch_normalization_6/beta-chess_model/batch_normalization_6/moving_mean1chess_model/batch_normalization_6/moving_variancechess_model/conv2d_7/kernelchess_model/conv2d_7/bias'chess_model/batch_normalization_7/gamma&chess_model/batch_normalization_7/beta-chess_model/batch_normalization_7/moving_mean1chess_model/batch_normalization_7/moving_variancechess_model/conv2d_8/kernelchess_model/conv2d_8/bias'chess_model/batch_normalization_8/gamma&chess_model/batch_normalization_8/beta-chess_model/batch_normalization_8/moving_mean1chess_model/batch_normalization_8/moving_variancechess_model/dense/kernelchess_model/dense/biaschess_model/dense_1/kernelchess_model/dense_1/biaschess_model/dense_2/kernelchess_model/dense_2/bias*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_178335
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?H
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,chess_model/dense/kernel/Read/ReadVariableOp*chess_model/dense/bias/Read/ReadVariableOp.chess_model/dense_1/kernel/Read/ReadVariableOp,chess_model/dense_1/bias/Read/ReadVariableOp9chess_model/batch_normalization/gamma/Read/ReadVariableOp8chess_model/batch_normalization/beta/Read/ReadVariableOp?chess_model/batch_normalization/moving_mean/Read/ReadVariableOpCchess_model/batch_normalization/moving_variance/Read/ReadVariableOp;chess_model/batch_normalization_1/gamma/Read/ReadVariableOp:chess_model/batch_normalization_1/beta/Read/ReadVariableOpAchess_model/batch_normalization_1/moving_mean/Read/ReadVariableOpEchess_model/batch_normalization_1/moving_variance/Read/ReadVariableOp;chess_model/batch_normalization_2/gamma/Read/ReadVariableOp:chess_model/batch_normalization_2/beta/Read/ReadVariableOpAchess_model/batch_normalization_2/moving_mean/Read/ReadVariableOpEchess_model/batch_normalization_2/moving_variance/Read/ReadVariableOp;chess_model/batch_normalization_3/gamma/Read/ReadVariableOp:chess_model/batch_normalization_3/beta/Read/ReadVariableOpAchess_model/batch_normalization_3/moving_mean/Read/ReadVariableOpEchess_model/batch_normalization_3/moving_variance/Read/ReadVariableOp;chess_model/batch_normalization_4/gamma/Read/ReadVariableOp:chess_model/batch_normalization_4/beta/Read/ReadVariableOpAchess_model/batch_normalization_4/moving_mean/Read/ReadVariableOpEchess_model/batch_normalization_4/moving_variance/Read/ReadVariableOp;chess_model/batch_normalization_5/gamma/Read/ReadVariableOp:chess_model/batch_normalization_5/beta/Read/ReadVariableOpAchess_model/batch_normalization_5/moving_mean/Read/ReadVariableOpEchess_model/batch_normalization_5/moving_variance/Read/ReadVariableOp;chess_model/batch_normalization_6/gamma/Read/ReadVariableOp:chess_model/batch_normalization_6/beta/Read/ReadVariableOpAchess_model/batch_normalization_6/moving_mean/Read/ReadVariableOpEchess_model/batch_normalization_6/moving_variance/Read/ReadVariableOp;chess_model/batch_normalization_7/gamma/Read/ReadVariableOp:chess_model/batch_normalization_7/beta/Read/ReadVariableOpAchess_model/batch_normalization_7/moving_mean/Read/ReadVariableOpEchess_model/batch_normalization_7/moving_variance/Read/ReadVariableOp;chess_model/batch_normalization_8/gamma/Read/ReadVariableOp:chess_model/batch_normalization_8/beta/Read/ReadVariableOpAchess_model/batch_normalization_8/moving_mean/Read/ReadVariableOpEchess_model/batch_normalization_8/moving_variance/Read/ReadVariableOp-chess_model/conv2d/kernel/Read/ReadVariableOp+chess_model/conv2d/bias/Read/ReadVariableOp/chess_model/conv2d_1/kernel/Read/ReadVariableOp-chess_model/conv2d_1/bias/Read/ReadVariableOp/chess_model/conv2d_2/kernel/Read/ReadVariableOp-chess_model/conv2d_2/bias/Read/ReadVariableOp/chess_model/conv2d_3/kernel/Read/ReadVariableOp-chess_model/conv2d_3/bias/Read/ReadVariableOp/chess_model/conv2d_4/kernel/Read/ReadVariableOp-chess_model/conv2d_4/bias/Read/ReadVariableOp/chess_model/conv2d_5/kernel/Read/ReadVariableOp-chess_model/conv2d_5/bias/Read/ReadVariableOp/chess_model/conv2d_6/kernel/Read/ReadVariableOp-chess_model/conv2d_6/bias/Read/ReadVariableOp/chess_model/conv2d_7/kernel/Read/ReadVariableOp-chess_model/conv2d_7/bias/Read/ReadVariableOp/chess_model/conv2d_8/kernel/Read/ReadVariableOp-chess_model/conv2d_8/bias/Read/ReadVariableOp.chess_model/dense_2/kernel/Read/ReadVariableOp,chess_model/dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3Adam/chess_model/dense/kernel/m/Read/ReadVariableOp1Adam/chess_model/dense/bias/m/Read/ReadVariableOp5Adam/chess_model/dense_1/kernel/m/Read/ReadVariableOp3Adam/chess_model/dense_1/bias/m/Read/ReadVariableOp@Adam/chess_model/batch_normalization/gamma/m/Read/ReadVariableOp?Adam/chess_model/batch_normalization/beta/m/Read/ReadVariableOpBAdam/chess_model/batch_normalization_1/gamma/m/Read/ReadVariableOpAAdam/chess_model/batch_normalization_1/beta/m/Read/ReadVariableOpBAdam/chess_model/batch_normalization_2/gamma/m/Read/ReadVariableOpAAdam/chess_model/batch_normalization_2/beta/m/Read/ReadVariableOpBAdam/chess_model/batch_normalization_3/gamma/m/Read/ReadVariableOpAAdam/chess_model/batch_normalization_3/beta/m/Read/ReadVariableOpBAdam/chess_model/batch_normalization_4/gamma/m/Read/ReadVariableOpAAdam/chess_model/batch_normalization_4/beta/m/Read/ReadVariableOpBAdam/chess_model/batch_normalization_5/gamma/m/Read/ReadVariableOpAAdam/chess_model/batch_normalization_5/beta/m/Read/ReadVariableOpBAdam/chess_model/batch_normalization_6/gamma/m/Read/ReadVariableOpAAdam/chess_model/batch_normalization_6/beta/m/Read/ReadVariableOpBAdam/chess_model/batch_normalization_7/gamma/m/Read/ReadVariableOpAAdam/chess_model/batch_normalization_7/beta/m/Read/ReadVariableOpBAdam/chess_model/batch_normalization_8/gamma/m/Read/ReadVariableOpAAdam/chess_model/batch_normalization_8/beta/m/Read/ReadVariableOp4Adam/chess_model/conv2d/kernel/m/Read/ReadVariableOp2Adam/chess_model/conv2d/bias/m/Read/ReadVariableOp6Adam/chess_model/conv2d_1/kernel/m/Read/ReadVariableOp4Adam/chess_model/conv2d_1/bias/m/Read/ReadVariableOp6Adam/chess_model/conv2d_2/kernel/m/Read/ReadVariableOp4Adam/chess_model/conv2d_2/bias/m/Read/ReadVariableOp6Adam/chess_model/conv2d_3/kernel/m/Read/ReadVariableOp4Adam/chess_model/conv2d_3/bias/m/Read/ReadVariableOp6Adam/chess_model/conv2d_4/kernel/m/Read/ReadVariableOp4Adam/chess_model/conv2d_4/bias/m/Read/ReadVariableOp6Adam/chess_model/conv2d_5/kernel/m/Read/ReadVariableOp4Adam/chess_model/conv2d_5/bias/m/Read/ReadVariableOp6Adam/chess_model/conv2d_6/kernel/m/Read/ReadVariableOp4Adam/chess_model/conv2d_6/bias/m/Read/ReadVariableOp6Adam/chess_model/conv2d_7/kernel/m/Read/ReadVariableOp4Adam/chess_model/conv2d_7/bias/m/Read/ReadVariableOp6Adam/chess_model/conv2d_8/kernel/m/Read/ReadVariableOp4Adam/chess_model/conv2d_8/bias/m/Read/ReadVariableOp5Adam/chess_model/dense_2/kernel/m/Read/ReadVariableOp3Adam/chess_model/dense_2/bias/m/Read/ReadVariableOp3Adam/chess_model/dense/kernel/v/Read/ReadVariableOp1Adam/chess_model/dense/bias/v/Read/ReadVariableOp5Adam/chess_model/dense_1/kernel/v/Read/ReadVariableOp3Adam/chess_model/dense_1/bias/v/Read/ReadVariableOp@Adam/chess_model/batch_normalization/gamma/v/Read/ReadVariableOp?Adam/chess_model/batch_normalization/beta/v/Read/ReadVariableOpBAdam/chess_model/batch_normalization_1/gamma/v/Read/ReadVariableOpAAdam/chess_model/batch_normalization_1/beta/v/Read/ReadVariableOpBAdam/chess_model/batch_normalization_2/gamma/v/Read/ReadVariableOpAAdam/chess_model/batch_normalization_2/beta/v/Read/ReadVariableOpBAdam/chess_model/batch_normalization_3/gamma/v/Read/ReadVariableOpAAdam/chess_model/batch_normalization_3/beta/v/Read/ReadVariableOpBAdam/chess_model/batch_normalization_4/gamma/v/Read/ReadVariableOpAAdam/chess_model/batch_normalization_4/beta/v/Read/ReadVariableOpBAdam/chess_model/batch_normalization_5/gamma/v/Read/ReadVariableOpAAdam/chess_model/batch_normalization_5/beta/v/Read/ReadVariableOpBAdam/chess_model/batch_normalization_6/gamma/v/Read/ReadVariableOpAAdam/chess_model/batch_normalization_6/beta/v/Read/ReadVariableOpBAdam/chess_model/batch_normalization_7/gamma/v/Read/ReadVariableOpAAdam/chess_model/batch_normalization_7/beta/v/Read/ReadVariableOpBAdam/chess_model/batch_normalization_8/gamma/v/Read/ReadVariableOpAAdam/chess_model/batch_normalization_8/beta/v/Read/ReadVariableOp4Adam/chess_model/conv2d/kernel/v/Read/ReadVariableOp2Adam/chess_model/conv2d/bias/v/Read/ReadVariableOp6Adam/chess_model/conv2d_1/kernel/v/Read/ReadVariableOp4Adam/chess_model/conv2d_1/bias/v/Read/ReadVariableOp6Adam/chess_model/conv2d_2/kernel/v/Read/ReadVariableOp4Adam/chess_model/conv2d_2/bias/v/Read/ReadVariableOp6Adam/chess_model/conv2d_3/kernel/v/Read/ReadVariableOp4Adam/chess_model/conv2d_3/bias/v/Read/ReadVariableOp6Adam/chess_model/conv2d_4/kernel/v/Read/ReadVariableOp4Adam/chess_model/conv2d_4/bias/v/Read/ReadVariableOp6Adam/chess_model/conv2d_5/kernel/v/Read/ReadVariableOp4Adam/chess_model/conv2d_5/bias/v/Read/ReadVariableOp6Adam/chess_model/conv2d_6/kernel/v/Read/ReadVariableOp4Adam/chess_model/conv2d_6/bias/v/Read/ReadVariableOp6Adam/chess_model/conv2d_7/kernel/v/Read/ReadVariableOp4Adam/chess_model/conv2d_7/bias/v/Read/ReadVariableOp6Adam/chess_model/conv2d_8/kernel/v/Read/ReadVariableOp4Adam/chess_model/conv2d_8/bias/v/Read/ReadVariableOp5Adam/chess_model/dense_2/kernel/v/Read/ReadVariableOp3Adam/chess_model/dense_2/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_180533
?0
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamechess_model/dense/kernelchess_model/dense/biaschess_model/dense_1/kernelchess_model/dense_1/bias%chess_model/batch_normalization/gamma$chess_model/batch_normalization/beta+chess_model/batch_normalization/moving_mean/chess_model/batch_normalization/moving_variance'chess_model/batch_normalization_1/gamma&chess_model/batch_normalization_1/beta-chess_model/batch_normalization_1/moving_mean1chess_model/batch_normalization_1/moving_variance'chess_model/batch_normalization_2/gamma&chess_model/batch_normalization_2/beta-chess_model/batch_normalization_2/moving_mean1chess_model/batch_normalization_2/moving_variance'chess_model/batch_normalization_3/gamma&chess_model/batch_normalization_3/beta-chess_model/batch_normalization_3/moving_mean1chess_model/batch_normalization_3/moving_variance'chess_model/batch_normalization_4/gamma&chess_model/batch_normalization_4/beta-chess_model/batch_normalization_4/moving_mean1chess_model/batch_normalization_4/moving_variance'chess_model/batch_normalization_5/gamma&chess_model/batch_normalization_5/beta-chess_model/batch_normalization_5/moving_mean1chess_model/batch_normalization_5/moving_variance'chess_model/batch_normalization_6/gamma&chess_model/batch_normalization_6/beta-chess_model/batch_normalization_6/moving_mean1chess_model/batch_normalization_6/moving_variance'chess_model/batch_normalization_7/gamma&chess_model/batch_normalization_7/beta-chess_model/batch_normalization_7/moving_mean1chess_model/batch_normalization_7/moving_variance'chess_model/batch_normalization_8/gamma&chess_model/batch_normalization_8/beta-chess_model/batch_normalization_8/moving_mean1chess_model/batch_normalization_8/moving_variancechess_model/conv2d/kernelchess_model/conv2d/biaschess_model/conv2d_1/kernelchess_model/conv2d_1/biaschess_model/conv2d_2/kernelchess_model/conv2d_2/biaschess_model/conv2d_3/kernelchess_model/conv2d_3/biaschess_model/conv2d_4/kernelchess_model/conv2d_4/biaschess_model/conv2d_5/kernelchess_model/conv2d_5/biaschess_model/conv2d_6/kernelchess_model/conv2d_6/biaschess_model/conv2d_7/kernelchess_model/conv2d_7/biaschess_model/conv2d_8/kernelchess_model/conv2d_8/biaschess_model/dense_2/kernelchess_model/dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/chess_model/dense/kernel/mAdam/chess_model/dense/bias/m!Adam/chess_model/dense_1/kernel/mAdam/chess_model/dense_1/bias/m,Adam/chess_model/batch_normalization/gamma/m+Adam/chess_model/batch_normalization/beta/m.Adam/chess_model/batch_normalization_1/gamma/m-Adam/chess_model/batch_normalization_1/beta/m.Adam/chess_model/batch_normalization_2/gamma/m-Adam/chess_model/batch_normalization_2/beta/m.Adam/chess_model/batch_normalization_3/gamma/m-Adam/chess_model/batch_normalization_3/beta/m.Adam/chess_model/batch_normalization_4/gamma/m-Adam/chess_model/batch_normalization_4/beta/m.Adam/chess_model/batch_normalization_5/gamma/m-Adam/chess_model/batch_normalization_5/beta/m.Adam/chess_model/batch_normalization_6/gamma/m-Adam/chess_model/batch_normalization_6/beta/m.Adam/chess_model/batch_normalization_7/gamma/m-Adam/chess_model/batch_normalization_7/beta/m.Adam/chess_model/batch_normalization_8/gamma/m-Adam/chess_model/batch_normalization_8/beta/m Adam/chess_model/conv2d/kernel/mAdam/chess_model/conv2d/bias/m"Adam/chess_model/conv2d_1/kernel/m Adam/chess_model/conv2d_1/bias/m"Adam/chess_model/conv2d_2/kernel/m Adam/chess_model/conv2d_2/bias/m"Adam/chess_model/conv2d_3/kernel/m Adam/chess_model/conv2d_3/bias/m"Adam/chess_model/conv2d_4/kernel/m Adam/chess_model/conv2d_4/bias/m"Adam/chess_model/conv2d_5/kernel/m Adam/chess_model/conv2d_5/bias/m"Adam/chess_model/conv2d_6/kernel/m Adam/chess_model/conv2d_6/bias/m"Adam/chess_model/conv2d_7/kernel/m Adam/chess_model/conv2d_7/bias/m"Adam/chess_model/conv2d_8/kernel/m Adam/chess_model/conv2d_8/bias/m!Adam/chess_model/dense_2/kernel/mAdam/chess_model/dense_2/bias/mAdam/chess_model/dense/kernel/vAdam/chess_model/dense/bias/v!Adam/chess_model/dense_1/kernel/vAdam/chess_model/dense_1/bias/v,Adam/chess_model/batch_normalization/gamma/v+Adam/chess_model/batch_normalization/beta/v.Adam/chess_model/batch_normalization_1/gamma/v-Adam/chess_model/batch_normalization_1/beta/v.Adam/chess_model/batch_normalization_2/gamma/v-Adam/chess_model/batch_normalization_2/beta/v.Adam/chess_model/batch_normalization_3/gamma/v-Adam/chess_model/batch_normalization_3/beta/v.Adam/chess_model/batch_normalization_4/gamma/v-Adam/chess_model/batch_normalization_4/beta/v.Adam/chess_model/batch_normalization_5/gamma/v-Adam/chess_model/batch_normalization_5/beta/v.Adam/chess_model/batch_normalization_6/gamma/v-Adam/chess_model/batch_normalization_6/beta/v.Adam/chess_model/batch_normalization_7/gamma/v-Adam/chess_model/batch_normalization_7/beta/v.Adam/chess_model/batch_normalization_8/gamma/v-Adam/chess_model/batch_normalization_8/beta/v Adam/chess_model/conv2d/kernel/vAdam/chess_model/conv2d/bias/v"Adam/chess_model/conv2d_1/kernel/v Adam/chess_model/conv2d_1/bias/v"Adam/chess_model/conv2d_2/kernel/v Adam/chess_model/conv2d_2/bias/v"Adam/chess_model/conv2d_3/kernel/v Adam/chess_model/conv2d_3/bias/v"Adam/chess_model/conv2d_4/kernel/v Adam/chess_model/conv2d_4/bias/v"Adam/chess_model/conv2d_5/kernel/v Adam/chess_model/conv2d_5/bias/v"Adam/chess_model/conv2d_6/kernel/v Adam/chess_model/conv2d_6/bias/v"Adam/chess_model/conv2d_7/kernel/v Adam/chess_model/conv2d_7/bias/v"Adam/chess_model/conv2d_8/kernel/v Adam/chess_model/conv2d_8/bias/v!Adam/chess_model/dense_2/kernel/vAdam/chess_model/dense_2/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_181002??(
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1220

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_4_layer_call_fn_179603

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1788442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_178128

inputs
chess_model_178006
chess_model_178008
chess_model_178010
chess_model_178012
chess_model_178014
chess_model_178016
chess_model_178018
chess_model_178020
chess_model_178022
chess_model_178024
chess_model_178026
chess_model_178028
chess_model_178030
chess_model_178032
chess_model_178034
chess_model_178036
chess_model_178038
chess_model_178040
chess_model_178042
chess_model_178044
chess_model_178046
chess_model_178048
chess_model_178050
chess_model_178052
chess_model_178054
chess_model_178056
chess_model_178058
chess_model_178060
chess_model_178062
chess_model_178064
chess_model_178066
chess_model_178068
chess_model_178070
chess_model_178072
chess_model_178074
chess_model_178076
chess_model_178078
chess_model_178080
chess_model_178082
chess_model_178084
chess_model_178086
chess_model_178088
chess_model_178090
chess_model_178092
chess_model_178094
chess_model_178096
chess_model_178098
chess_model_178100
chess_model_178102
chess_model_178104
chess_model_178106
chess_model_178108
chess_model_178110
chess_model_178112
chess_model_178114
chess_model_178116
chess_model_178118
chess_model_178120
chess_model_178122
chess_model_178124
identity??#chess_model/StatefulPartitionedCall?
#chess_model/StatefulPartitionedCallStatefulPartitionedCallinputschess_model_178006chess_model_178008chess_model_178010chess_model_178012chess_model_178014chess_model_178016chess_model_178018chess_model_178020chess_model_178022chess_model_178024chess_model_178026chess_model_178028chess_model_178030chess_model_178032chess_model_178034chess_model_178036chess_model_178038chess_model_178040chess_model_178042chess_model_178044chess_model_178046chess_model_178048chess_model_178050chess_model_178052chess_model_178054chess_model_178056chess_model_178058chess_model_178060chess_model_178062chess_model_178064chess_model_178066chess_model_178068chess_model_178070chess_model_178072chess_model_178074chess_model_178076chess_model_178078chess_model_178080chess_model_178082chess_model_178084chess_model_178086chess_model_178088chess_model_178090chess_model_178092chess_model_178094chess_model_178096chess_model_178098chess_model_178100chess_model_178102chess_model_178104chess_model_178106chess_model_178108chess_model_178110chess_model_178112chess_model_178114chess_model_178116chess_model_178118chess_model_178120chess_model_178122chess_model_178124*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *2
f-R+
)__inference_restored_function_body_1780052%
#chess_model/StatefulPartitionedCall?
IdentityIdentity,chess_model/StatefulPartitionedCall:output:0$^chess_model/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2J
#chess_model/StatefulPartitionedCall#chess_model/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_179495

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_10_1800405
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
Z
>__inference_re_lu_layer_call_and_return_conditional_losses_626

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_179833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_178428

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_179641

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_178844

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1911

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_8_layer_call_and_return_conditional_losses_425

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
ñ
?3
E__inference_chess_model_layer_call_and_return_conditional_losses_2351

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
re_lu/Relu_1Relu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dre_lu/Relu_1:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
add/addAddV2re_lu/Relu:activations:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2	
add/addl
re_lu/Relu_2Reluadd/add:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_2?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dre_lu/Relu_2:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
re_lu/Relu_3Relu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_3?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dre_lu/Relu_3:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
re_lu/Relu_4Relu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_4?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dre_lu/Relu_4:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
	add/add_1AddV2re_lu/Relu_3:activations:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
	add/add_1n
re_lu/Relu_5Reluadd/add_1:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_5?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dre_lu/Relu_5:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
re_lu/Relu_6Relu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_6?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dre_lu/Relu_6:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
re_lu/Relu_7Relu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_7?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dre_lu/Relu_7:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_8/BiasAdd?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
	add/add_2AddV2re_lu/Relu_6:activations:0*batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
	add/add_2n
re_lu/Relu_8Reluadd/add_2:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_8o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapere_lu/Relu_8:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Tanh?
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?
+chess_model/dense/kernel/Regularizer/SquareSquareBchess_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2-
+chess_model/dense/kernel/Regularizer/Square?
*chess_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*chess_model/dense/kernel/Regularizer/Const?
(chess_model/dense/kernel/Regularizer/SumSum/chess_model/dense/kernel/Regularizer/Square:y:03chess_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/Sum?
*chess_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*chess_model/dense/kernel/Regularizer/mul/x?
(chess_model/dense/kernel/Regularizer/mulMul3chess_model/dense/kernel/Regularizer/mul/x:output:01chess_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/mul?
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02>
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_1/kernel/Regularizer/SquareSquareDchess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2/
-chess_model/dense_1/kernel/Regularizer/Square?
,chess_model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_1/kernel/Regularizer/Const?
*chess_model/dense_1/kernel/Regularizer/SumSum1chess_model/dense_1/kernel/Regularizer/Square:y:05chess_model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/Sum?
,chess_model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_1/kernel/Regularizer/mul/x?
*chess_model/dense_1/kernel/Regularizer/mulMul5chess_model/dense_1/kernel/Regularizer/mul/x:output:03chess_model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/mul?
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02=
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
,chess_model/conv2d/kernel/Regularizer/SquareSquareCchess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:?2.
,chess_model/conv2d/kernel/Regularizer/Square?
+chess_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+chess_model/conv2d/kernel/Regularizer/Const?
)chess_model/conv2d/kernel/Regularizer/SumSum0chess_model/conv2d/kernel/Regularizer/Square:y:04chess_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/Sum?
+chess_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+chess_model/conv2d/kernel/Regularizer/mul/x?
)chess_model/conv2d/kernel/Regularizer/mulMul4chess_model/conv2d/kernel/Regularizer/mul/x:output:02chess_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/mul?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_1/kernel/Regularizer/SquareSquareEchess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_1/kernel/Regularizer/Square?
-chess_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_1/kernel/Regularizer/Const?
+chess_model/conv2d_1/kernel/Regularizer/SumSum2chess_model/conv2d_1/kernel/Regularizer/Square:y:06chess_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/Sum?
-chess_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_1/kernel/Regularizer/mul/x?
+chess_model/conv2d_1/kernel/Regularizer/mulMul6chess_model/conv2d_1/kernel/Regularizer/mul/x:output:04chess_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/mul?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_2/kernel/Regularizer/SquareSquareEchess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_2/kernel/Regularizer/Square?
-chess_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_2/kernel/Regularizer/Const?
+chess_model/conv2d_2/kernel/Regularizer/SumSum2chess_model/conv2d_2/kernel/Regularizer/Square:y:06chess_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/Sum?
-chess_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_2/kernel/Regularizer/mul/x?
+chess_model/conv2d_2/kernel/Regularizer/mulMul6chess_model/conv2d_2/kernel/Regularizer/mul/x:output:04chess_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/mul?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_3/kernel/Regularizer/SquareSquareEchess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_3/kernel/Regularizer/Square?
-chess_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_3/kernel/Regularizer/Const?
+chess_model/conv2d_3/kernel/Regularizer/SumSum2chess_model/conv2d_3/kernel/Regularizer/Square:y:06chess_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/Sum?
-chess_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_3/kernel/Regularizer/mul/x?
+chess_model/conv2d_3/kernel/Regularizer/mulMul6chess_model/conv2d_3/kernel/Regularizer/mul/x:output:04chess_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/mul?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_4/kernel/Regularizer/SquareSquareEchess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_4/kernel/Regularizer/Square?
-chess_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_4/kernel/Regularizer/Const?
+chess_model/conv2d_4/kernel/Regularizer/SumSum2chess_model/conv2d_4/kernel/Regularizer/Square:y:06chess_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/Sum?
-chess_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_4/kernel/Regularizer/mul/x?
+chess_model/conv2d_4/kernel/Regularizer/mulMul6chess_model/conv2d_4/kernel/Regularizer/mul/x:output:04chess_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/mul?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_5/kernel/Regularizer/SquareSquareEchess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_5/kernel/Regularizer/Square?
-chess_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_5/kernel/Regularizer/Const?
+chess_model/conv2d_5/kernel/Regularizer/SumSum2chess_model/conv2d_5/kernel/Regularizer/Square:y:06chess_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/Sum?
-chess_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_5/kernel/Regularizer/mul/x?
+chess_model/conv2d_5/kernel/Regularizer/mulMul6chess_model/conv2d_5/kernel/Regularizer/mul/x:output:04chess_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/mul?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_6/kernel/Regularizer/SquareSquareEchess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_6/kernel/Regularizer/Square?
-chess_model/conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_6/kernel/Regularizer/Const?
+chess_model/conv2d_6/kernel/Regularizer/SumSum2chess_model/conv2d_6/kernel/Regularizer/Square:y:06chess_model/conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/Sum?
-chess_model/conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_6/kernel/Regularizer/mul/x?
+chess_model/conv2d_6/kernel/Regularizer/mulMul6chess_model/conv2d_6/kernel/Regularizer/mul/x:output:04chess_model/conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/mul?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_7/kernel/Regularizer/SquareSquareEchess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_7/kernel/Regularizer/Square?
-chess_model/conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_7/kernel/Regularizer/Const?
+chess_model/conv2d_7/kernel/Regularizer/SumSum2chess_model/conv2d_7/kernel/Regularizer/Square:y:06chess_model/conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/Sum?
-chess_model/conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_7/kernel/Regularizer/mul/x?
+chess_model/conv2d_7/kernel/Regularizer/mulMul6chess_model/conv2d_7/kernel/Regularizer/mul/x:output:04chess_model/conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/mul?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_8/kernel/Regularizer/SquareSquareEchess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_8/kernel/Regularizer/Square?
-chess_model/conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_8/kernel/Regularizer/Const?
+chess_model/conv2d_8/kernel/Regularizer/SumSum2chess_model/conv2d_8/kernel/Regularizer/Square:y:06chess_model/conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/Sum?
-chess_model/conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_8/kernel/Regularizer/mul/x?
+chess_model/conv2d_8/kernel/Regularizer/mulMul6chess_model/conv2d_8/kernel/Regularizer/mul/x:output:04chess_model/conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/mul?
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02>
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_2/kernel/Regularizer/SquareSquareDchess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2/
-chess_model/dense_2/kernel/Regularizer/Square?
,chess_model/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_2/kernel/Regularizer/Const?
*chess_model/dense_2/kernel/Regularizer/SumSum1chess_model/dense_2/kernel/Regularizer/Square:y:05chess_model/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/Sum?
,chess_model/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_2/kernel/Regularizer/mul/x?
*chess_model/dense_2/kernel/Regularizer/mulMul5chess_model/dense_2/kernel/Regularizer/mul/x:output:03chess_model/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/mul?
IdentityIdentitydense_2/Tanh:y:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1<^chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp;^chess_model/dense/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12z
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp2x
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_179623

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2574

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
[
?__inference_re_lu_layer_call_and_return_conditional_losses_1988

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_2_1799525
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:?*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:?2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
B__inference_conv2d_8_layer_call_and_return_conditional_losses_3354

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_8/kernel/Regularizer/SquareSquareEchess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_8/kernel/Regularizer/Square?
-chess_model/conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_8/kernel/Regularizer/Const?
+chess_model/conv2d_8/kernel/Regularizer/SumSum2chess_model/conv2d_8/kernel/Regularizer/Square:y:06chess_model/conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/Sum?
-chess_model/conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_8/kernel/Regularizer/mul/x?
+chess_model/conv2d_8/kernel/Regularizer/mulMul6chess_model/conv2d_8/kernel/Regularizer/mul/x:output:04chess_model/conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_chess_model_layer_call_fn_4020
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*L
_read_only_resource_inputs.
,*	
 !"%&'(+,-.1234789:;<*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_chess_model_layer_call_and_return_conditional_losses_39552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
g
=__inference_add_layer_call_and_return_conditional_losses_3518

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:??????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3476

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
E__inference_chess_model_layer_call_and_return_conditional_losses_3955

inputs
conv2d_3312
conv2d_3314
batch_normalization_3317
batch_normalization_3319
batch_normalization_3321
batch_normalization_3323
conv2d_1_3327
conv2d_1_3329
batch_normalization_1_3332
batch_normalization_1_3334
batch_normalization_1_3336
batch_normalization_1_3338
conv2d_2_3342
conv2d_2_3344
batch_normalization_2_3347
batch_normalization_2_3349
batch_normalization_2_3351
batch_normalization_2_3353
conv2d_3_3358
conv2d_3_3360
batch_normalization_3_3363
batch_normalization_3_3365
batch_normalization_3_3367
batch_normalization_3_3369
conv2d_4_3373
conv2d_4_3375
batch_normalization_4_3378
batch_normalization_4_3380
batch_normalization_4_3382
batch_normalization_4_3384
conv2d_5_3388
conv2d_5_3390
batch_normalization_5_3393
batch_normalization_5_3395
batch_normalization_5_3397
batch_normalization_5_3399
conv2d_6_3404
conv2d_6_3406
batch_normalization_6_3409
batch_normalization_6_3411
batch_normalization_6_3413
batch_normalization_6_3415
conv2d_7_3419
conv2d_7_3421
batch_normalization_7_3424
batch_normalization_7_3426
batch_normalization_7_3428
batch_normalization_7_3430
conv2d_8_3434
conv2d_8_3436
batch_normalization_8_3439
batch_normalization_8_3441
batch_normalization_8_3443
batch_normalization_8_3445

dense_3452

dense_3454
dense_1_3457
dense_1_3459
dense_2_3462
dense_2_3464
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3312conv2d_3314*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_5042 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3317batch_normalization_3319batch_normalization_3321batch_normalization_3323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_37152-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_19882
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_3327conv2d_1_3329*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_11412"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_3332batch_normalization_1_3334batch_normalization_1_3336batch_normalization_1_3338*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_7262/
-batch_normalization_1/StatefulPartitionedCall?
re_lu/PartitionedCall_1PartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_19882
re_lu/PartitionedCall_1?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_1:output:0conv2d_2_3342conv2d_2_3344*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_35792"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_3347batch_normalization_2_3349batch_normalization_2_3351batch_normalization_2_3353*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_27112/
-batch_normalization_2/StatefulPartitionedCall?
add/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:06batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_37482
add/PartitionedCall?
re_lu/PartitionedCall_2PartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_19882
re_lu/PartitionedCall_2?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_2:output:0conv2d_3_3358conv2d_3_3360*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_11842"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_3363batch_normalization_3_3365batch_normalization_3_3367batch_normalization_3_3369*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6872/
-batch_normalization_3/StatefulPartitionedCall?
re_lu/PartitionedCall_3PartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6922
re_lu/PartitionedCall_3?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_3:output:0conv2d_4_3373conv2d_4_3375*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_35122"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_3378batch_normalization_4_3380batch_normalization_4_3382batch_normalization_4_3384*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25742/
-batch_normalization_4/StatefulPartitionedCall?
re_lu/PartitionedCall_4PartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6922
re_lu/PartitionedCall_4?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_4:output:0conv2d_5_3388conv2d_5_3390*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_34252"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_3393batch_normalization_5_3395batch_normalization_5_3397batch_normalization_5_3399*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_19112/
-batch_normalization_5/StatefulPartitionedCall?
add/PartitionedCall_1PartitionedCall re_lu/PartitionedCall_3:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_34582
add/PartitionedCall_1?
re_lu/PartitionedCall_5PartitionedCalladd/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6922
re_lu/PartitionedCall_5?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_5:output:0conv2d_6_3404conv2d_6_3406*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_6_layer_call_and_return_conditional_losses_7082"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_3409batch_normalization_6_3411batch_normalization_6_3413batch_normalization_6_3415*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_37842/
-batch_normalization_6/StatefulPartitionedCall?
re_lu/PartitionedCall_6PartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6262
re_lu/PartitionedCall_6?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_6:output:0conv2d_7_3419conv2d_7_3421*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_7_layer_call_and_return_conditional_losses_6422"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_3424batch_normalization_7_3426batch_normalization_7_3428batch_normalization_7_3430*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_12202/
-batch_normalization_7/StatefulPartitionedCall?
re_lu/PartitionedCall_7PartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6262
re_lu/PartitionedCall_7?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_7:output:0conv2d_8_3434conv2d_8_3436*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_33542"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_3439batch_normalization_8_3441batch_normalization_8_3443batch_normalization_8_3445*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4252/
-batch_normalization_8/StatefulPartitionedCall?
add/PartitionedCall_2PartitionedCall re_lu/PartitionedCall_6:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_35182
add/PartitionedCall_2?
re_lu/PartitionedCall_8PartitionedCalladd/PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6262
re_lu/PartitionedCall_8o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape re_lu/PartitionedCall_8:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0
dense_3452
dense_3454*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_26562
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3457dense_1_3459*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_33202!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_3462dense_2_3464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_34422!
dense_2/StatefulPartitionedCall?
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_3452* 
_output_shapes
:
??*
dtype02<
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?
+chess_model/dense/kernel/Regularizer/SquareSquareBchess_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2-
+chess_model/dense/kernel/Regularizer/Square?
*chess_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*chess_model/dense/kernel/Regularizer/Const?
(chess_model/dense/kernel/Regularizer/SumSum/chess_model/dense/kernel/Regularizer/Square:y:03chess_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/Sum?
*chess_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*chess_model/dense/kernel/Regularizer/mul/x?
(chess_model/dense/kernel/Regularizer/mulMul3chess_model/dense/kernel/Regularizer/mul/x:output:01chess_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/mul?
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3457* 
_output_shapes
:
??*
dtype02>
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_1/kernel/Regularizer/SquareSquareDchess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2/
-chess_model/dense_1/kernel/Regularizer/Square?
,chess_model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_1/kernel/Regularizer/Const?
*chess_model/dense_1/kernel/Regularizer/SumSum1chess_model/dense_1/kernel/Regularizer/Square:y:05chess_model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/Sum?
,chess_model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_1/kernel/Regularizer/mul/x?
*chess_model/dense_1/kernel/Regularizer/mulMul5chess_model/dense_1/kernel/Regularizer/mul/x:output:03chess_model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/mul?
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3312*'
_output_shapes
:?*
dtype02=
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
,chess_model/conv2d/kernel/Regularizer/SquareSquareCchess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:?2.
,chess_model/conv2d/kernel/Regularizer/Square?
+chess_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+chess_model/conv2d/kernel/Regularizer/Const?
)chess_model/conv2d/kernel/Regularizer/SumSum0chess_model/conv2d/kernel/Regularizer/Square:y:04chess_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/Sum?
+chess_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+chess_model/conv2d/kernel/Regularizer/mul/x?
)chess_model/conv2d/kernel/Regularizer/mulMul4chess_model/conv2d/kernel/Regularizer/mul/x:output:02chess_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/mul?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_3327*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_1/kernel/Regularizer/SquareSquareEchess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_1/kernel/Regularizer/Square?
-chess_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_1/kernel/Regularizer/Const?
+chess_model/conv2d_1/kernel/Regularizer/SumSum2chess_model/conv2d_1/kernel/Regularizer/Square:y:06chess_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/Sum?
-chess_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_1/kernel/Regularizer/mul/x?
+chess_model/conv2d_1/kernel/Regularizer/mulMul6chess_model/conv2d_1/kernel/Regularizer/mul/x:output:04chess_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/mul?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_3342*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_2/kernel/Regularizer/SquareSquareEchess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_2/kernel/Regularizer/Square?
-chess_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_2/kernel/Regularizer/Const?
+chess_model/conv2d_2/kernel/Regularizer/SumSum2chess_model/conv2d_2/kernel/Regularizer/Square:y:06chess_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/Sum?
-chess_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_2/kernel/Regularizer/mul/x?
+chess_model/conv2d_2/kernel/Regularizer/mulMul6chess_model/conv2d_2/kernel/Regularizer/mul/x:output:04chess_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/mul?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_3358*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_3/kernel/Regularizer/SquareSquareEchess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_3/kernel/Regularizer/Square?
-chess_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_3/kernel/Regularizer/Const?
+chess_model/conv2d_3/kernel/Regularizer/SumSum2chess_model/conv2d_3/kernel/Regularizer/Square:y:06chess_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/Sum?
-chess_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_3/kernel/Regularizer/mul/x?
+chess_model/conv2d_3/kernel/Regularizer/mulMul6chess_model/conv2d_3/kernel/Regularizer/mul/x:output:04chess_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/mul?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_3373*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_4/kernel/Regularizer/SquareSquareEchess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_4/kernel/Regularizer/Square?
-chess_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_4/kernel/Regularizer/Const?
+chess_model/conv2d_4/kernel/Regularizer/SumSum2chess_model/conv2d_4/kernel/Regularizer/Square:y:06chess_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/Sum?
-chess_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_4/kernel/Regularizer/mul/x?
+chess_model/conv2d_4/kernel/Regularizer/mulMul6chess_model/conv2d_4/kernel/Regularizer/mul/x:output:04chess_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/mul?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_3388*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_5/kernel/Regularizer/SquareSquareEchess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_5/kernel/Regularizer/Square?
-chess_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_5/kernel/Regularizer/Const?
+chess_model/conv2d_5/kernel/Regularizer/SumSum2chess_model/conv2d_5/kernel/Regularizer/Square:y:06chess_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/Sum?
-chess_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_5/kernel/Regularizer/mul/x?
+chess_model/conv2d_5/kernel/Regularizer/mulMul6chess_model/conv2d_5/kernel/Regularizer/mul/x:output:04chess_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/mul?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_3404*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_6/kernel/Regularizer/SquareSquareEchess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_6/kernel/Regularizer/Square?
-chess_model/conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_6/kernel/Regularizer/Const?
+chess_model/conv2d_6/kernel/Regularizer/SumSum2chess_model/conv2d_6/kernel/Regularizer/Square:y:06chess_model/conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/Sum?
-chess_model/conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_6/kernel/Regularizer/mul/x?
+chess_model/conv2d_6/kernel/Regularizer/mulMul6chess_model/conv2d_6/kernel/Regularizer/mul/x:output:04chess_model/conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/mul?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_3419*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_7/kernel/Regularizer/SquareSquareEchess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_7/kernel/Regularizer/Square?
-chess_model/conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_7/kernel/Regularizer/Const?
+chess_model/conv2d_7/kernel/Regularizer/SumSum2chess_model/conv2d_7/kernel/Regularizer/Square:y:06chess_model/conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/Sum?
-chess_model/conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_7/kernel/Regularizer/mul/x?
+chess_model/conv2d_7/kernel/Regularizer/mulMul6chess_model/conv2d_7/kernel/Regularizer/mul/x:output:04chess_model/conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/mul?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_3434*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_8/kernel/Regularizer/SquareSquareEchess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_8/kernel/Regularizer/Square?
-chess_model/conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_8/kernel/Regularizer/Const?
+chess_model/conv2d_8/kernel/Regularizer/SumSum2chess_model/conv2d_8/kernel/Regularizer/Square:y:06chess_model/conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/Sum?
-chess_model/conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_8/kernel/Regularizer/mul/x?
+chess_model/conv2d_8/kernel/Regularizer/mulMul6chess_model/conv2d_8/kernel/Regularizer/mul/x:output:04chess_model/conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/mul?
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_3462*
_output_shapes
:	?*
dtype02>
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_2/kernel/Regularizer/SquareSquareDchess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2/
-chess_model/dense_2/kernel/Regularizer/Square?
,chess_model/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_2/kernel/Regularizer/Const?
*chess_model/dense_2/kernel/Regularizer/SumSum1chess_model/dense_2/kernel/Regularizer/Square:y:05chess_model/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/Sum?
,chess_model/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_2/kernel/Regularizer/mul/x?
*chess_model/dense_2/kernel/Regularizer/mulMul5chess_model/dense_2/kernel/Regularizer/mul/x:output:03chess_model/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/mul?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall<^chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp;^chess_model/dense/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2z
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp2x
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_5_1799855
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
Ͷ
?R
__inference__traced_save_180533
file_prefix7
3savev2_chess_model_dense_kernel_read_readvariableop5
1savev2_chess_model_dense_bias_read_readvariableop9
5savev2_chess_model_dense_1_kernel_read_readvariableop7
3savev2_chess_model_dense_1_bias_read_readvariableopD
@savev2_chess_model_batch_normalization_gamma_read_readvariableopC
?savev2_chess_model_batch_normalization_beta_read_readvariableopJ
Fsavev2_chess_model_batch_normalization_moving_mean_read_readvariableopN
Jsavev2_chess_model_batch_normalization_moving_variance_read_readvariableopF
Bsavev2_chess_model_batch_normalization_1_gamma_read_readvariableopE
Asavev2_chess_model_batch_normalization_1_beta_read_readvariableopL
Hsavev2_chess_model_batch_normalization_1_moving_mean_read_readvariableopP
Lsavev2_chess_model_batch_normalization_1_moving_variance_read_readvariableopF
Bsavev2_chess_model_batch_normalization_2_gamma_read_readvariableopE
Asavev2_chess_model_batch_normalization_2_beta_read_readvariableopL
Hsavev2_chess_model_batch_normalization_2_moving_mean_read_readvariableopP
Lsavev2_chess_model_batch_normalization_2_moving_variance_read_readvariableopF
Bsavev2_chess_model_batch_normalization_3_gamma_read_readvariableopE
Asavev2_chess_model_batch_normalization_3_beta_read_readvariableopL
Hsavev2_chess_model_batch_normalization_3_moving_mean_read_readvariableopP
Lsavev2_chess_model_batch_normalization_3_moving_variance_read_readvariableopF
Bsavev2_chess_model_batch_normalization_4_gamma_read_readvariableopE
Asavev2_chess_model_batch_normalization_4_beta_read_readvariableopL
Hsavev2_chess_model_batch_normalization_4_moving_mean_read_readvariableopP
Lsavev2_chess_model_batch_normalization_4_moving_variance_read_readvariableopF
Bsavev2_chess_model_batch_normalization_5_gamma_read_readvariableopE
Asavev2_chess_model_batch_normalization_5_beta_read_readvariableopL
Hsavev2_chess_model_batch_normalization_5_moving_mean_read_readvariableopP
Lsavev2_chess_model_batch_normalization_5_moving_variance_read_readvariableopF
Bsavev2_chess_model_batch_normalization_6_gamma_read_readvariableopE
Asavev2_chess_model_batch_normalization_6_beta_read_readvariableopL
Hsavev2_chess_model_batch_normalization_6_moving_mean_read_readvariableopP
Lsavev2_chess_model_batch_normalization_6_moving_variance_read_readvariableopF
Bsavev2_chess_model_batch_normalization_7_gamma_read_readvariableopE
Asavev2_chess_model_batch_normalization_7_beta_read_readvariableopL
Hsavev2_chess_model_batch_normalization_7_moving_mean_read_readvariableopP
Lsavev2_chess_model_batch_normalization_7_moving_variance_read_readvariableopF
Bsavev2_chess_model_batch_normalization_8_gamma_read_readvariableopE
Asavev2_chess_model_batch_normalization_8_beta_read_readvariableopL
Hsavev2_chess_model_batch_normalization_8_moving_mean_read_readvariableopP
Lsavev2_chess_model_batch_normalization_8_moving_variance_read_readvariableop8
4savev2_chess_model_conv2d_kernel_read_readvariableop6
2savev2_chess_model_conv2d_bias_read_readvariableop:
6savev2_chess_model_conv2d_1_kernel_read_readvariableop8
4savev2_chess_model_conv2d_1_bias_read_readvariableop:
6savev2_chess_model_conv2d_2_kernel_read_readvariableop8
4savev2_chess_model_conv2d_2_bias_read_readvariableop:
6savev2_chess_model_conv2d_3_kernel_read_readvariableop8
4savev2_chess_model_conv2d_3_bias_read_readvariableop:
6savev2_chess_model_conv2d_4_kernel_read_readvariableop8
4savev2_chess_model_conv2d_4_bias_read_readvariableop:
6savev2_chess_model_conv2d_5_kernel_read_readvariableop8
4savev2_chess_model_conv2d_5_bias_read_readvariableop:
6savev2_chess_model_conv2d_6_kernel_read_readvariableop8
4savev2_chess_model_conv2d_6_bias_read_readvariableop:
6savev2_chess_model_conv2d_7_kernel_read_readvariableop8
4savev2_chess_model_conv2d_7_bias_read_readvariableop:
6savev2_chess_model_conv2d_8_kernel_read_readvariableop8
4savev2_chess_model_conv2d_8_bias_read_readvariableop9
5savev2_chess_model_dense_2_kernel_read_readvariableop7
3savev2_chess_model_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop>
:savev2_adam_chess_model_dense_kernel_m_read_readvariableop<
8savev2_adam_chess_model_dense_bias_m_read_readvariableop@
<savev2_adam_chess_model_dense_1_kernel_m_read_readvariableop>
:savev2_adam_chess_model_dense_1_bias_m_read_readvariableopK
Gsavev2_adam_chess_model_batch_normalization_gamma_m_read_readvariableopJ
Fsavev2_adam_chess_model_batch_normalization_beta_m_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_1_gamma_m_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_1_beta_m_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_2_gamma_m_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_2_beta_m_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_3_gamma_m_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_3_beta_m_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_4_gamma_m_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_4_beta_m_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_5_gamma_m_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_5_beta_m_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_6_gamma_m_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_6_beta_m_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_7_gamma_m_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_7_beta_m_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_8_gamma_m_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_8_beta_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_kernel_m_read_readvariableop=
9savev2_adam_chess_model_conv2d_bias_m_read_readvariableopA
=savev2_adam_chess_model_conv2d_1_kernel_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_chess_model_conv2d_2_kernel_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_2_bias_m_read_readvariableopA
=savev2_adam_chess_model_conv2d_3_kernel_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_chess_model_conv2d_4_kernel_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_4_bias_m_read_readvariableopA
=savev2_adam_chess_model_conv2d_5_kernel_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_5_bias_m_read_readvariableopA
=savev2_adam_chess_model_conv2d_6_kernel_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_6_bias_m_read_readvariableopA
=savev2_adam_chess_model_conv2d_7_kernel_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_7_bias_m_read_readvariableopA
=savev2_adam_chess_model_conv2d_8_kernel_m_read_readvariableop?
;savev2_adam_chess_model_conv2d_8_bias_m_read_readvariableop@
<savev2_adam_chess_model_dense_2_kernel_m_read_readvariableop>
:savev2_adam_chess_model_dense_2_bias_m_read_readvariableop>
:savev2_adam_chess_model_dense_kernel_v_read_readvariableop<
8savev2_adam_chess_model_dense_bias_v_read_readvariableop@
<savev2_adam_chess_model_dense_1_kernel_v_read_readvariableop>
:savev2_adam_chess_model_dense_1_bias_v_read_readvariableopK
Gsavev2_adam_chess_model_batch_normalization_gamma_v_read_readvariableopJ
Fsavev2_adam_chess_model_batch_normalization_beta_v_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_1_gamma_v_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_1_beta_v_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_2_gamma_v_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_2_beta_v_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_3_gamma_v_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_3_beta_v_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_4_gamma_v_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_4_beta_v_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_5_gamma_v_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_5_beta_v_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_6_gamma_v_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_6_beta_v_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_7_gamma_v_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_7_beta_v_read_readvariableopM
Isavev2_adam_chess_model_batch_normalization_8_gamma_v_read_readvariableopL
Hsavev2_adam_chess_model_batch_normalization_8_beta_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_kernel_v_read_readvariableop=
9savev2_adam_chess_model_conv2d_bias_v_read_readvariableopA
=savev2_adam_chess_model_conv2d_1_kernel_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_chess_model_conv2d_2_kernel_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_2_bias_v_read_readvariableopA
=savev2_adam_chess_model_conv2d_3_kernel_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_chess_model_conv2d_4_kernel_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_4_bias_v_read_readvariableopA
=savev2_adam_chess_model_conv2d_5_kernel_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_5_bias_v_read_readvariableopA
=savev2_adam_chess_model_conv2d_6_kernel_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_6_bias_v_read_readvariableopA
=savev2_adam_chess_model_conv2d_7_kernel_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_7_bias_v_read_readvariableopA
=savev2_adam_chess_model_conv2d_8_kernel_v_read_readvariableop?
;savev2_adam_chess_model_conv2d_8_bias_v_read_readvariableop@
<savev2_adam_chess_model_dense_2_kernel_v_read_readvariableop>
:savev2_adam_chess_model_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?K
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?J
value?JB?J?B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB+normalize1/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize1/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize2/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize2/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize3/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize3/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize4/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize4/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize5/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize5/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize6/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize6/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize7/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize7/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize8/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize8/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize9/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize9/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB.convolution1/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution1/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution2/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution2/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution3/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution3/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution4/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution4/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution5/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution5/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution6/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution6/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution7/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution7/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution8/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution8/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution9/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution9/bias/.ATTRIBUTES/VARIABLE_VALUEB+valueHead/kernel/.ATTRIBUTES/VARIABLE_VALUEB)valueHead/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGvalueHead/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEvalueHead/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGvalueHead/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEvalueHead/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?O
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_chess_model_dense_kernel_read_readvariableop1savev2_chess_model_dense_bias_read_readvariableop5savev2_chess_model_dense_1_kernel_read_readvariableop3savev2_chess_model_dense_1_bias_read_readvariableop@savev2_chess_model_batch_normalization_gamma_read_readvariableop?savev2_chess_model_batch_normalization_beta_read_readvariableopFsavev2_chess_model_batch_normalization_moving_mean_read_readvariableopJsavev2_chess_model_batch_normalization_moving_variance_read_readvariableopBsavev2_chess_model_batch_normalization_1_gamma_read_readvariableopAsavev2_chess_model_batch_normalization_1_beta_read_readvariableopHsavev2_chess_model_batch_normalization_1_moving_mean_read_readvariableopLsavev2_chess_model_batch_normalization_1_moving_variance_read_readvariableopBsavev2_chess_model_batch_normalization_2_gamma_read_readvariableopAsavev2_chess_model_batch_normalization_2_beta_read_readvariableopHsavev2_chess_model_batch_normalization_2_moving_mean_read_readvariableopLsavev2_chess_model_batch_normalization_2_moving_variance_read_readvariableopBsavev2_chess_model_batch_normalization_3_gamma_read_readvariableopAsavev2_chess_model_batch_normalization_3_beta_read_readvariableopHsavev2_chess_model_batch_normalization_3_moving_mean_read_readvariableopLsavev2_chess_model_batch_normalization_3_moving_variance_read_readvariableopBsavev2_chess_model_batch_normalization_4_gamma_read_readvariableopAsavev2_chess_model_batch_normalization_4_beta_read_readvariableopHsavev2_chess_model_batch_normalization_4_moving_mean_read_readvariableopLsavev2_chess_model_batch_normalization_4_moving_variance_read_readvariableopBsavev2_chess_model_batch_normalization_5_gamma_read_readvariableopAsavev2_chess_model_batch_normalization_5_beta_read_readvariableopHsavev2_chess_model_batch_normalization_5_moving_mean_read_readvariableopLsavev2_chess_model_batch_normalization_5_moving_variance_read_readvariableopBsavev2_chess_model_batch_normalization_6_gamma_read_readvariableopAsavev2_chess_model_batch_normalization_6_beta_read_readvariableopHsavev2_chess_model_batch_normalization_6_moving_mean_read_readvariableopLsavev2_chess_model_batch_normalization_6_moving_variance_read_readvariableopBsavev2_chess_model_batch_normalization_7_gamma_read_readvariableopAsavev2_chess_model_batch_normalization_7_beta_read_readvariableopHsavev2_chess_model_batch_normalization_7_moving_mean_read_readvariableopLsavev2_chess_model_batch_normalization_7_moving_variance_read_readvariableopBsavev2_chess_model_batch_normalization_8_gamma_read_readvariableopAsavev2_chess_model_batch_normalization_8_beta_read_readvariableopHsavev2_chess_model_batch_normalization_8_moving_mean_read_readvariableopLsavev2_chess_model_batch_normalization_8_moving_variance_read_readvariableop4savev2_chess_model_conv2d_kernel_read_readvariableop2savev2_chess_model_conv2d_bias_read_readvariableop6savev2_chess_model_conv2d_1_kernel_read_readvariableop4savev2_chess_model_conv2d_1_bias_read_readvariableop6savev2_chess_model_conv2d_2_kernel_read_readvariableop4savev2_chess_model_conv2d_2_bias_read_readvariableop6savev2_chess_model_conv2d_3_kernel_read_readvariableop4savev2_chess_model_conv2d_3_bias_read_readvariableop6savev2_chess_model_conv2d_4_kernel_read_readvariableop4savev2_chess_model_conv2d_4_bias_read_readvariableop6savev2_chess_model_conv2d_5_kernel_read_readvariableop4savev2_chess_model_conv2d_5_bias_read_readvariableop6savev2_chess_model_conv2d_6_kernel_read_readvariableop4savev2_chess_model_conv2d_6_bias_read_readvariableop6savev2_chess_model_conv2d_7_kernel_read_readvariableop4savev2_chess_model_conv2d_7_bias_read_readvariableop6savev2_chess_model_conv2d_8_kernel_read_readvariableop4savev2_chess_model_conv2d_8_bias_read_readvariableop5savev2_chess_model_dense_2_kernel_read_readvariableop3savev2_chess_model_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_adam_chess_model_dense_kernel_m_read_readvariableop8savev2_adam_chess_model_dense_bias_m_read_readvariableop<savev2_adam_chess_model_dense_1_kernel_m_read_readvariableop:savev2_adam_chess_model_dense_1_bias_m_read_readvariableopGsavev2_adam_chess_model_batch_normalization_gamma_m_read_readvariableopFsavev2_adam_chess_model_batch_normalization_beta_m_read_readvariableopIsavev2_adam_chess_model_batch_normalization_1_gamma_m_read_readvariableopHsavev2_adam_chess_model_batch_normalization_1_beta_m_read_readvariableopIsavev2_adam_chess_model_batch_normalization_2_gamma_m_read_readvariableopHsavev2_adam_chess_model_batch_normalization_2_beta_m_read_readvariableopIsavev2_adam_chess_model_batch_normalization_3_gamma_m_read_readvariableopHsavev2_adam_chess_model_batch_normalization_3_beta_m_read_readvariableopIsavev2_adam_chess_model_batch_normalization_4_gamma_m_read_readvariableopHsavev2_adam_chess_model_batch_normalization_4_beta_m_read_readvariableopIsavev2_adam_chess_model_batch_normalization_5_gamma_m_read_readvariableopHsavev2_adam_chess_model_batch_normalization_5_beta_m_read_readvariableopIsavev2_adam_chess_model_batch_normalization_6_gamma_m_read_readvariableopHsavev2_adam_chess_model_batch_normalization_6_beta_m_read_readvariableopIsavev2_adam_chess_model_batch_normalization_7_gamma_m_read_readvariableopHsavev2_adam_chess_model_batch_normalization_7_beta_m_read_readvariableopIsavev2_adam_chess_model_batch_normalization_8_gamma_m_read_readvariableopHsavev2_adam_chess_model_batch_normalization_8_beta_m_read_readvariableop;savev2_adam_chess_model_conv2d_kernel_m_read_readvariableop9savev2_adam_chess_model_conv2d_bias_m_read_readvariableop=savev2_adam_chess_model_conv2d_1_kernel_m_read_readvariableop;savev2_adam_chess_model_conv2d_1_bias_m_read_readvariableop=savev2_adam_chess_model_conv2d_2_kernel_m_read_readvariableop;savev2_adam_chess_model_conv2d_2_bias_m_read_readvariableop=savev2_adam_chess_model_conv2d_3_kernel_m_read_readvariableop;savev2_adam_chess_model_conv2d_3_bias_m_read_readvariableop=savev2_adam_chess_model_conv2d_4_kernel_m_read_readvariableop;savev2_adam_chess_model_conv2d_4_bias_m_read_readvariableop=savev2_adam_chess_model_conv2d_5_kernel_m_read_readvariableop;savev2_adam_chess_model_conv2d_5_bias_m_read_readvariableop=savev2_adam_chess_model_conv2d_6_kernel_m_read_readvariableop;savev2_adam_chess_model_conv2d_6_bias_m_read_readvariableop=savev2_adam_chess_model_conv2d_7_kernel_m_read_readvariableop;savev2_adam_chess_model_conv2d_7_bias_m_read_readvariableop=savev2_adam_chess_model_conv2d_8_kernel_m_read_readvariableop;savev2_adam_chess_model_conv2d_8_bias_m_read_readvariableop<savev2_adam_chess_model_dense_2_kernel_m_read_readvariableop:savev2_adam_chess_model_dense_2_bias_m_read_readvariableop:savev2_adam_chess_model_dense_kernel_v_read_readvariableop8savev2_adam_chess_model_dense_bias_v_read_readvariableop<savev2_adam_chess_model_dense_1_kernel_v_read_readvariableop:savev2_adam_chess_model_dense_1_bias_v_read_readvariableopGsavev2_adam_chess_model_batch_normalization_gamma_v_read_readvariableopFsavev2_adam_chess_model_batch_normalization_beta_v_read_readvariableopIsavev2_adam_chess_model_batch_normalization_1_gamma_v_read_readvariableopHsavev2_adam_chess_model_batch_normalization_1_beta_v_read_readvariableopIsavev2_adam_chess_model_batch_normalization_2_gamma_v_read_readvariableopHsavev2_adam_chess_model_batch_normalization_2_beta_v_read_readvariableopIsavev2_adam_chess_model_batch_normalization_3_gamma_v_read_readvariableopHsavev2_adam_chess_model_batch_normalization_3_beta_v_read_readvariableopIsavev2_adam_chess_model_batch_normalization_4_gamma_v_read_readvariableopHsavev2_adam_chess_model_batch_normalization_4_beta_v_read_readvariableopIsavev2_adam_chess_model_batch_normalization_5_gamma_v_read_readvariableopHsavev2_adam_chess_model_batch_normalization_5_beta_v_read_readvariableopIsavev2_adam_chess_model_batch_normalization_6_gamma_v_read_readvariableopHsavev2_adam_chess_model_batch_normalization_6_beta_v_read_readvariableopIsavev2_adam_chess_model_batch_normalization_7_gamma_v_read_readvariableopHsavev2_adam_chess_model_batch_normalization_7_beta_v_read_readvariableopIsavev2_adam_chess_model_batch_normalization_8_gamma_v_read_readvariableopHsavev2_adam_chess_model_batch_normalization_8_beta_v_read_readvariableop;savev2_adam_chess_model_conv2d_kernel_v_read_readvariableop9savev2_adam_chess_model_conv2d_bias_v_read_readvariableop=savev2_adam_chess_model_conv2d_1_kernel_v_read_readvariableop;savev2_adam_chess_model_conv2d_1_bias_v_read_readvariableop=savev2_adam_chess_model_conv2d_2_kernel_v_read_readvariableop;savev2_adam_chess_model_conv2d_2_bias_v_read_readvariableop=savev2_adam_chess_model_conv2d_3_kernel_v_read_readvariableop;savev2_adam_chess_model_conv2d_3_bias_v_read_readvariableop=savev2_adam_chess_model_conv2d_4_kernel_v_read_readvariableop;savev2_adam_chess_model_conv2d_4_bias_v_read_readvariableop=savev2_adam_chess_model_conv2d_5_kernel_v_read_readvariableop;savev2_adam_chess_model_conv2d_5_bias_v_read_readvariableop=savev2_adam_chess_model_conv2d_6_kernel_v_read_readvariableop;savev2_adam_chess_model_conv2d_6_bias_v_read_readvariableop=savev2_adam_chess_model_conv2d_7_kernel_v_read_readvariableop;savev2_adam_chess_model_conv2d_7_bias_v_read_readvariableop=savev2_adam_chess_model_conv2d_8_kernel_v_read_readvariableop;savev2_adam_chess_model_conv2d_8_bias_v_read_readvariableop<savev2_adam_chess_model_dense_2_kernel_v_read_readvariableop:savev2_adam_chess_model_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:	?:: : : : : : : : : :
??:?:
??:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:	?::
??:?:
??:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:! 

_output_shapes	
:?:!!

_output_shapes	
:?:!"

_output_shapes	
:?:!#

_output_shapes	
:?:!$

_output_shapes	
:?:!%

_output_shapes	
:?:!&

_output_shapes	
:?:!'

_output_shapes	
:?:!(

_output_shapes	
:?:-))
'
_output_shapes
:?:!*

_output_shapes	
:?:.+*
(
_output_shapes
:??:!,

_output_shapes	
:?:.-*
(
_output_shapes
:??:!.

_output_shapes	
:?:./*
(
_output_shapes
:??:!0

_output_shapes	
:?:.1*
(
_output_shapes
:??:!2

_output_shapes	
:?:.3*
(
_output_shapes
:??:!4

_output_shapes	
:?:.5*
(
_output_shapes
:??:!6

_output_shapes	
:?:.7*
(
_output_shapes
:??:!8

_output_shapes	
:?:.9*
(
_output_shapes
:??:!:

_output_shapes	
:?:%;!

_output_shapes
:	?: <

_output_shapes
::=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :&F"
 
_output_shapes
:
??:!G

_output_shapes	
:?:&H"
 
_output_shapes
:
??:!I

_output_shapes	
:?:!J

_output_shapes	
:?:!K

_output_shapes	
:?:!L

_output_shapes	
:?:!M

_output_shapes	
:?:!N

_output_shapes	
:?:!O

_output_shapes	
:?:!P

_output_shapes	
:?:!Q

_output_shapes	
:?:!R

_output_shapes	
:?:!S

_output_shapes	
:?:!T

_output_shapes	
:?:!U

_output_shapes	
:?:!V

_output_shapes	
:?:!W

_output_shapes	
:?:!X

_output_shapes	
:?:!Y

_output_shapes	
:?:!Z

_output_shapes	
:?:![

_output_shapes	
:?:-\)
'
_output_shapes
:?:!]

_output_shapes	
:?:.^*
(
_output_shapes
:??:!_

_output_shapes	
:?:.`*
(
_output_shapes
:??:!a

_output_shapes	
:?:.b*
(
_output_shapes
:??:!c

_output_shapes	
:?:.d*
(
_output_shapes
:??:!e

_output_shapes	
:?:.f*
(
_output_shapes
:??:!g

_output_shapes	
:?:.h*
(
_output_shapes
:??:!i

_output_shapes	
:?:.j*
(
_output_shapes
:??:!k

_output_shapes	
:?:.l*
(
_output_shapes
:??:!m

_output_shapes	
:?:%n!

_output_shapes
:	?: o

_output_shapes
::&p"
 
_output_shapes
:
??:!q

_output_shapes	
:?:&r"
 
_output_shapes
:
??:!s

_output_shapes	
:?:!t

_output_shapes	
:?:!u

_output_shapes	
:?:!v

_output_shapes	
:?:!w

_output_shapes	
:?:!x

_output_shapes	
:?:!y

_output_shapes	
:?:!z

_output_shapes	
:?:!{

_output_shapes	
:?:!|

_output_shapes	
:?:!}

_output_shapes	
:?:!~

_output_shapes	
:?:!

_output_shapes	
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:.?)
'
_output_shapes
:?:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:&?!

_output_shapes
:	?:!?

_output_shapes
::?

_output_shapes
: 
?

?
__inference_loss_fn_4_1799745
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
N__inference_batch_normalization_7_layer_call_and_return_conditional_losses_771

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
=__inference_add_layer_call_and_return_conditional_losses_3748

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:??????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_726

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_3512

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_4/kernel/Regularizer/SquareSquareEchess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_4/kernel/Regularizer/Square?
-chess_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_4/kernel/Regularizer/Const?
+chess_model/conv2d_4/kernel/Regularizer/SumSum2chess_model/conv2d_4/kernel/Regularizer/Square:y:06chess_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/Sum?
-chess_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_4/kernel/Regularizer/mul/x?
+chess_model/conv2d_4/kernel/Regularizer/mulMul6chess_model/conv2d_4/kernel/Regularizer/mul/x:output:04chess_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_179431

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
L__inference_batch_normalization_layer_call_and_return_conditional_losses_397

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3409

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
E__inference_chess_model_layer_call_and_return_conditional_losses_4329

inputs
conv2d_1893
conv2d_1895
batch_normalization_1916
batch_normalization_1918
batch_normalization_1920
batch_normalization_1922
conv2d_1_1949
conv2d_1_1951
batch_normalization_1_1972
batch_normalization_1_1974
batch_normalization_1_1976
batch_normalization_1_1978
conv2d_2_1999
conv2d_2_2001
batch_normalization_2_2022
batch_normalization_2_2024
batch_normalization_2_2026
batch_normalization_2_2028
conv2d_3_2057
conv2d_3_2059
batch_normalization_3_2080
batch_normalization_3_2082
batch_normalization_3_2084
batch_normalization_3_2086
conv2d_4_2112
conv2d_4_2114
batch_normalization_4_2135
batch_normalization_4_2137
batch_normalization_4_2139
batch_normalization_4_2141
conv2d_5_2162
conv2d_5_2164
batch_normalization_5_2185
batch_normalization_5_2187
batch_normalization_5_2189
batch_normalization_5_2191
conv2d_6_2219
conv2d_6_2221
batch_normalization_6_2242
batch_normalization_6_2244
batch_normalization_6_2246
batch_normalization_6_2248
conv2d_7_2274
conv2d_7_2276
batch_normalization_7_2297
batch_normalization_7_2299
batch_normalization_7_2301
batch_normalization_7_2303
conv2d_8_2324
conv2d_8_2326
batch_normalization_8_2347
batch_normalization_8_2349
batch_normalization_8_2351
batch_normalization_8_2353

dense_2384

dense_2386
dense_1_2407
dense_1_2409
dense_2_2430
dense_2_2432
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1893conv2d_1895*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_5042 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1916batch_normalization_1918batch_normalization_1920batch_normalization_1922*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_3972-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_19882
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_1949conv2d_1_1951*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_11412"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1972batch_normalization_1_1974batch_normalization_1_1976batch_normalization_1_1978*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41032/
-batch_normalization_1/StatefulPartitionedCall?
re_lu/PartitionedCall_1PartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_19882
re_lu/PartitionedCall_1?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_1:output:0conv2d_2_1999conv2d_2_2001*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_35792"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_2022batch_normalization_2_2024batch_normalization_2_2026batch_normalization_2_2028*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34092/
-batch_normalization_2/StatefulPartitionedCall?
add/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:06batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_37482
add/PartitionedCall?
re_lu/PartitionedCall_2PartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_19882
re_lu/PartitionedCall_2?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_2:output:0conv2d_3_2057conv2d_3_2059*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_11842"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_2080batch_normalization_3_2082batch_normalization_3_2084batch_normalization_3_2086*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41582/
-batch_normalization_3/StatefulPartitionedCall?
re_lu/PartitionedCall_3PartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6922
re_lu/PartitionedCall_3?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_3:output:0conv2d_4_2112conv2d_4_2114*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_35122"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_2135batch_normalization_4_2137batch_normalization_4_2139batch_normalization_4_2141*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_25922/
-batch_normalization_4/StatefulPartitionedCall?
re_lu/PartitionedCall_4PartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6922
re_lu/PartitionedCall_4?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_4:output:0conv2d_5_2162conv2d_5_2164*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_34252"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_2185batch_normalization_5_2187batch_normalization_5_2189batch_normalization_5_2191*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5222/
-batch_normalization_5/StatefulPartitionedCall?
add/PartitionedCall_1PartitionedCall re_lu/PartitionedCall_3:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_34582
add/PartitionedCall_1?
re_lu/PartitionedCall_5PartitionedCalladd/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6922
re_lu/PartitionedCall_5?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_5:output:0conv2d_6_2219conv2d_6_2221*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_6_layer_call_and_return_conditional_losses_7082"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_2242batch_normalization_6_2244batch_normalization_6_2246batch_normalization_6_2248*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_20062/
-batch_normalization_6/StatefulPartitionedCall?
re_lu/PartitionedCall_6PartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6262
re_lu/PartitionedCall_6?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_6:output:0conv2d_7_2274conv2d_7_2276*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_7_layer_call_and_return_conditional_losses_6422"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_2297batch_normalization_7_2299batch_normalization_7_2301batch_normalization_7_2303*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7712/
-batch_normalization_7/StatefulPartitionedCall?
re_lu/PartitionedCall_7PartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6262
re_lu/PartitionedCall_7?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_7:output:0conv2d_8_2324conv2d_8_2326*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_33542"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_2347batch_normalization_8_2349batch_normalization_8_2351batch_normalization_8_2353*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34762/
-batch_normalization_8/StatefulPartitionedCall?
add/PartitionedCall_2PartitionedCall re_lu/PartitionedCall_6:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_add_layer_call_and_return_conditional_losses_35182
add/PartitionedCall_2?
re_lu/PartitionedCall_8PartitionedCalladd/PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_re_lu_layer_call_and_return_conditional_losses_6262
re_lu/PartitionedCall_8o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape re_lu/PartitionedCall_8:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0
dense_2384
dense_2386*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_26562
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2407dense_1_2409*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_33202!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_2430dense_2_2432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_34422!
dense_2/StatefulPartitionedCall?
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_2384* 
_output_shapes
:
??*
dtype02<
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?
+chess_model/dense/kernel/Regularizer/SquareSquareBchess_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2-
+chess_model/dense/kernel/Regularizer/Square?
*chess_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*chess_model/dense/kernel/Regularizer/Const?
(chess_model/dense/kernel/Regularizer/SumSum/chess_model/dense/kernel/Regularizer/Square:y:03chess_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/Sum?
*chess_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*chess_model/dense/kernel/Regularizer/mul/x?
(chess_model/dense/kernel/Regularizer/mulMul3chess_model/dense/kernel/Regularizer/mul/x:output:01chess_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/mul?
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2407* 
_output_shapes
:
??*
dtype02>
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_1/kernel/Regularizer/SquareSquareDchess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2/
-chess_model/dense_1/kernel/Regularizer/Square?
,chess_model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_1/kernel/Regularizer/Const?
*chess_model/dense_1/kernel/Regularizer/SumSum1chess_model/dense_1/kernel/Regularizer/Square:y:05chess_model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/Sum?
,chess_model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_1/kernel/Regularizer/mul/x?
*chess_model/dense_1/kernel/Regularizer/mulMul5chess_model/dense_1/kernel/Regularizer/mul/x:output:03chess_model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/mul?
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1893*'
_output_shapes
:?*
dtype02=
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
,chess_model/conv2d/kernel/Regularizer/SquareSquareCchess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:?2.
,chess_model/conv2d/kernel/Regularizer/Square?
+chess_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+chess_model/conv2d/kernel/Regularizer/Const?
)chess_model/conv2d/kernel/Regularizer/SumSum0chess_model/conv2d/kernel/Regularizer/Square:y:04chess_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/Sum?
+chess_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+chess_model/conv2d/kernel/Regularizer/mul/x?
)chess_model/conv2d/kernel/Regularizer/mulMul4chess_model/conv2d/kernel/Regularizer/mul/x:output:02chess_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/mul?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_1949*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_1/kernel/Regularizer/SquareSquareEchess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_1/kernel/Regularizer/Square?
-chess_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_1/kernel/Regularizer/Const?
+chess_model/conv2d_1/kernel/Regularizer/SumSum2chess_model/conv2d_1/kernel/Regularizer/Square:y:06chess_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/Sum?
-chess_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_1/kernel/Regularizer/mul/x?
+chess_model/conv2d_1/kernel/Regularizer/mulMul6chess_model/conv2d_1/kernel/Regularizer/mul/x:output:04chess_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/mul?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_1999*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_2/kernel/Regularizer/SquareSquareEchess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_2/kernel/Regularizer/Square?
-chess_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_2/kernel/Regularizer/Const?
+chess_model/conv2d_2/kernel/Regularizer/SumSum2chess_model/conv2d_2/kernel/Regularizer/Square:y:06chess_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/Sum?
-chess_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_2/kernel/Regularizer/mul/x?
+chess_model/conv2d_2/kernel/Regularizer/mulMul6chess_model/conv2d_2/kernel/Regularizer/mul/x:output:04chess_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/mul?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_2057*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_3/kernel/Regularizer/SquareSquareEchess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_3/kernel/Regularizer/Square?
-chess_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_3/kernel/Regularizer/Const?
+chess_model/conv2d_3/kernel/Regularizer/SumSum2chess_model/conv2d_3/kernel/Regularizer/Square:y:06chess_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/Sum?
-chess_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_3/kernel/Regularizer/mul/x?
+chess_model/conv2d_3/kernel/Regularizer/mulMul6chess_model/conv2d_3/kernel/Regularizer/mul/x:output:04chess_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/mul?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_2112*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_4/kernel/Regularizer/SquareSquareEchess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_4/kernel/Regularizer/Square?
-chess_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_4/kernel/Regularizer/Const?
+chess_model/conv2d_4/kernel/Regularizer/SumSum2chess_model/conv2d_4/kernel/Regularizer/Square:y:06chess_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/Sum?
-chess_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_4/kernel/Regularizer/mul/x?
+chess_model/conv2d_4/kernel/Regularizer/mulMul6chess_model/conv2d_4/kernel/Regularizer/mul/x:output:04chess_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/mul?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_2162*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_5/kernel/Regularizer/SquareSquareEchess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_5/kernel/Regularizer/Square?
-chess_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_5/kernel/Regularizer/Const?
+chess_model/conv2d_5/kernel/Regularizer/SumSum2chess_model/conv2d_5/kernel/Regularizer/Square:y:06chess_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/Sum?
-chess_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_5/kernel/Regularizer/mul/x?
+chess_model/conv2d_5/kernel/Regularizer/mulMul6chess_model/conv2d_5/kernel/Regularizer/mul/x:output:04chess_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/mul?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_2219*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_6/kernel/Regularizer/SquareSquareEchess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_6/kernel/Regularizer/Square?
-chess_model/conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_6/kernel/Regularizer/Const?
+chess_model/conv2d_6/kernel/Regularizer/SumSum2chess_model/conv2d_6/kernel/Regularizer/Square:y:06chess_model/conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/Sum?
-chess_model/conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_6/kernel/Regularizer/mul/x?
+chess_model/conv2d_6/kernel/Regularizer/mulMul6chess_model/conv2d_6/kernel/Regularizer/mul/x:output:04chess_model/conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/mul?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_2274*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_7/kernel/Regularizer/SquareSquareEchess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_7/kernel/Regularizer/Square?
-chess_model/conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_7/kernel/Regularizer/Const?
+chess_model/conv2d_7/kernel/Regularizer/SumSum2chess_model/conv2d_7/kernel/Regularizer/Square:y:06chess_model/conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/Sum?
-chess_model/conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_7/kernel/Regularizer/mul/x?
+chess_model/conv2d_7/kernel/Regularizer/mulMul6chess_model/conv2d_7/kernel/Regularizer/mul/x:output:04chess_model/conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/mul?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_2324*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_8/kernel/Regularizer/SquareSquareEchess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_8/kernel/Regularizer/Square?
-chess_model/conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_8/kernel/Regularizer/Const?
+chess_model/conv2d_8/kernel/Regularizer/SumSum2chess_model/conv2d_8/kernel/Regularizer/Square:y:06chess_model/conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/Sum?
-chess_model/conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_8/kernel/Regularizer/mul/x?
+chess_model/conv2d_8/kernel/Regularizer/mulMul6chess_model/conv2d_8/kernel/Regularizer/mul/x:output:04chess_model/conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/mul?
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_2430*
_output_shapes
:	?*
dtype02>
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_2/kernel/Regularizer/SquareSquareDchess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2/
-chess_model/dense_2/kernel/Regularizer/Square?
,chess_model/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_2/kernel/Regularizer/Const?
*chess_model/dense_2/kernel/Regularizer/SumSum1chess_model/dense_2/kernel/Regularizer/Square:y:05chess_model/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/Sum?
,chess_model/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_2/kernel/Regularizer/mul/x?
*chess_model/dense_2/kernel/Regularizer/mulMul5chess_model/dense_2/kernel/Regularizer/mul/x:output:03chess_model/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/mul?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall<^chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp;^chess_model/dense/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2z
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp2x
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1184

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_3/kernel/Regularizer/SquareSquareEchess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_3/kernel/Regularizer/Square?
-chess_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_3/kernel/Regularizer/Const?
+chess_model/conv2d_3/kernel/Regularizer/SumSum2chess_model/conv2d_3/kernel/Regularizer/Square:y:06chess_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/Sum?
-chess_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_3/kernel/Regularizer/mul/x?
+chess_model/conv2d_3/kernel/Regularizer/mulMul6chess_model/conv2d_3/kernel/Regularizer/mul/x:output:04chess_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_178605

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_4_layer_call_fn_179590

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1788132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_8_layer_call_fn_179859

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1792602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_179687

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?d
"__inference__traced_restore_181002
file_prefix-
)assignvariableop_chess_model_dense_kernel-
)assignvariableop_1_chess_model_dense_bias1
-assignvariableop_2_chess_model_dense_1_kernel/
+assignvariableop_3_chess_model_dense_1_bias<
8assignvariableop_4_chess_model_batch_normalization_gamma;
7assignvariableop_5_chess_model_batch_normalization_betaB
>assignvariableop_6_chess_model_batch_normalization_moving_meanF
Bassignvariableop_7_chess_model_batch_normalization_moving_variance>
:assignvariableop_8_chess_model_batch_normalization_1_gamma=
9assignvariableop_9_chess_model_batch_normalization_1_betaE
Aassignvariableop_10_chess_model_batch_normalization_1_moving_meanI
Eassignvariableop_11_chess_model_batch_normalization_1_moving_variance?
;assignvariableop_12_chess_model_batch_normalization_2_gamma>
:assignvariableop_13_chess_model_batch_normalization_2_betaE
Aassignvariableop_14_chess_model_batch_normalization_2_moving_meanI
Eassignvariableop_15_chess_model_batch_normalization_2_moving_variance?
;assignvariableop_16_chess_model_batch_normalization_3_gamma>
:assignvariableop_17_chess_model_batch_normalization_3_betaE
Aassignvariableop_18_chess_model_batch_normalization_3_moving_meanI
Eassignvariableop_19_chess_model_batch_normalization_3_moving_variance?
;assignvariableop_20_chess_model_batch_normalization_4_gamma>
:assignvariableop_21_chess_model_batch_normalization_4_betaE
Aassignvariableop_22_chess_model_batch_normalization_4_moving_meanI
Eassignvariableop_23_chess_model_batch_normalization_4_moving_variance?
;assignvariableop_24_chess_model_batch_normalization_5_gamma>
:assignvariableop_25_chess_model_batch_normalization_5_betaE
Aassignvariableop_26_chess_model_batch_normalization_5_moving_meanI
Eassignvariableop_27_chess_model_batch_normalization_5_moving_variance?
;assignvariableop_28_chess_model_batch_normalization_6_gamma>
:assignvariableop_29_chess_model_batch_normalization_6_betaE
Aassignvariableop_30_chess_model_batch_normalization_6_moving_meanI
Eassignvariableop_31_chess_model_batch_normalization_6_moving_variance?
;assignvariableop_32_chess_model_batch_normalization_7_gamma>
:assignvariableop_33_chess_model_batch_normalization_7_betaE
Aassignvariableop_34_chess_model_batch_normalization_7_moving_meanI
Eassignvariableop_35_chess_model_batch_normalization_7_moving_variance?
;assignvariableop_36_chess_model_batch_normalization_8_gamma>
:assignvariableop_37_chess_model_batch_normalization_8_betaE
Aassignvariableop_38_chess_model_batch_normalization_8_moving_meanI
Eassignvariableop_39_chess_model_batch_normalization_8_moving_variance1
-assignvariableop_40_chess_model_conv2d_kernel/
+assignvariableop_41_chess_model_conv2d_bias3
/assignvariableop_42_chess_model_conv2d_1_kernel1
-assignvariableop_43_chess_model_conv2d_1_bias3
/assignvariableop_44_chess_model_conv2d_2_kernel1
-assignvariableop_45_chess_model_conv2d_2_bias3
/assignvariableop_46_chess_model_conv2d_3_kernel1
-assignvariableop_47_chess_model_conv2d_3_bias3
/assignvariableop_48_chess_model_conv2d_4_kernel1
-assignvariableop_49_chess_model_conv2d_4_bias3
/assignvariableop_50_chess_model_conv2d_5_kernel1
-assignvariableop_51_chess_model_conv2d_5_bias3
/assignvariableop_52_chess_model_conv2d_6_kernel1
-assignvariableop_53_chess_model_conv2d_6_bias3
/assignvariableop_54_chess_model_conv2d_7_kernel1
-assignvariableop_55_chess_model_conv2d_7_bias3
/assignvariableop_56_chess_model_conv2d_8_kernel1
-assignvariableop_57_chess_model_conv2d_8_bias2
.assignvariableop_58_chess_model_dense_2_kernel0
,assignvariableop_59_chess_model_dense_2_bias!
assignvariableop_60_adam_iter#
assignvariableop_61_adam_beta_1#
assignvariableop_62_adam_beta_2"
assignvariableop_63_adam_decay*
&assignvariableop_64_adam_learning_rate
assignvariableop_65_total
assignvariableop_66_count
assignvariableop_67_total_1
assignvariableop_68_count_17
3assignvariableop_69_adam_chess_model_dense_kernel_m5
1assignvariableop_70_adam_chess_model_dense_bias_m9
5assignvariableop_71_adam_chess_model_dense_1_kernel_m7
3assignvariableop_72_adam_chess_model_dense_1_bias_mD
@assignvariableop_73_adam_chess_model_batch_normalization_gamma_mC
?assignvariableop_74_adam_chess_model_batch_normalization_beta_mF
Bassignvariableop_75_adam_chess_model_batch_normalization_1_gamma_mE
Aassignvariableop_76_adam_chess_model_batch_normalization_1_beta_mF
Bassignvariableop_77_adam_chess_model_batch_normalization_2_gamma_mE
Aassignvariableop_78_adam_chess_model_batch_normalization_2_beta_mF
Bassignvariableop_79_adam_chess_model_batch_normalization_3_gamma_mE
Aassignvariableop_80_adam_chess_model_batch_normalization_3_beta_mF
Bassignvariableop_81_adam_chess_model_batch_normalization_4_gamma_mE
Aassignvariableop_82_adam_chess_model_batch_normalization_4_beta_mF
Bassignvariableop_83_adam_chess_model_batch_normalization_5_gamma_mE
Aassignvariableop_84_adam_chess_model_batch_normalization_5_beta_mF
Bassignvariableop_85_adam_chess_model_batch_normalization_6_gamma_mE
Aassignvariableop_86_adam_chess_model_batch_normalization_6_beta_mF
Bassignvariableop_87_adam_chess_model_batch_normalization_7_gamma_mE
Aassignvariableop_88_adam_chess_model_batch_normalization_7_beta_mF
Bassignvariableop_89_adam_chess_model_batch_normalization_8_gamma_mE
Aassignvariableop_90_adam_chess_model_batch_normalization_8_beta_m8
4assignvariableop_91_adam_chess_model_conv2d_kernel_m6
2assignvariableop_92_adam_chess_model_conv2d_bias_m:
6assignvariableop_93_adam_chess_model_conv2d_1_kernel_m8
4assignvariableop_94_adam_chess_model_conv2d_1_bias_m:
6assignvariableop_95_adam_chess_model_conv2d_2_kernel_m8
4assignvariableop_96_adam_chess_model_conv2d_2_bias_m:
6assignvariableop_97_adam_chess_model_conv2d_3_kernel_m8
4assignvariableop_98_adam_chess_model_conv2d_3_bias_m:
6assignvariableop_99_adam_chess_model_conv2d_4_kernel_m9
5assignvariableop_100_adam_chess_model_conv2d_4_bias_m;
7assignvariableop_101_adam_chess_model_conv2d_5_kernel_m9
5assignvariableop_102_adam_chess_model_conv2d_5_bias_m;
7assignvariableop_103_adam_chess_model_conv2d_6_kernel_m9
5assignvariableop_104_adam_chess_model_conv2d_6_bias_m;
7assignvariableop_105_adam_chess_model_conv2d_7_kernel_m9
5assignvariableop_106_adam_chess_model_conv2d_7_bias_m;
7assignvariableop_107_adam_chess_model_conv2d_8_kernel_m9
5assignvariableop_108_adam_chess_model_conv2d_8_bias_m:
6assignvariableop_109_adam_chess_model_dense_2_kernel_m8
4assignvariableop_110_adam_chess_model_dense_2_bias_m8
4assignvariableop_111_adam_chess_model_dense_kernel_v6
2assignvariableop_112_adam_chess_model_dense_bias_v:
6assignvariableop_113_adam_chess_model_dense_1_kernel_v8
4assignvariableop_114_adam_chess_model_dense_1_bias_vE
Aassignvariableop_115_adam_chess_model_batch_normalization_gamma_vD
@assignvariableop_116_adam_chess_model_batch_normalization_beta_vG
Cassignvariableop_117_adam_chess_model_batch_normalization_1_gamma_vF
Bassignvariableop_118_adam_chess_model_batch_normalization_1_beta_vG
Cassignvariableop_119_adam_chess_model_batch_normalization_2_gamma_vF
Bassignvariableop_120_adam_chess_model_batch_normalization_2_beta_vG
Cassignvariableop_121_adam_chess_model_batch_normalization_3_gamma_vF
Bassignvariableop_122_adam_chess_model_batch_normalization_3_beta_vG
Cassignvariableop_123_adam_chess_model_batch_normalization_4_gamma_vF
Bassignvariableop_124_adam_chess_model_batch_normalization_4_beta_vG
Cassignvariableop_125_adam_chess_model_batch_normalization_5_gamma_vF
Bassignvariableop_126_adam_chess_model_batch_normalization_5_beta_vG
Cassignvariableop_127_adam_chess_model_batch_normalization_6_gamma_vF
Bassignvariableop_128_adam_chess_model_batch_normalization_6_beta_vG
Cassignvariableop_129_adam_chess_model_batch_normalization_7_gamma_vF
Bassignvariableop_130_adam_chess_model_batch_normalization_7_beta_vG
Cassignvariableop_131_adam_chess_model_batch_normalization_8_gamma_vF
Bassignvariableop_132_adam_chess_model_batch_normalization_8_beta_v9
5assignvariableop_133_adam_chess_model_conv2d_kernel_v7
3assignvariableop_134_adam_chess_model_conv2d_bias_v;
7assignvariableop_135_adam_chess_model_conv2d_1_kernel_v9
5assignvariableop_136_adam_chess_model_conv2d_1_bias_v;
7assignvariableop_137_adam_chess_model_conv2d_2_kernel_v9
5assignvariableop_138_adam_chess_model_conv2d_2_bias_v;
7assignvariableop_139_adam_chess_model_conv2d_3_kernel_v9
5assignvariableop_140_adam_chess_model_conv2d_3_bias_v;
7assignvariableop_141_adam_chess_model_conv2d_4_kernel_v9
5assignvariableop_142_adam_chess_model_conv2d_4_bias_v;
7assignvariableop_143_adam_chess_model_conv2d_5_kernel_v9
5assignvariableop_144_adam_chess_model_conv2d_5_bias_v;
7assignvariableop_145_adam_chess_model_conv2d_6_kernel_v9
5assignvariableop_146_adam_chess_model_conv2d_6_bias_v;
7assignvariableop_147_adam_chess_model_conv2d_7_kernel_v9
5assignvariableop_148_adam_chess_model_conv2d_7_bias_v;
7assignvariableop_149_adam_chess_model_conv2d_8_kernel_v9
5assignvariableop_150_adam_chess_model_conv2d_8_bias_v:
6assignvariableop_151_adam_chess_model_dense_2_kernel_v8
4assignvariableop_152_adam_chess_model_dense_2_bias_v
identity_154??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_139?AssignVariableOp_14?AssignVariableOp_140?AssignVariableOp_141?AssignVariableOp_142?AssignVariableOp_143?AssignVariableOp_144?AssignVariableOp_145?AssignVariableOp_146?AssignVariableOp_147?AssignVariableOp_148?AssignVariableOp_149?AssignVariableOp_15?AssignVariableOp_150?AssignVariableOp_151?AssignVariableOp_152?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?K
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?J
value?JB?J?B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB+normalize1/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize1/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize2/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize2/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize3/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize3/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize4/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize4/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize5/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize5/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize6/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize6/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize7/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize7/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize8/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize8/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+normalize9/gamma/.ATTRIBUTES/VARIABLE_VALUEB*normalize9/beta/.ATTRIBUTES/VARIABLE_VALUEB1normalize9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5normalize9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB.convolution1/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution1/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution2/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution2/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution3/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution3/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution4/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution4/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution5/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution5/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution6/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution6/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution7/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution7/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution8/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution8/bias/.ATTRIBUTES/VARIABLE_VALUEB.convolution9/kernel/.ATTRIBUTES/VARIABLE_VALUEB,convolution9/bias/.ATTRIBUTES/VARIABLE_VALUEB+valueHead/kernel/.ATTRIBUTES/VARIABLE_VALUEB)valueHead/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGnormalize9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFnormalize9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJconvolution9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconvolution9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGvalueHead/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEvalueHead/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGnormalize9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFnormalize9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJconvolution9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconvolution9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGvalueHead/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEvalueHead/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_chess_model_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_chess_model_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_chess_model_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_chess_model_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp8assignvariableop_4_chess_model_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp7assignvariableop_5_chess_model_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp>assignvariableop_6_chess_model_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpBassignvariableop_7_chess_model_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp:assignvariableop_8_chess_model_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp9assignvariableop_9_chess_model_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpAassignvariableop_10_chess_model_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpEassignvariableop_11_chess_model_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp;assignvariableop_12_chess_model_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp:assignvariableop_13_chess_model_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpAassignvariableop_14_chess_model_batch_normalization_2_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpEassignvariableop_15_chess_model_batch_normalization_2_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp;assignvariableop_16_chess_model_batch_normalization_3_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_chess_model_batch_normalization_3_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpAassignvariableop_18_chess_model_batch_normalization_3_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpEassignvariableop_19_chess_model_batch_normalization_3_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp;assignvariableop_20_chess_model_batch_normalization_4_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp:assignvariableop_21_chess_model_batch_normalization_4_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpAassignvariableop_22_chess_model_batch_normalization_4_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpEassignvariableop_23_chess_model_batch_normalization_4_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp;assignvariableop_24_chess_model_batch_normalization_5_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_chess_model_batch_normalization_5_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpAassignvariableop_26_chess_model_batch_normalization_5_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpEassignvariableop_27_chess_model_batch_normalization_5_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp;assignvariableop_28_chess_model_batch_normalization_6_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_chess_model_batch_normalization_6_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpAassignvariableop_30_chess_model_batch_normalization_6_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpEassignvariableop_31_chess_model_batch_normalization_6_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp;assignvariableop_32_chess_model_batch_normalization_7_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp:assignvariableop_33_chess_model_batch_normalization_7_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpAassignvariableop_34_chess_model_batch_normalization_7_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpEassignvariableop_35_chess_model_batch_normalization_7_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp;assignvariableop_36_chess_model_batch_normalization_8_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp:assignvariableop_37_chess_model_batch_normalization_8_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpAassignvariableop_38_chess_model_batch_normalization_8_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpEassignvariableop_39_chess_model_batch_normalization_8_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp-assignvariableop_40_chess_model_conv2d_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_chess_model_conv2d_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp/assignvariableop_42_chess_model_conv2d_1_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp-assignvariableop_43_chess_model_conv2d_1_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp/assignvariableop_44_chess_model_conv2d_2_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp-assignvariableop_45_chess_model_conv2d_2_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp/assignvariableop_46_chess_model_conv2d_3_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_chess_model_conv2d_3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp/assignvariableop_48_chess_model_conv2d_4_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp-assignvariableop_49_chess_model_conv2d_4_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp/assignvariableop_50_chess_model_conv2d_5_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp-assignvariableop_51_chess_model_conv2d_5_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp/assignvariableop_52_chess_model_conv2d_6_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp-assignvariableop_53_chess_model_conv2d_6_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp/assignvariableop_54_chess_model_conv2d_7_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp-assignvariableop_55_chess_model_conv2d_7_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp/assignvariableop_56_chess_model_conv2d_8_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp-assignvariableop_57_chess_model_conv2d_8_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp.assignvariableop_58_chess_model_dense_2_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp,assignvariableop_59_chess_model_dense_2_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpassignvariableop_60_adam_iterIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpassignvariableop_61_adam_beta_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpassignvariableop_62_adam_beta_2Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpassignvariableop_63_adam_decayIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp&assignvariableop_64_adam_learning_rateIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpassignvariableop_65_totalIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpassignvariableop_66_countIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpassignvariableop_67_total_1Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpassignvariableop_68_count_1Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp3assignvariableop_69_adam_chess_model_dense_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp1assignvariableop_70_adam_chess_model_dense_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp5assignvariableop_71_adam_chess_model_dense_1_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp3assignvariableop_72_adam_chess_model_dense_1_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp@assignvariableop_73_adam_chess_model_batch_normalization_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp?assignvariableop_74_adam_chess_model_batch_normalization_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpBassignvariableop_75_adam_chess_model_batch_normalization_1_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOpAassignvariableop_76_adam_chess_model_batch_normalization_1_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOpBassignvariableop_77_adam_chess_model_batch_normalization_2_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOpAassignvariableop_78_adam_chess_model_batch_normalization_2_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOpBassignvariableop_79_adam_chess_model_batch_normalization_3_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOpAassignvariableop_80_adam_chess_model_batch_normalization_3_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOpBassignvariableop_81_adam_chess_model_batch_normalization_4_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOpAassignvariableop_82_adam_chess_model_batch_normalization_4_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOpBassignvariableop_83_adam_chess_model_batch_normalization_5_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOpAassignvariableop_84_adam_chess_model_batch_normalization_5_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOpBassignvariableop_85_adam_chess_model_batch_normalization_6_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOpAassignvariableop_86_adam_chess_model_batch_normalization_6_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOpBassignvariableop_87_adam_chess_model_batch_normalization_7_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOpAassignvariableop_88_adam_chess_model_batch_normalization_7_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOpBassignvariableop_89_adam_chess_model_batch_normalization_8_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOpAassignvariableop_90_adam_chess_model_batch_normalization_8_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp4assignvariableop_91_adam_chess_model_conv2d_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp2assignvariableop_92_adam_chess_model_conv2d_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_chess_model_conv2d_1_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp4assignvariableop_94_adam_chess_model_conv2d_1_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp6assignvariableop_95_adam_chess_model_conv2d_2_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp4assignvariableop_96_adam_chess_model_conv2d_2_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp6assignvariableop_97_adam_chess_model_conv2d_3_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp4assignvariableop_98_adam_chess_model_conv2d_3_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp6assignvariableop_99_adam_chess_model_conv2d_4_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp5assignvariableop_100_adam_chess_model_conv2d_4_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_chess_model_conv2d_5_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp5assignvariableop_102_adam_chess_model_conv2d_5_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp7assignvariableop_103_adam_chess_model_conv2d_6_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp5assignvariableop_104_adam_chess_model_conv2d_6_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp7assignvariableop_105_adam_chess_model_conv2d_7_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp5assignvariableop_106_adam_chess_model_conv2d_7_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp7assignvariableop_107_adam_chess_model_conv2d_8_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp5assignvariableop_108_adam_chess_model_conv2d_8_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp6assignvariableop_109_adam_chess_model_dense_2_kernel_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp4assignvariableop_110_adam_chess_model_dense_2_bias_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp4assignvariableop_111_adam_chess_model_dense_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp2assignvariableop_112_adam_chess_model_dense_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp6assignvariableop_113_adam_chess_model_dense_1_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp4assignvariableop_114_adam_chess_model_dense_1_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOpAassignvariableop_115_adam_chess_model_batch_normalization_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp@assignvariableop_116_adam_chess_model_batch_normalization_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOpCassignvariableop_117_adam_chess_model_batch_normalization_1_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOpBassignvariableop_118_adam_chess_model_batch_normalization_1_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOpCassignvariableop_119_adam_chess_model_batch_normalization_2_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOpBassignvariableop_120_adam_chess_model_batch_normalization_2_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOpCassignvariableop_121_adam_chess_model_batch_normalization_3_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOpBassignvariableop_122_adam_chess_model_batch_normalization_3_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOpCassignvariableop_123_adam_chess_model_batch_normalization_4_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOpBassignvariableop_124_adam_chess_model_batch_normalization_4_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOpCassignvariableop_125_adam_chess_model_batch_normalization_5_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOpBassignvariableop_126_adam_chess_model_batch_normalization_5_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOpCassignvariableop_127_adam_chess_model_batch_normalization_6_gamma_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOpBassignvariableop_128_adam_chess_model_batch_normalization_6_beta_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOpCassignvariableop_129_adam_chess_model_batch_normalization_7_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOpBassignvariableop_130_adam_chess_model_batch_normalization_7_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOpCassignvariableop_131_adam_chess_model_batch_normalization_8_gamma_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOpBassignvariableop_132_adam_chess_model_batch_normalization_8_beta_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOp5assignvariableop_133_adam_chess_model_conv2d_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOp3assignvariableop_134_adam_chess_model_conv2d_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135?
AssignVariableOp_135AssignVariableOp7assignvariableop_135_adam_chess_model_conv2d_1_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136?
AssignVariableOp_136AssignVariableOp5assignvariableop_136_adam_chess_model_conv2d_1_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137?
AssignVariableOp_137AssignVariableOp7assignvariableop_137_adam_chess_model_conv2d_2_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138?
AssignVariableOp_138AssignVariableOp5assignvariableop_138_adam_chess_model_conv2d_2_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139?
AssignVariableOp_139AssignVariableOp7assignvariableop_139_adam_chess_model_conv2d_3_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140?
AssignVariableOp_140AssignVariableOp5assignvariableop_140_adam_chess_model_conv2d_3_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141?
AssignVariableOp_141AssignVariableOp7assignvariableop_141_adam_chess_model_conv2d_4_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142?
AssignVariableOp_142AssignVariableOp5assignvariableop_142_adam_chess_model_conv2d_4_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143?
AssignVariableOp_143AssignVariableOp7assignvariableop_143_adam_chess_model_conv2d_5_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144?
AssignVariableOp_144AssignVariableOp5assignvariableop_144_adam_chess_model_conv2d_5_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145?
AssignVariableOp_145AssignVariableOp7assignvariableop_145_adam_chess_model_conv2d_6_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146?
AssignVariableOp_146AssignVariableOp5assignvariableop_146_adam_chess_model_conv2d_6_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147?
AssignVariableOp_147AssignVariableOp7assignvariableop_147_adam_chess_model_conv2d_7_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148?
AssignVariableOp_148AssignVariableOp5assignvariableop_148_adam_chess_model_conv2d_7_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_148q
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:2
Identity_149?
AssignVariableOp_149AssignVariableOp7assignvariableop_149_adam_chess_model_conv2d_8_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149q
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:2
Identity_150?
AssignVariableOp_150AssignVariableOp5assignvariableop_150_adam_chess_model_conv2d_8_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_150q
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:2
Identity_151?
AssignVariableOp_151AssignVariableOp6assignvariableop_151_adam_chess_model_dense_2_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_151q
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:2
Identity_152?
AssignVariableOp_152AssignVariableOp4assignvariableop_152_adam_chess_model_dense_2_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_153Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_153?
Identity_154IdentityIdentity_153:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_154"%
identity_154Identity_154:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_179577

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_179751

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_3_layer_call_fn_179526

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1787092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_3_layer_call_fn_179539

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1787402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_179513

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_6_1799965
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
*__inference_chess_model_layer_call_fn_4085

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*L
_read_only_resource_inputs.
,*	
 !"%&'(+,-.1234789:;<*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_chess_model_layer_call_and_return_conditional_losses_39552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_178532

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_3_layer_call_and_return_conditional_losses_687

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_179156

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_3_1799635
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
N__inference_batch_normalization_5_layer_call_and_return_conditional_losses_522

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_dense_layer_call_and_return_conditional_losses_2656

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?
+chess_model/dense/kernel/Regularizer/SquareSquareBchess_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2-
+chess_model/dense/kernel/Regularizer/Square?
*chess_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*chess_model/dense/kernel/Regularizer/Const?
(chess_model/dense/kernel/Regularizer/SumSum/chess_model/dense/kernel/Regularizer/Square:y:03chess_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/Sum?
*chess_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*chess_model/dense/kernel/Regularizer/mul/x?
(chess_model/dense/kernel/Regularizer/mulMul3chess_model/dense/kernel/Regularizer/mul/x:output:01chess_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp;^chess_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2x
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?9
E__inference_chess_model_layer_call_and_return_conditional_losses_3303

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?$batch_normalization_7/AssignNewValue?&batch_normalization_7/AssignNewValue_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
re_lu/Relu_1Relu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dre_lu/Relu_1:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
add/addAddV2re_lu/Relu:activations:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2	
add/addl
re_lu/Relu_2Reluadd/add:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_2?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dre_lu/Relu_2:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
re_lu/Relu_3Relu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_3?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dre_lu/Relu_3:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
re_lu/Relu_4Relu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_4?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dre_lu/Relu_4:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
	add/add_1AddV2re_lu/Relu_3:activations:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
	add/add_1n
re_lu/Relu_5Reluadd/add_1:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_5?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dre_lu/Relu_5:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1?
re_lu/Relu_6Relu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_6?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dre_lu/Relu_6:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_7/FusedBatchNormV3?
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue?
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1?
re_lu/Relu_7Relu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_7?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dre_lu/Relu_7:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_8/BiasAdd?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_8/FusedBatchNormV3?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1?
	add/add_2AddV2re_lu/Relu_6:activations:0*batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
	add/add_2n
re_lu/Relu_8Reluadd/add_2:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_8o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapere_lu/Relu_8:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Tanh?
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?
+chess_model/dense/kernel/Regularizer/SquareSquareBchess_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2-
+chess_model/dense/kernel/Regularizer/Square?
*chess_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*chess_model/dense/kernel/Regularizer/Const?
(chess_model/dense/kernel/Regularizer/SumSum/chess_model/dense/kernel/Regularizer/Square:y:03chess_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/Sum?
*chess_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*chess_model/dense/kernel/Regularizer/mul/x?
(chess_model/dense/kernel/Regularizer/mulMul3chess_model/dense/kernel/Regularizer/mul/x:output:01chess_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/mul?
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02>
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_1/kernel/Regularizer/SquareSquareDchess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2/
-chess_model/dense_1/kernel/Regularizer/Square?
,chess_model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_1/kernel/Regularizer/Const?
*chess_model/dense_1/kernel/Regularizer/SumSum1chess_model/dense_1/kernel/Regularizer/Square:y:05chess_model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/Sum?
,chess_model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_1/kernel/Regularizer/mul/x?
*chess_model/dense_1/kernel/Regularizer/mulMul5chess_model/dense_1/kernel/Regularizer/mul/x:output:03chess_model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/mul?
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02=
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
,chess_model/conv2d/kernel/Regularizer/SquareSquareCchess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:?2.
,chess_model/conv2d/kernel/Regularizer/Square?
+chess_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+chess_model/conv2d/kernel/Regularizer/Const?
)chess_model/conv2d/kernel/Regularizer/SumSum0chess_model/conv2d/kernel/Regularizer/Square:y:04chess_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/Sum?
+chess_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+chess_model/conv2d/kernel/Regularizer/mul/x?
)chess_model/conv2d/kernel/Regularizer/mulMul4chess_model/conv2d/kernel/Regularizer/mul/x:output:02chess_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/mul?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_1/kernel/Regularizer/SquareSquareEchess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_1/kernel/Regularizer/Square?
-chess_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_1/kernel/Regularizer/Const?
+chess_model/conv2d_1/kernel/Regularizer/SumSum2chess_model/conv2d_1/kernel/Regularizer/Square:y:06chess_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/Sum?
-chess_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_1/kernel/Regularizer/mul/x?
+chess_model/conv2d_1/kernel/Regularizer/mulMul6chess_model/conv2d_1/kernel/Regularizer/mul/x:output:04chess_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/mul?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_2/kernel/Regularizer/SquareSquareEchess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_2/kernel/Regularizer/Square?
-chess_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_2/kernel/Regularizer/Const?
+chess_model/conv2d_2/kernel/Regularizer/SumSum2chess_model/conv2d_2/kernel/Regularizer/Square:y:06chess_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/Sum?
-chess_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_2/kernel/Regularizer/mul/x?
+chess_model/conv2d_2/kernel/Regularizer/mulMul6chess_model/conv2d_2/kernel/Regularizer/mul/x:output:04chess_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/mul?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_3/kernel/Regularizer/SquareSquareEchess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_3/kernel/Regularizer/Square?
-chess_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_3/kernel/Regularizer/Const?
+chess_model/conv2d_3/kernel/Regularizer/SumSum2chess_model/conv2d_3/kernel/Regularizer/Square:y:06chess_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/Sum?
-chess_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_3/kernel/Regularizer/mul/x?
+chess_model/conv2d_3/kernel/Regularizer/mulMul6chess_model/conv2d_3/kernel/Regularizer/mul/x:output:04chess_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/mul?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_4/kernel/Regularizer/SquareSquareEchess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_4/kernel/Regularizer/Square?
-chess_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_4/kernel/Regularizer/Const?
+chess_model/conv2d_4/kernel/Regularizer/SumSum2chess_model/conv2d_4/kernel/Regularizer/Square:y:06chess_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/Sum?
-chess_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_4/kernel/Regularizer/mul/x?
+chess_model/conv2d_4/kernel/Regularizer/mulMul6chess_model/conv2d_4/kernel/Regularizer/mul/x:output:04chess_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/mul?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_5/kernel/Regularizer/SquareSquareEchess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_5/kernel/Regularizer/Square?
-chess_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_5/kernel/Regularizer/Const?
+chess_model/conv2d_5/kernel/Regularizer/SumSum2chess_model/conv2d_5/kernel/Regularizer/Square:y:06chess_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/Sum?
-chess_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_5/kernel/Regularizer/mul/x?
+chess_model/conv2d_5/kernel/Regularizer/mulMul6chess_model/conv2d_5/kernel/Regularizer/mul/x:output:04chess_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/mul?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_6/kernel/Regularizer/SquareSquareEchess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_6/kernel/Regularizer/Square?
-chess_model/conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_6/kernel/Regularizer/Const?
+chess_model/conv2d_6/kernel/Regularizer/SumSum2chess_model/conv2d_6/kernel/Regularizer/Square:y:06chess_model/conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/Sum?
-chess_model/conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_6/kernel/Regularizer/mul/x?
+chess_model/conv2d_6/kernel/Regularizer/mulMul6chess_model/conv2d_6/kernel/Regularizer/mul/x:output:04chess_model/conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/mul?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_7/kernel/Regularizer/SquareSquareEchess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_7/kernel/Regularizer/Square?
-chess_model/conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_7/kernel/Regularizer/Const?
+chess_model/conv2d_7/kernel/Regularizer/SumSum2chess_model/conv2d_7/kernel/Regularizer/Square:y:06chess_model/conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/Sum?
-chess_model/conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_7/kernel/Regularizer/mul/x?
+chess_model/conv2d_7/kernel/Regularizer/mulMul6chess_model/conv2d_7/kernel/Regularizer/mul/x:output:04chess_model/conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/mul?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_8/kernel/Regularizer/SquareSquareEchess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_8/kernel/Regularizer/Square?
-chess_model/conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_8/kernel/Regularizer/Const?
+chess_model/conv2d_8/kernel/Regularizer/SumSum2chess_model/conv2d_8/kernel/Regularizer/Square:y:06chess_model/conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/Sum?
-chess_model/conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_8/kernel/Regularizer/mul/x?
+chess_model/conv2d_8/kernel/Regularizer/mulMul6chess_model/conv2d_8/kernel/Regularizer/mul/x:output:04chess_model/conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/mul?
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02>
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_2/kernel/Regularizer/SquareSquareDchess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2/
-chess_model/dense_2/kernel/Regularizer/Square?
,chess_model/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_2/kernel/Regularizer/Const?
*chess_model/dense_2/kernel/Regularizer/SumSum1chess_model/dense_2/kernel/Regularizer/Square:y:05chess_model/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/Sum?
,chess_model/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_2/kernel/Regularizer/mul/x?
*chess_model/dense_2/kernel/Regularizer/mulMul5chess_model/dense_2/kernel/Regularizer/mul/x:output:03chess_model/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/mul? 
IdentityIdentitydense_2/Tanh:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1<^chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp;^chess_model/dense/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12z
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp2x
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ʊ
?3
E__inference_chess_model_layer_call_and_return_conditional_losses_1080
input_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
re_lu/Relu_1Relu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dre_lu/Relu_1:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
add/addAddV2re_lu/Relu:activations:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2	
add/addl
re_lu/Relu_2Reluadd/add:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_2?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dre_lu/Relu_2:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
re_lu/Relu_3Relu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_3?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dre_lu/Relu_3:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
re_lu/Relu_4Relu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_4?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dre_lu/Relu_4:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
	add/add_1AddV2re_lu/Relu_3:activations:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
	add/add_1n
re_lu/Relu_5Reluadd/add_1:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_5?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dre_lu/Relu_5:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
re_lu/Relu_6Relu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_6?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dre_lu/Relu_6:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
re_lu/Relu_7Relu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_7?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dre_lu/Relu_7:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_8/BiasAdd?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
	add/add_2AddV2re_lu/Relu_6:activations:0*batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
	add/add_2n
re_lu/Relu_8Reluadd/add_2:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_8o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapere_lu/Relu_8:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Tanh?
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?
+chess_model/dense/kernel/Regularizer/SquareSquareBchess_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2-
+chess_model/dense/kernel/Regularizer/Square?
*chess_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*chess_model/dense/kernel/Regularizer/Const?
(chess_model/dense/kernel/Regularizer/SumSum/chess_model/dense/kernel/Regularizer/Square:y:03chess_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/Sum?
*chess_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*chess_model/dense/kernel/Regularizer/mul/x?
(chess_model/dense/kernel/Regularizer/mulMul3chess_model/dense/kernel/Regularizer/mul/x:output:01chess_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/mul?
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02>
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_1/kernel/Regularizer/SquareSquareDchess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2/
-chess_model/dense_1/kernel/Regularizer/Square?
,chess_model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_1/kernel/Regularizer/Const?
*chess_model/dense_1/kernel/Regularizer/SumSum1chess_model/dense_1/kernel/Regularizer/Square:y:05chess_model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/Sum?
,chess_model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_1/kernel/Regularizer/mul/x?
*chess_model/dense_1/kernel/Regularizer/mulMul5chess_model/dense_1/kernel/Regularizer/mul/x:output:03chess_model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/mul?
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02=
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
,chess_model/conv2d/kernel/Regularizer/SquareSquareCchess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:?2.
,chess_model/conv2d/kernel/Regularizer/Square?
+chess_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+chess_model/conv2d/kernel/Regularizer/Const?
)chess_model/conv2d/kernel/Regularizer/SumSum0chess_model/conv2d/kernel/Regularizer/Square:y:04chess_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/Sum?
+chess_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+chess_model/conv2d/kernel/Regularizer/mul/x?
)chess_model/conv2d/kernel/Regularizer/mulMul4chess_model/conv2d/kernel/Regularizer/mul/x:output:02chess_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/mul?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_1/kernel/Regularizer/SquareSquareEchess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_1/kernel/Regularizer/Square?
-chess_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_1/kernel/Regularizer/Const?
+chess_model/conv2d_1/kernel/Regularizer/SumSum2chess_model/conv2d_1/kernel/Regularizer/Square:y:06chess_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/Sum?
-chess_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_1/kernel/Regularizer/mul/x?
+chess_model/conv2d_1/kernel/Regularizer/mulMul6chess_model/conv2d_1/kernel/Regularizer/mul/x:output:04chess_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/mul?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_2/kernel/Regularizer/SquareSquareEchess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_2/kernel/Regularizer/Square?
-chess_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_2/kernel/Regularizer/Const?
+chess_model/conv2d_2/kernel/Regularizer/SumSum2chess_model/conv2d_2/kernel/Regularizer/Square:y:06chess_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/Sum?
-chess_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_2/kernel/Regularizer/mul/x?
+chess_model/conv2d_2/kernel/Regularizer/mulMul6chess_model/conv2d_2/kernel/Regularizer/mul/x:output:04chess_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/mul?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_3/kernel/Regularizer/SquareSquareEchess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_3/kernel/Regularizer/Square?
-chess_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_3/kernel/Regularizer/Const?
+chess_model/conv2d_3/kernel/Regularizer/SumSum2chess_model/conv2d_3/kernel/Regularizer/Square:y:06chess_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/Sum?
-chess_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_3/kernel/Regularizer/mul/x?
+chess_model/conv2d_3/kernel/Regularizer/mulMul6chess_model/conv2d_3/kernel/Regularizer/mul/x:output:04chess_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/mul?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_4/kernel/Regularizer/SquareSquareEchess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_4/kernel/Regularizer/Square?
-chess_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_4/kernel/Regularizer/Const?
+chess_model/conv2d_4/kernel/Regularizer/SumSum2chess_model/conv2d_4/kernel/Regularizer/Square:y:06chess_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/Sum?
-chess_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_4/kernel/Regularizer/mul/x?
+chess_model/conv2d_4/kernel/Regularizer/mulMul6chess_model/conv2d_4/kernel/Regularizer/mul/x:output:04chess_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/mul?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_5/kernel/Regularizer/SquareSquareEchess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_5/kernel/Regularizer/Square?
-chess_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_5/kernel/Regularizer/Const?
+chess_model/conv2d_5/kernel/Regularizer/SumSum2chess_model/conv2d_5/kernel/Regularizer/Square:y:06chess_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/Sum?
-chess_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_5/kernel/Regularizer/mul/x?
+chess_model/conv2d_5/kernel/Regularizer/mulMul6chess_model/conv2d_5/kernel/Regularizer/mul/x:output:04chess_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/mul?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_6/kernel/Regularizer/SquareSquareEchess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_6/kernel/Regularizer/Square?
-chess_model/conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_6/kernel/Regularizer/Const?
+chess_model/conv2d_6/kernel/Regularizer/SumSum2chess_model/conv2d_6/kernel/Regularizer/Square:y:06chess_model/conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/Sum?
-chess_model/conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_6/kernel/Regularizer/mul/x?
+chess_model/conv2d_6/kernel/Regularizer/mulMul6chess_model/conv2d_6/kernel/Regularizer/mul/x:output:04chess_model/conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/mul?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_7/kernel/Regularizer/SquareSquareEchess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_7/kernel/Regularizer/Square?
-chess_model/conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_7/kernel/Regularizer/Const?
+chess_model/conv2d_7/kernel/Regularizer/SumSum2chess_model/conv2d_7/kernel/Regularizer/Square:y:06chess_model/conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/Sum?
-chess_model/conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_7/kernel/Regularizer/mul/x?
+chess_model/conv2d_7/kernel/Regularizer/mulMul6chess_model/conv2d_7/kernel/Regularizer/mul/x:output:04chess_model/conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/mul?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_8/kernel/Regularizer/SquareSquareEchess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_8/kernel/Regularizer/Square?
-chess_model/conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_8/kernel/Regularizer/Const?
+chess_model/conv2d_8/kernel/Regularizer/SumSum2chess_model/conv2d_8/kernel/Regularizer/Square:y:06chess_model/conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/Sum?
-chess_model/conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_8/kernel/Regularizer/mul/x?
+chess_model/conv2d_8/kernel/Regularizer/mulMul6chess_model/conv2d_8/kernel/Regularizer/mul/x:output:04chess_model/conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/mul?
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02>
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_2/kernel/Regularizer/SquareSquareDchess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2/
-chess_model/dense_2/kernel/Regularizer/Square?
,chess_model/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_2/kernel/Regularizer/Const?
*chess_model/dense_2/kernel/Regularizer/SumSum1chess_model/dense_2/kernel/Regularizer/Square:y:05chess_model/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/Sum?
,chess_model/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_2/kernel/Regularizer/mul/x?
*chess_model/dense_2/kernel/Regularizer/mulMul5chess_model/dense_2/kernel/Regularizer/mul/x:output:03chess_model/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/mul?
IdentityIdentitydense_2/Tanh:y:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1<^chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp;^chess_model/dense/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12z
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp2x
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_179385

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
Z
>__inference_re_lu_layer_call_and_return_conditional_losses_692

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_178917

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_chess_model_layer_call_fn_4394
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_chess_model_layer_call_and_return_conditional_losses_43292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
)__inference_restored_function_body_178005

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_chess_model_layer_call_and_return_conditional_losses_23512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_8_layer_call_fn_179846

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1792292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_3425

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_5/kernel/Regularizer/SquareSquareEchess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_5/kernel/Regularizer/Square?
-chess_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_5/kernel/Regularizer/Const?
+chess_model/conv2d_5/kernel/Regularizer/SumSum2chess_model/conv2d_5/kernel/Regularizer/Square:y:06chess_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/Sum?
-chess_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_5/kernel/Regularizer/mul/x?
+chess_model/conv2d_5/kernel/Regularizer/mulMul6chess_model/conv2d_5/kernel/Regularizer/mul/x:output:04chess_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_6_layer_call_fn_179731

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1790522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_1_layer_call_fn_179398

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1785012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_178740

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2711

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_8_1800185
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?

?
__inference_loss_fn_0_1799305
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
6__inference_batch_normalization_1_layer_call_fn_179411

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1785322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_5_layer_call_fn_179667

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1789482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4158

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_178813

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_178501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_179229

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_dense_1_layer_call_and_return_conditional_losses_3320

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02>
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_1/kernel/Regularizer/SquareSquareDchess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2/
-chess_model/dense_1/kernel/Regularizer/Square?
,chess_model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_1/kernel/Regularizer/Const?
*chess_model/dense_1/kernel/Regularizer/SumSum1chess_model/dense_1/kernel/Regularizer/Square:y:05chess_model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/Sum?
,chess_model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_1/kernel/Regularizer/mul/x?
*chess_model/dense_1/kernel/Regularizer/mulMul5chess_model/dense_1/kernel/Regularizer/mul/x:output:03chess_model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp=^chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2|
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_conv2d_layer_call_and_return_conditional_losses_504

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02=
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
,chess_model/conv2d/kernel/Regularizer/SquareSquareCchess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:?2.
,chess_model/conv2d/kernel/Regularizer/Square?
+chess_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+chess_model/conv2d/kernel/Regularizer/Const?
)chess_model/conv2d/kernel/Regularizer/SumSum0chess_model/conv2d/kernel/Regularizer/Square:y:04chess_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/Sum?
+chess_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+chess_model/conv2d/kernel/Regularizer/mul/x?
)chess_model/conv2d/kernel/Regularizer/mulMul4chess_model/conv2d/kernel/Regularizer/mul/x:output:02chess_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_179769

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_3579

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_2/kernel/Regularizer/SquareSquareEchess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_2/kernel/Regularizer/Square?
-chess_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_2/kernel/Regularizer/Const?
+chess_model/conv2d_2/kernel/Regularizer/SumSum2chess_model/conv2d_2/kernel/Regularizer/Square:y:06chess_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/Sum?
-chess_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_2/kernel/Regularizer/mul/x?
+chess_model/conv2d_2/kernel/Regularizer/mulMul6chess_model/conv2d_2/kernel/Regularizer/mul/x:output:04chess_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3715

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_179559

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_178709

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_2_layer_call_fn_179462

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1786052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_179815

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_179367

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_1_1799415
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2006

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
=__inference_add_layer_call_and_return_conditional_losses_3458

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:??????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_5_layer_call_fn_179654

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1789172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_179052

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_179303

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_7_layer_call_fn_179782

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1791252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_11_1800515
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2592

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_179125

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_178948

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?9
E__inference_chess_model_layer_call_and_return_conditional_losses_3002
input_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?$batch_normalization_7/AssignNewValue?&batch_normalization_7/AssignNewValue_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
re_lu/Relu_1Relu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dre_lu/Relu_1:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
add/addAddV2re_lu/Relu:activations:0*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2	
add/addl
re_lu/Relu_2Reluadd/add:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_2?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dre_lu/Relu_2:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
re_lu/Relu_3Relu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_3?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dre_lu/Relu_3:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
re_lu/Relu_4Relu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_4?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dre_lu/Relu_4:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
	add/add_1AddV2re_lu/Relu_3:activations:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
	add/add_1n
re_lu/Relu_5Reluadd/add_1:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_5?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dre_lu/Relu_5:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1?
re_lu/Relu_6Relu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_6?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dre_lu/Relu_6:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_7/FusedBatchNormV3?
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue?
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1?
re_lu/Relu_7Relu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_7?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dre_lu/Relu_7:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_8/BiasAdd?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_8/FusedBatchNormV3?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1?
	add/add_2AddV2re_lu/Relu_6:activations:0*batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
	add/add_2n
re_lu/Relu_8Reluadd/add_2:z:0*
T0*0
_output_shapes
:??????????2
re_lu/Relu_8o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapere_lu/Relu_8:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Tanh?
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp?
+chess_model/dense/kernel/Regularizer/SquareSquareBchess_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2-
+chess_model/dense/kernel/Regularizer/Square?
*chess_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*chess_model/dense/kernel/Regularizer/Const?
(chess_model/dense/kernel/Regularizer/SumSum/chess_model/dense/kernel/Regularizer/Square:y:03chess_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/Sum?
*chess_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*chess_model/dense/kernel/Regularizer/mul/x?
(chess_model/dense/kernel/Regularizer/mulMul3chess_model/dense/kernel/Regularizer/mul/x:output:01chess_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(chess_model/dense/kernel/Regularizer/mul?
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02>
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_1/kernel/Regularizer/SquareSquareDchess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2/
-chess_model/dense_1/kernel/Regularizer/Square?
,chess_model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_1/kernel/Regularizer/Const?
*chess_model/dense_1/kernel/Regularizer/SumSum1chess_model/dense_1/kernel/Regularizer/Square:y:05chess_model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/Sum?
,chess_model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_1/kernel/Regularizer/mul/x?
*chess_model/dense_1/kernel/Regularizer/mulMul5chess_model/dense_1/kernel/Regularizer/mul/x:output:03chess_model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_1/kernel/Regularizer/mul?
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02=
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
,chess_model/conv2d/kernel/Regularizer/SquareSquareCchess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:?2.
,chess_model/conv2d/kernel/Regularizer/Square?
+chess_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+chess_model/conv2d/kernel/Regularizer/Const?
)chess_model/conv2d/kernel/Regularizer/SumSum0chess_model/conv2d/kernel/Regularizer/Square:y:04chess_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/Sum?
+chess_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+chess_model/conv2d/kernel/Regularizer/mul/x?
)chess_model/conv2d/kernel/Regularizer/mulMul4chess_model/conv2d/kernel/Regularizer/mul/x:output:02chess_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)chess_model/conv2d/kernel/Regularizer/mul?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_1/kernel/Regularizer/SquareSquareEchess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_1/kernel/Regularizer/Square?
-chess_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_1/kernel/Regularizer/Const?
+chess_model/conv2d_1/kernel/Regularizer/SumSum2chess_model/conv2d_1/kernel/Regularizer/Square:y:06chess_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/Sum?
-chess_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_1/kernel/Regularizer/mul/x?
+chess_model/conv2d_1/kernel/Regularizer/mulMul6chess_model/conv2d_1/kernel/Regularizer/mul/x:output:04chess_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/mul?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_2/kernel/Regularizer/SquareSquareEchess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_2/kernel/Regularizer/Square?
-chess_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_2/kernel/Regularizer/Const?
+chess_model/conv2d_2/kernel/Regularizer/SumSum2chess_model/conv2d_2/kernel/Regularizer/Square:y:06chess_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/Sum?
-chess_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_2/kernel/Regularizer/mul/x?
+chess_model/conv2d_2/kernel/Regularizer/mulMul6chess_model/conv2d_2/kernel/Regularizer/mul/x:output:04chess_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_2/kernel/Regularizer/mul?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_3/kernel/Regularizer/SquareSquareEchess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_3/kernel/Regularizer/Square?
-chess_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_3/kernel/Regularizer/Const?
+chess_model/conv2d_3/kernel/Regularizer/SumSum2chess_model/conv2d_3/kernel/Regularizer/Square:y:06chess_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/Sum?
-chess_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_3/kernel/Regularizer/mul/x?
+chess_model/conv2d_3/kernel/Regularizer/mulMul6chess_model/conv2d_3/kernel/Regularizer/mul/x:output:04chess_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_3/kernel/Regularizer/mul?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_4/kernel/Regularizer/SquareSquareEchess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_4/kernel/Regularizer/Square?
-chess_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_4/kernel/Regularizer/Const?
+chess_model/conv2d_4/kernel/Regularizer/SumSum2chess_model/conv2d_4/kernel/Regularizer/Square:y:06chess_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/Sum?
-chess_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_4/kernel/Regularizer/mul/x?
+chess_model/conv2d_4/kernel/Regularizer/mulMul6chess_model/conv2d_4/kernel/Regularizer/mul/x:output:04chess_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_4/kernel/Regularizer/mul?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_5/kernel/Regularizer/SquareSquareEchess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_5/kernel/Regularizer/Square?
-chess_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_5/kernel/Regularizer/Const?
+chess_model/conv2d_5/kernel/Regularizer/SumSum2chess_model/conv2d_5/kernel/Regularizer/Square:y:06chess_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/Sum?
-chess_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_5/kernel/Regularizer/mul/x?
+chess_model/conv2d_5/kernel/Regularizer/mulMul6chess_model/conv2d_5/kernel/Regularizer/mul/x:output:04chess_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_5/kernel/Regularizer/mul?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_6/kernel/Regularizer/SquareSquareEchess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_6/kernel/Regularizer/Square?
-chess_model/conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_6/kernel/Regularizer/Const?
+chess_model/conv2d_6/kernel/Regularizer/SumSum2chess_model/conv2d_6/kernel/Regularizer/Square:y:06chess_model/conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/Sum?
-chess_model/conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_6/kernel/Regularizer/mul/x?
+chess_model/conv2d_6/kernel/Regularizer/mulMul6chess_model/conv2d_6/kernel/Regularizer/mul/x:output:04chess_model/conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/mul?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_7/kernel/Regularizer/SquareSquareEchess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_7/kernel/Regularizer/Square?
-chess_model/conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_7/kernel/Regularizer/Const?
+chess_model/conv2d_7/kernel/Regularizer/SumSum2chess_model/conv2d_7/kernel/Regularizer/Square:y:06chess_model/conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/Sum?
-chess_model/conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_7/kernel/Regularizer/mul/x?
+chess_model/conv2d_7/kernel/Regularizer/mulMul6chess_model/conv2d_7/kernel/Regularizer/mul/x:output:04chess_model/conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/mul?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_8/kernel/Regularizer/SquareSquareEchess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_8/kernel/Regularizer/Square?
-chess_model/conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_8/kernel/Regularizer/Const?
+chess_model/conv2d_8/kernel/Regularizer/SumSum2chess_model/conv2d_8/kernel/Regularizer/Square:y:06chess_model/conv2d_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/Sum?
-chess_model/conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_8/kernel/Regularizer/mul/x?
+chess_model/conv2d_8/kernel/Regularizer/mulMul6chess_model/conv2d_8/kernel/Regularizer/mul/x:output:04chess_model/conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_8/kernel/Regularizer/mul?
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02>
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_2/kernel/Regularizer/SquareSquareDchess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2/
-chess_model/dense_2/kernel/Regularizer/Square?
,chess_model/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_2/kernel/Regularizer/Const?
*chess_model/dense_2/kernel/Regularizer/SumSum1chess_model/dense_2/kernel/Regularizer/Square:y:05chess_model/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/Sum?
,chess_model/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_2/kernel/Regularizer/mul/x?
*chess_model/dense_2/kernel/Regularizer/mulMul5chess_model/dense_2/kernel/Regularizer/mul/x:output:03chess_model/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/mul? 
IdentityIdentitydense_2/Tanh:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1<^chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp>^chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp;^chess_model/dense/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp=^chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12z
;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp;chess_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp2~
=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_8/kernel/Regularizer/Square/ReadVariableOp2x
:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp:chess_model/dense/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_1/kernel/Regularizer/Square/ReadVariableOp2|
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_179705

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_179449

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_179347

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1784282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1141

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_1/kernel/Regularizer/SquareSquareEchess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_1/kernel/Regularizer/Square?
-chess_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_1/kernel/Regularizer/Const?
+chess_model/conv2d_1/kernel/Regularizer/SumSum2chess_model/conv2d_1/kernel/Regularizer/Square:y:06chess_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/Sum?
-chess_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_1/kernel/Regularizer/mul/x?
+chess_model/conv2d_1/kernel/Regularizer/mulMul6chess_model/conv2d_1/kernel/Regularizer/mul/x:output:04chess_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_conv2d_7_layer_call_and_return_conditional_losses_642

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_7/kernel/Regularizer/SquareSquareEchess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_7/kernel/Regularizer/Square?
-chess_model/conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_7/kernel/Regularizer/Const?
+chess_model/conv2d_7/kernel/Regularizer/SumSum2chess_model/conv2d_7/kernel/Regularizer/Square:y:06chess_model/conv2d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/Sum?
-chess_model/conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_7/kernel/Regularizer/mul/x?
+chess_model/conv2d_7/kernel/Regularizer/mulMul6chess_model/conv2d_7/kernel/Regularizer/mul/x:output:04chess_model/conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_7/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_7/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_conv2d_6_layer_call_and_return_conditional_losses_708

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02?
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
.chess_model/conv2d_6/kernel/Regularizer/SquareSquareEchess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??20
.chess_model/conv2d_6/kernel/Regularizer/Square?
-chess_model/conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-chess_model/conv2d_6/kernel/Regularizer/Const?
+chess_model/conv2d_6/kernel/Regularizer/SumSum2chess_model/conv2d_6/kernel/Regularizer/Square:y:06chess_model/conv2d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/Sum?
-chess_model/conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-chess_model/conv2d_6/kernel/Regularizer/mul/x?
+chess_model/conv2d_6/kernel/Regularizer/mulMul6chess_model/conv2d_6/kernel/Regularizer/mul/x:output:04chess_model/conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+chess_model/conv2d_6/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp=chess_model/conv2d_6/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_178335

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1781282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_6_layer_call_fn_179718

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1790212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3784

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_178636

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_7_1800075
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
4__inference_batch_normalization_layer_call_fn_179334

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1783972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_179321

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_7_layer_call_fn_179795

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1791562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_9_1800295
1kernel_regularizer_square_readvariableop_resource
identity??(kernel/Regularizer/Square/ReadVariableOp?
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(kernel/Regularizer/Square/ReadVariableOp?
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2
kernel/Regularizer/Square?
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/Const?
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82
kernel/Regularizer/mul/x?
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul?
IdentityIdentitykernel/Regularizer/mul:z:0)^kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_179260

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4103

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_179021

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_chess_model_layer_call_fn_4459

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_chess_model_layer_call_and_return_conditional_losses_43292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_2_layer_call_fn_179475

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1786362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_dense_2_layer_call_and_return_conditional_losses_3442

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02>
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp?
-chess_model/dense_2/kernel/Regularizer/SquareSquareDchess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2/
-chess_model/dense_2/kernel/Regularizer/Square?
,chess_model/dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,chess_model/dense_2/kernel/Regularizer/Const?
*chess_model/dense_2/kernel/Regularizer/SumSum1chess_model/dense_2/kernel/Regularizer/Square:y:05chess_model/dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/Sum?
,chess_model/dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82.
,chess_model/dense_2/kernel/Regularizer/mul/x?
*chess_model/dense_2/kernel/Regularizer/mulMul5chess_model/dense_2/kernel/Regularizer/mul/x:output:03chess_model/dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*chess_model/dense_2/kernel/Regularizer/mul?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp=^chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2|
<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp<chess_model/dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_178397

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
inputs7
serving_default_inputs:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?

dense1

dense2

normalize1

normalize2

normalize3

normalize4

normalize5

normalize6
	
normalize7


normalize8

normalize9
convolution1
convolution2
convolution3
convolution4
convolution5
convolution6
convolution7
convolution8
convolution9
add
relu
	valueHead

signatures
	optimizer
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ChessModel", "name": "chess_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ChessModel"}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0003000000142492354, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
 bias
#!_self_saveable_object_factories
"trainable_variables
#	variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
?

&kernel
'bias
#(_self_saveable_object_factories
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
?	
-axis
	.gamma
/beta
0moving_mean
1moving_variance
#2_self_saveable_object_factories
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 6, 256]}}
?	
7axis
	8gamma
9beta
:moving_mean
;moving_variance
#<_self_saveable_object_factories
=trainable_variables
>	variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 6, 256]}}
?	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
#F_self_saveable_object_factories
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 6, 256]}}
?	
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
#P_self_saveable_object_factories
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4, 4, 256]}}
?	
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
#Z_self_saveable_object_factories
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4, 4, 256]}}
?	
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
#d_self_saveable_object_factories
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4, 4, 256]}}
?	
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance
#n_self_saveable_object_factories
otrainable_variables
p	variables
qregularization_losses
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2, 2, 256]}}
?	
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
#x_self_saveable_object_factories
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2, 2, 256]}}
?	
}axis
	~gamma
beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2, 2, 256]}}
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 13]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 8, 13]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 13}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 8, 8, 13]}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 6, 256]}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 6, 256]}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 6, 256]}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4, 4, 256]}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4, 4, 256]}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4, 4, 256]}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2, 2, 256]}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2, 2, 256]}}
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 6, 6, 256]}, {"class_name": "TensorShape", "items": [1, 6, 6, 256]}]}
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
-
?serving_default"
signature_map
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem? m?&m?'m?.m?/m?8m?9m?Bm?Cm?Lm?Mm?Vm?Wm?`m?am?jm?km?tm?um?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?v? v?&v?'v?.v?/v?8v?9v?Bv?Cv?Lv?Mv?Vv?Wv?`v?av?jv?kv?tv?uv?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
?
0
 1
&2
'3
.4
/5
86
97
B8
C9
L10
M11
V12
W13
`14
a15
j16
k17
t18
u19
~20
21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41"
trackable_list_wrapper
?
0
 1
&2
'3
.4
/5
06
17
88
99
:10
;11
B12
C13
D14
E15
L16
M17
N18
O19
V20
W21
X22
Y23
`24
a25
b26
c27
j28
k29
l30
m31
t32
u33
v34
w35
~36
37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
trainable_variables
?metrics
	variables
 ?layer_regularization_losses
regularization_losses
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*
??2chess_model/dense/kernel
%:#?2chess_model/dense/bias
 "
trackable_dict_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
"trainable_variables
?metrics
#	variables
 ?layer_regularization_losses
$regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,
??2chess_model/dense_1/kernel
':%?2chess_model/dense_1/bias
 "
trackable_dict_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
)trainable_variables
?metrics
*	variables
 ?layer_regularization_losses
+regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
4:2?2%chess_model/batch_normalization/gamma
3:1?2$chess_model/batch_normalization/beta
<::? (2+chess_model/batch_normalization/moving_mean
@:>? (2/chess_model/batch_normalization/moving_variance
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
<
.0
/1
02
13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
3trainable_variables
?metrics
4	variables
 ?layer_regularization_losses
5regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4?2'chess_model/batch_normalization_1/gamma
5:3?2&chess_model/batch_normalization_1/beta
>:<? (2-chess_model/batch_normalization_1/moving_mean
B:@? (21chess_model/batch_normalization_1/moving_variance
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
=trainable_variables
?metrics
>	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4?2'chess_model/batch_normalization_2/gamma
5:3?2&chess_model/batch_normalization_2/beta
>:<? (2-chess_model/batch_normalization_2/moving_mean
B:@? (21chess_model/batch_normalization_2/moving_variance
 "
trackable_dict_wrapper
.
B0
C1"
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Gtrainable_variables
?metrics
H	variables
 ?layer_regularization_losses
Iregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4?2'chess_model/batch_normalization_3/gamma
5:3?2&chess_model/batch_normalization_3/beta
>:<? (2-chess_model/batch_normalization_3/moving_mean
B:@? (21chess_model/batch_normalization_3/moving_variance
 "
trackable_dict_wrapper
.
L0
M1"
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
Qtrainable_variables
?metrics
R	variables
 ?layer_regularization_losses
Sregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4?2'chess_model/batch_normalization_4/gamma
5:3?2&chess_model/batch_normalization_4/beta
>:<? (2-chess_model/batch_normalization_4/moving_mean
B:@? (21chess_model/batch_normalization_4/moving_variance
 "
trackable_dict_wrapper
.
V0
W1"
trackable_list_wrapper
<
V0
W1
X2
Y3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
[trainable_variables
?metrics
\	variables
 ?layer_regularization_losses
]regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4?2'chess_model/batch_normalization_5/gamma
5:3?2&chess_model/batch_normalization_5/beta
>:<? (2-chess_model/batch_normalization_5/moving_mean
B:@? (21chess_model/batch_normalization_5/moving_variance
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
<
`0
a1
b2
c3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
etrainable_variables
?metrics
f	variables
 ?layer_regularization_losses
gregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4?2'chess_model/batch_normalization_6/gamma
5:3?2&chess_model/batch_normalization_6/beta
>:<? (2-chess_model/batch_normalization_6/moving_mean
B:@? (21chess_model/batch_normalization_6/moving_variance
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
<
j0
k1
l2
m3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
otrainable_variables
?metrics
p	variables
 ?layer_regularization_losses
qregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4?2'chess_model/batch_normalization_7/gamma
5:3?2&chess_model/batch_normalization_7/beta
>:<? (2-chess_model/batch_normalization_7/moving_mean
B:@? (21chess_model/batch_normalization_7/moving_variance
 "
trackable_dict_wrapper
.
t0
u1"
trackable_list_wrapper
<
t0
u1
v2
w3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
ytrainable_variables
?metrics
z	variables
 ?layer_regularization_losses
{regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4?2'chess_model/batch_normalization_8/gamma
5:3?2&chess_model/batch_normalization_8/beta
>:<? (2-chess_model/batch_normalization_8/moving_mean
B:@? (21chess_model/batch_normalization_8/moving_variance
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
>
~0
1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2?2chess_model/conv2d/kernel
&:$?2chess_model/conv2d/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5??2chess_model/conv2d_1/kernel
(:&?2chess_model/conv2d_1/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5??2chess_model/conv2d_2/kernel
(:&?2chess_model/conv2d_2/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5??2chess_model/conv2d_3/kernel
(:&?2chess_model/conv2d_3/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5??2chess_model/conv2d_4/kernel
(:&?2chess_model/conv2d_4/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5??2chess_model/conv2d_5/kernel
(:&?2chess_model/conv2d_5/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5??2chess_model/conv2d_6/kernel
(:&?2chess_model/conv2d_6/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5??2chess_model/conv2d_7/kernel
(:&?2chess_model/conv2d_7/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
7:5??2chess_model/conv2d_8/kernel
(:&?2chess_model/conv2d_8/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+	?2chess_model/dense_2/kernel
&:$2chess_model/dense_2/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?trainable_variables
?metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
00
11
:2
;3
D4
E5
N6
O7
X8
Y9
b10
c11
l12
m13
v14
w15
?16
?17"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
1:/
??2Adam/chess_model/dense/kernel/m
*:(?2Adam/chess_model/dense/bias/m
3:1
??2!Adam/chess_model/dense_1/kernel/m
,:*?2Adam/chess_model/dense_1/bias/m
9:7?2,Adam/chess_model/batch_normalization/gamma/m
8:6?2+Adam/chess_model/batch_normalization/beta/m
;:9?2.Adam/chess_model/batch_normalization_1/gamma/m
::8?2-Adam/chess_model/batch_normalization_1/beta/m
;:9?2.Adam/chess_model/batch_normalization_2/gamma/m
::8?2-Adam/chess_model/batch_normalization_2/beta/m
;:9?2.Adam/chess_model/batch_normalization_3/gamma/m
::8?2-Adam/chess_model/batch_normalization_3/beta/m
;:9?2.Adam/chess_model/batch_normalization_4/gamma/m
::8?2-Adam/chess_model/batch_normalization_4/beta/m
;:9?2.Adam/chess_model/batch_normalization_5/gamma/m
::8?2-Adam/chess_model/batch_normalization_5/beta/m
;:9?2.Adam/chess_model/batch_normalization_6/gamma/m
::8?2-Adam/chess_model/batch_normalization_6/beta/m
;:9?2.Adam/chess_model/batch_normalization_7/gamma/m
::8?2-Adam/chess_model/batch_normalization_7/beta/m
;:9?2.Adam/chess_model/batch_normalization_8/gamma/m
::8?2-Adam/chess_model/batch_normalization_8/beta/m
9:7?2 Adam/chess_model/conv2d/kernel/m
+:)?2Adam/chess_model/conv2d/bias/m
<::??2"Adam/chess_model/conv2d_1/kernel/m
-:+?2 Adam/chess_model/conv2d_1/bias/m
<::??2"Adam/chess_model/conv2d_2/kernel/m
-:+?2 Adam/chess_model/conv2d_2/bias/m
<::??2"Adam/chess_model/conv2d_3/kernel/m
-:+?2 Adam/chess_model/conv2d_3/bias/m
<::??2"Adam/chess_model/conv2d_4/kernel/m
-:+?2 Adam/chess_model/conv2d_4/bias/m
<::??2"Adam/chess_model/conv2d_5/kernel/m
-:+?2 Adam/chess_model/conv2d_5/bias/m
<::??2"Adam/chess_model/conv2d_6/kernel/m
-:+?2 Adam/chess_model/conv2d_6/bias/m
<::??2"Adam/chess_model/conv2d_7/kernel/m
-:+?2 Adam/chess_model/conv2d_7/bias/m
<::??2"Adam/chess_model/conv2d_8/kernel/m
-:+?2 Adam/chess_model/conv2d_8/bias/m
2:0	?2!Adam/chess_model/dense_2/kernel/m
+:)2Adam/chess_model/dense_2/bias/m
1:/
??2Adam/chess_model/dense/kernel/v
*:(?2Adam/chess_model/dense/bias/v
3:1
??2!Adam/chess_model/dense_1/kernel/v
,:*?2Adam/chess_model/dense_1/bias/v
9:7?2,Adam/chess_model/batch_normalization/gamma/v
8:6?2+Adam/chess_model/batch_normalization/beta/v
;:9?2.Adam/chess_model/batch_normalization_1/gamma/v
::8?2-Adam/chess_model/batch_normalization_1/beta/v
;:9?2.Adam/chess_model/batch_normalization_2/gamma/v
::8?2-Adam/chess_model/batch_normalization_2/beta/v
;:9?2.Adam/chess_model/batch_normalization_3/gamma/v
::8?2-Adam/chess_model/batch_normalization_3/beta/v
;:9?2.Adam/chess_model/batch_normalization_4/gamma/v
::8?2-Adam/chess_model/batch_normalization_4/beta/v
;:9?2.Adam/chess_model/batch_normalization_5/gamma/v
::8?2-Adam/chess_model/batch_normalization_5/beta/v
;:9?2.Adam/chess_model/batch_normalization_6/gamma/v
::8?2-Adam/chess_model/batch_normalization_6/beta/v
;:9?2.Adam/chess_model/batch_normalization_7/gamma/v
::8?2-Adam/chess_model/batch_normalization_7/beta/v
;:9?2.Adam/chess_model/batch_normalization_8/gamma/v
::8?2-Adam/chess_model/batch_normalization_8/beta/v
9:7?2 Adam/chess_model/conv2d/kernel/v
+:)?2Adam/chess_model/conv2d/bias/v
<::??2"Adam/chess_model/conv2d_1/kernel/v
-:+?2 Adam/chess_model/conv2d_1/bias/v
<::??2"Adam/chess_model/conv2d_2/kernel/v
-:+?2 Adam/chess_model/conv2d_2/bias/v
<::??2"Adam/chess_model/conv2d_3/kernel/v
-:+?2 Adam/chess_model/conv2d_3/bias/v
<::??2"Adam/chess_model/conv2d_4/kernel/v
-:+?2 Adam/chess_model/conv2d_4/bias/v
<::??2"Adam/chess_model/conv2d_5/kernel/v
-:+?2 Adam/chess_model/conv2d_5/bias/v
<::??2"Adam/chess_model/conv2d_6/kernel/v
-:+?2 Adam/chess_model/conv2d_6/bias/v
<::??2"Adam/chess_model/conv2d_7/kernel/v
-:+?2 Adam/chess_model/conv2d_7/bias/v
<::??2"Adam/chess_model/conv2d_8/kernel/v
-:+?2 Adam/chess_model/conv2d_8/bias/v
2:0	?2!Adam/chess_model/dense_2/kernel/v
+:)2Adam/chess_model/dense_2/bias/v
?2?
!__inference__wrapped_model_178128?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%
inputs?????????
?2?
*__inference_chess_model_layer_call_fn_4394
*__inference_chess_model_layer_call_fn_4459
*__inference_chess_model_layer_call_fn_4085
*__inference_chess_model_layer_call_fn_4020?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_chess_model_layer_call_and_return_conditional_losses_2351
E__inference_chess_model_layer_call_and_return_conditional_losses_3303
E__inference_chess_model_layer_call_and_return_conditional_losses_1080
E__inference_chess_model_layer_call_and_return_conditional_losses_3002?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_batch_normalization_layer_call_fn_179334
4__inference_batch_normalization_layer_call_fn_179347?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_179303
O__inference_batch_normalization_layer_call_and_return_conditional_losses_179321?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_1_layer_call_fn_179398
6__inference_batch_normalization_1_layer_call_fn_179411?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_179385
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_179367?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_2_layer_call_fn_179462
6__inference_batch_normalization_2_layer_call_fn_179475?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_179449
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_179431?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_3_layer_call_fn_179539
6__inference_batch_normalization_3_layer_call_fn_179526?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_179513
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_179495?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_4_layer_call_fn_179590
6__inference_batch_normalization_4_layer_call_fn_179603?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_179577
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_179559?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_5_layer_call_fn_179667
6__inference_batch_normalization_5_layer_call_fn_179654?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_179623
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_179641?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_6_layer_call_fn_179718
6__inference_batch_normalization_6_layer_call_fn_179731?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_179687
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_179705?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_7_layer_call_fn_179795
6__inference_batch_normalization_7_layer_call_fn_179782?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_179751
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_179769?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_8_layer_call_fn_179859
6__inference_batch_normalization_8_layer_call_fn_179846?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_179815
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_179833?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_178335inputs"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_179930?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_179941?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_179952?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_179963?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_179974?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_179985?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_6_179996?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_7_180007?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_8_180018?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_9_180029?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_10_180040?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_11_180051?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? ?
!__inference__wrapped_model_178128?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??7?4
-?*
(?%
inputs?????????
? "3?0
.
output_1"?
output_1??????????
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_179367?89:;N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_179385?89:;N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_1_layer_call_fn_179398?89:;N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_1_layer_call_fn_179411?89:;N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_179431?BCDEN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_179449?BCDEN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_2_layer_call_fn_179462?BCDEN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_2_layer_call_fn_179475?BCDEN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_179495?LMNON?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_179513?LMNON?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_3_layer_call_fn_179526?LMNON?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_3_layer_call_fn_179539?LMNON?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_179559?VWXYN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_179577?VWXYN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_4_layer_call_fn_179590?VWXYN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_4_layer_call_fn_179603?VWXYN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_179623?`abcN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_179641?`abcN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_5_layer_call_fn_179654?`abcN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_5_layer_call_fn_179667?`abcN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_179687?jklmN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_179705?jklmN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_6_layer_call_fn_179718?jklmN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_6_layer_call_fn_179731?jklmN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_179751?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_179769?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_7_layer_call_fn_179782?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_7_layer_call_fn_179795?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_179815?~??N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_179833?~??N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_8_layer_call_fn_179846?~??N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_8_layer_call_fn_179859?~??N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
O__inference_batch_normalization_layer_call_and_return_conditional_losses_179303?./01N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_179321?./01N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
4__inference_batch_normalization_layer_call_fn_179334?./01N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
4__inference_batch_normalization_layer_call_fn_179347?./01N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
E__inference_chess_model_layer_call_and_return_conditional_losses_1080?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??<?9
2?/
)?&
input_1?????????
p 
? "%?"
?
0?????????
? ?
E__inference_chess_model_layer_call_and_return_conditional_losses_2351?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??;?8
1?.
(?%
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_chess_model_layer_call_and_return_conditional_losses_3002?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??<?9
2?/
)?&
input_1?????????
p
? "%?"
?
0?????????
? ?
E__inference_chess_model_layer_call_and_return_conditional_losses_3303?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??;?8
1?.
(?%
inputs?????????
p
? "%?"
?
0?????????
? ?
*__inference_chess_model_layer_call_fn_4020?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??<?9
2?/
)?&
input_1?????????
p
? "???????????
*__inference_chess_model_layer_call_fn_4085?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??;?8
1?.
(?%
inputs?????????
p
? "???????????
*__inference_chess_model_layer_call_fn_4394?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??<?9
2?/
)?&
input_1?????????
p 
? "???????????
*__inference_chess_model_layer_call_fn_4459?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??;?8
1?.
(?%
inputs?????????
p 
? "??????????;
__inference_loss_fn_0_179930?

? 
? "? =
__inference_loss_fn_10_180040??

? 
? "? =
__inference_loss_fn_11_180051??

? 
? "? ;
__inference_loss_fn_1_179941&?

? 
? "? <
__inference_loss_fn_2_179952??

? 
? "? <
__inference_loss_fn_3_179963??

? 
? "? <
__inference_loss_fn_4_179974??

? 
? "? <
__inference_loss_fn_5_179985??

? 
? "? <
__inference_loss_fn_6_179996??

? 
? "? <
__inference_loss_fn_7_180007??

? 
? "? <
__inference_loss_fn_8_180018??

? 
? "? <
__inference_loss_fn_9_180029??

? 
? "? ?
$__inference_signature_wrapper_178335?R??./01??89:;??BCDE??LMNO??VWXY??`abc??jklm??tuvw??~?? &'??A?>
? 
7?4
2
inputs(?%
inputs?????????"3?0
.
output_1"?
output_1?????????