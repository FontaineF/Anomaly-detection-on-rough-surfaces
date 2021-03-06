??:
??
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
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??3
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_3/kernel
?
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
: *
dtype0
?
batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_26/gamma
?
0batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_26/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_26/beta
?
/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_26/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_26/moving_mean
?
6batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_26/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_26/moving_variance
?
:batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_26/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
: *
dtype0
?
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_27/gamma
?
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_27/beta
?
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_27/moving_mean
?
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_27/moving_variance
?
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
: *
dtype0
?
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_28/gamma
?
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_28/beta
?
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_28/moving_mean
?
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_28/moving_variance
?
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_4/kernel
?
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:*
dtype0
?
batch_normalization_29/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_29/gamma
?
0batch_normalization_29/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_29/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_29/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_29/beta
?
/batch_normalization_29/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_29/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_29/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_29/moving_mean
?
6batch_normalization_29/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_29/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_29/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_29/moving_variance
?
:batch_normalization_29/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_29/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:*
dtype0
?
batch_normalization_30/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_30/gamma
?
0batch_normalization_30/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_30/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_30/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_30/beta
?
/batch_normalization_30/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_30/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_30/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_30/moving_mean
?
6batch_normalization_30/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_30/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_30/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_30/moving_variance
?
:batch_normalization_30/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_30/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
:*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:*
dtype0
?
batch_normalization_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_31/gamma
?
0batch_normalization_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_31/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_31/beta
?
/batch_normalization_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_31/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_31/moving_mean
?
6batch_normalization_31/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_31/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_31/moving_variance
?
:batch_normalization_31/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_31/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_5/kernel
?
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:*
dtype0
?
batch_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_32/gamma
?
0batch_normalization_32/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_32/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_32/beta
?
/batch_normalization_32/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_32/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_32/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_32/moving_mean
?
6batch_normalization_32/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_32/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_32/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_32/moving_variance
?
:batch_normalization_32/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_32/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:*
dtype0
?
batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_33/gamma
?
0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_33/beta
?
/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_33/moving_mean
?
6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_33/moving_variance
?
:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:*
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
:*
dtype0
?
batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_34/gamma
?
0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_34/beta
?
/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_34/moving_mean
?
6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_34/moving_variance
?
:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_29/kernel
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*&
_output_shapes
:*
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:*
dtype0
?
batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_35/gamma
?
0batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_35/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_35/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_35/beta
?
/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_35/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_35/moving_mean
?
6batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_35/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_35/moving_variance
?
:batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_35/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
valueծBѮ Bɮ
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
 	keras_api
?
!layer_with_weights-0
!layer-0
"layer_with_weights-1
"layer-1
#layer-2
$layer_with_weights-2
$layer-3
%layer_with_weights-3
%layer-4
&layer-5
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?
+layer_with_weights-0
+layer-0
,layer_with_weights-1
,layer-1
-layer-2
.	variables
/trainable_variables
0regularization_losses
1	keras_api
?
2layer_with_weights-0
2layer-0
3layer_with_weights-1
3layer-1
4layer-2
5layer_with_weights-2
5layer-3
6layer_with_weights-3
6layer-4
7layer-5
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?
<layer_with_weights-0
<layer-0
=layer_with_weights-1
=layer-1
>layer-2
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?
Clayer_with_weights-0
Clayer-0
Dlayer_with_weights-1
Dlayer-1
Elayer-2
Flayer_with_weights-2
Flayer-3
Glayer_with_weights-3
Glayer-4
Hlayer-5
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?
Mlayer_with_weights-0
Mlayer-0
Nlayer_with_weights-1
Nlayer-1
Olayer-2
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?
0
1
T2
U3
V4
W5
X6
Y7
Z8
[9
\10
]11
^12
_13
`14
a15
b16
c17
d18
e19
f20
g21
h22
i23
j24
k25
l26
m27
n28
o29
p30
q31
r32
s33
t34
u35
v36
w37
x38
y39
z40
{41
|42
}43
~44
45
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
?60
?61
?
0
1
T2
U3
V4
W5
Z6
[7
\8
]9
`10
a11
b12
c13
f14
g15
h16
i17
l18
m19
n20
o21
r22
s23
t24
u25
x26
y27
z28
{29
~30
31
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
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
l

Tkernel
Ubias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
T0
U1
V2
W3
X4
Y5

T0
U1
V2
W3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
l

Zkernel
[bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	\gamma
]beta
^moving_mean
_moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

`kernel
abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	bgamma
cbeta
dmoving_mean
emoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
8
Z0
[1
\2
]3
`4
a5
b6
c7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
l

fkernel
gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	hgamma
ibeta
jmoving_mean
kmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
f0
g1
h2
i3
j4
k5

f0
g1
h2
i3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
l

lkernel
mbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	ngamma
obeta
pmoving_mean
qmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

rkernel
sbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	tgamma
ubeta
vmoving_mean
wmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
l0
m1
n2
o3
p4
q5
r6
s7
t8
u9
v10
w11
8
l0
m1
n2
o3
r4
s5
t6
u7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
l

xkernel
ybias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	zgamma
{beta
|moving_mean
}moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
x0
y1
z2
{3
|4
}5

x0
y1
z2
{3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
l

~kernel
bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
`
~0
1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
>
~0
1
?2
?3
?4
?5
?6
?7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
0
?0
?1
?2
?3
?4
?5
 
?0
?1
?2
?3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
US
VARIABLE_VALUEconv2d_transpose_3/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_transpose_3/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_26/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_26/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_26/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_26/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_23/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_23/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_27/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_27/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_27/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_27/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_24/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_24/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_28/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_28/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_28/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_28/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_4/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_4/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_29/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_29/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_29/moving_mean'variables/24/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_29/moving_variance'variables/25/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_25/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_25/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_30/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_30/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_30/moving_mean'variables/30/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_30/moving_variance'variables/31/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_26/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_26/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_31/gamma'variables/34/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_31/beta'variables/35/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_31/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_31/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_5/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_5/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_32/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_32/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_32/moving_mean'variables/42/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_32/moving_variance'variables/43/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_27/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_27/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_33/gamma'variables/46/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_33/beta'variables/47/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_33/moving_mean'variables/48/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_33/moving_variance'variables/49/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_28/kernel'variables/50/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_28/bias'variables/51/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_34/gamma'variables/52/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_34/beta'variables/53/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_34/moving_mean'variables/54/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_34/moving_variance'variables/55/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_29/kernel'variables/56/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_29/bias'variables/57/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_35/gamma'variables/58/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_35/beta'variables/59/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_35/moving_mean'variables/60/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_35/moving_variance'variables/61/.ATTRIBUTES/VARIABLE_VALUE
?
X0
Y1
^2
_3
d4
e5
j6
k7
p8
q9
v10
w11
|12
}13
?14
?15
?16
?17
?18
?19
F
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

T0
U1

T0
U1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

V0
W1
X2
Y3

V0
W1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

X0
Y1

0
1
2
 
 
 

Z0
[1

Z0
[1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

\0
]1
^2
_3

\0
]1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

`0
a1

`0
a1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

b0
c1
d2
e3

b0
c1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

^0
_1
d2
e3
*
!0
"1
#2
$3
%4
&5
 
 
 

f0
g1

f0
g1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

h0
i1
j2
k3

h0
i1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

j0
k1

+0
,1
-2
 
 
 

l0
m1

l0
m1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

n0
o1
p2
q3

n0
o1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

r0
s1

r0
s1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

t0
u1
v2
w3

t0
u1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

p0
q1
v2
w3
*
20
31
42
53
64
75
 
 
 

x0
y1

x0
y1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

z0
{1
|2
}3

z0
{1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

|0
}1

<0
=1
>2
 
 
 

~0
1

~0
1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
?0
?1
?2
?3
*
C0
D1
E2
F3
G4
H5
 
 
 

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

?0
?1

M0
N1
O2
 
 
 
 
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
^0
_1
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

d0
e1
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

j0
k1
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

p0
q1
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

v0
w1
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

|0
}1
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
{
serving_default_args_0Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0dense_1/kerneldense_1/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_23/kernelconv2d_23/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_24/kernelconv2d_24/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_varianceconv2d_25/kernelconv2d_25/biasbatch_normalization_30/gammabatch_normalization_30/beta"batch_normalization_30/moving_mean&batch_normalization_30/moving_varianceconv2d_26/kernelconv2d_26/biasbatch_normalization_31/gammabatch_normalization_31/beta"batch_normalization_31/moving_mean&batch_normalization_31/moving_varianceconv2d_transpose_5/kernelconv2d_transpose_5/biasbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_27/kernelconv2d_27/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_28/kernelconv2d_28/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_29/kernelconv2d_29/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variance*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_61114
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp0batch_normalization_26/gamma/Read/ReadVariableOp/batch_normalization_26/beta/Read/ReadVariableOp6batch_normalization_26/moving_mean/Read/ReadVariableOp:batch_normalization_26/moving_variance/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp0batch_normalization_28/gamma/Read/ReadVariableOp/batch_normalization_28/beta/Read/ReadVariableOp6batch_normalization_28/moving_mean/Read/ReadVariableOp:batch_normalization_28/moving_variance/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp0batch_normalization_29/gamma/Read/ReadVariableOp/batch_normalization_29/beta/Read/ReadVariableOp6batch_normalization_29/moving_mean/Read/ReadVariableOp:batch_normalization_29/moving_variance/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp0batch_normalization_30/gamma/Read/ReadVariableOp/batch_normalization_30/beta/Read/ReadVariableOp6batch_normalization_30/moving_mean/Read/ReadVariableOp:batch_normalization_30/moving_variance/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp0batch_normalization_31/gamma/Read/ReadVariableOp/batch_normalization_31/beta/Read/ReadVariableOp6batch_normalization_31/moving_mean/Read/ReadVariableOp:batch_normalization_31/moving_variance/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOp0batch_normalization_32/gamma/Read/ReadVariableOp/batch_normalization_32/beta/Read/ReadVariableOp6batch_normalization_32/moving_mean/Read/ReadVariableOp:batch_normalization_32/moving_variance/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp0batch_normalization_33/gamma/Read/ReadVariableOp/batch_normalization_33/beta/Read/ReadVariableOp6batch_normalization_33/moving_mean/Read/ReadVariableOp:batch_normalization_33/moving_variance/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp0batch_normalization_34/gamma/Read/ReadVariableOp/batch_normalization_34/beta/Read/ReadVariableOp6batch_normalization_34/moving_mean/Read/ReadVariableOp:batch_normalization_34/moving_variance/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp0batch_normalization_35/gamma/Read/ReadVariableOp/batch_normalization_35/beta/Read/ReadVariableOp6batch_normalization_35/moving_mean/Read/ReadVariableOp:batch_normalization_35/moving_variance/Read/ReadVariableOpConst*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_64728
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_23/kernelconv2d_23/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_24/kernelconv2d_24/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_varianceconv2d_25/kernelconv2d_25/biasbatch_normalization_30/gammabatch_normalization_30/beta"batch_normalization_30/moving_mean&batch_normalization_30/moving_varianceconv2d_26/kernelconv2d_26/biasbatch_normalization_31/gammabatch_normalization_31/beta"batch_normalization_31/moving_mean&batch_normalization_31/moving_varianceconv2d_transpose_5/kernelconv2d_transpose_5/biasbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_27/kernelconv2d_27/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_28/kernelconv2d_28/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_29/kernelconv2d_29/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variance*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_64924??0
?

?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_63043

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62960

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63838

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63802

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57960

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_26_layer_call_fn_63172

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_57841h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64509

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63113

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?N
B__inference_model_4_layer_call_and_return_conditional_losses_61910

inputs:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?c
Isequential_15_conv2d_transpose_3_conv2d_transpose_readvariableop_resource: N
@sequential_15_conv2d_transpose_3_biasadd_readvariableop_resource: J
<sequential_15_batch_normalization_26_readvariableop_resource: L
>sequential_15_batch_normalization_26_readvariableop_1_resource: [
Msequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_resource: ]
Osequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource: P
6sequential_16_conv2d_23_conv2d_readvariableop_resource:  E
7sequential_16_conv2d_23_biasadd_readvariableop_resource: J
<sequential_16_batch_normalization_27_readvariableop_resource: L
>sequential_16_batch_normalization_27_readvariableop_1_resource: [
Msequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_resource: ]
Osequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource: P
6sequential_16_conv2d_24_conv2d_readvariableop_resource:  E
7sequential_16_conv2d_24_biasadd_readvariableop_resource: J
<sequential_16_batch_normalization_28_readvariableop_resource: L
>sequential_16_batch_normalization_28_readvariableop_1_resource: [
Msequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_resource: ]
Osequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource: c
Isequential_17_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: N
@sequential_17_conv2d_transpose_4_biasadd_readvariableop_resource:J
<sequential_17_batch_normalization_29_readvariableop_resource:L
>sequential_17_batch_normalization_29_readvariableop_1_resource:[
Msequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_18_conv2d_25_conv2d_readvariableop_resource:E
7sequential_18_conv2d_25_biasadd_readvariableop_resource:J
<sequential_18_batch_normalization_30_readvariableop_resource:L
>sequential_18_batch_normalization_30_readvariableop_1_resource:[
Msequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_resource:]
Osequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_18_conv2d_26_conv2d_readvariableop_resource:E
7sequential_18_conv2d_26_biasadd_readvariableop_resource:J
<sequential_18_batch_normalization_31_readvariableop_resource:L
>sequential_18_batch_normalization_31_readvariableop_1_resource:[
Msequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_resource:]
Osequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:c
Isequential_19_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:N
@sequential_19_conv2d_transpose_5_biasadd_readvariableop_resource:J
<sequential_19_batch_normalization_32_readvariableop_resource:L
>sequential_19_batch_normalization_32_readvariableop_1_resource:[
Msequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:]
Osequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_20_conv2d_27_conv2d_readvariableop_resource:E
7sequential_20_conv2d_27_biasadd_readvariableop_resource:J
<sequential_20_batch_normalization_33_readvariableop_resource:L
>sequential_20_batch_normalization_33_readvariableop_1_resource:[
Msequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:]
Osequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_20_conv2d_28_conv2d_readvariableop_resource:E
7sequential_20_conv2d_28_biasadd_readvariableop_resource:J
<sequential_20_batch_normalization_34_readvariableop_resource:L
>sequential_20_batch_normalization_34_readvariableop_1_resource:[
Msequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:]
Osequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_21_conv2d_29_conv2d_readvariableop_resource:E
7sequential_21_conv2d_29_biasadd_readvariableop_resource:J
<sequential_21_batch_normalization_35_readvariableop_resource:L
>sequential_21_batch_normalization_35_readvariableop_1_resource:[
Msequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_resource:]
Osequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?3sequential_15/batch_normalization_26/AssignNewValue?5sequential_15/batch_normalization_26/AssignNewValue_1?Dsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Fsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?3sequential_15/batch_normalization_26/ReadVariableOp?5sequential_15/batch_normalization_26/ReadVariableOp_1?7sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp?@sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?3sequential_16/batch_normalization_27/AssignNewValue?5sequential_16/batch_normalization_27/AssignNewValue_1?Dsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Fsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?3sequential_16/batch_normalization_27/ReadVariableOp?5sequential_16/batch_normalization_27/ReadVariableOp_1?3sequential_16/batch_normalization_28/AssignNewValue?5sequential_16/batch_normalization_28/AssignNewValue_1?Dsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Fsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?3sequential_16/batch_normalization_28/ReadVariableOp?5sequential_16/batch_normalization_28/ReadVariableOp_1?.sequential_16/conv2d_23/BiasAdd/ReadVariableOp?-sequential_16/conv2d_23/Conv2D/ReadVariableOp?.sequential_16/conv2d_24/BiasAdd/ReadVariableOp?-sequential_16/conv2d_24/Conv2D/ReadVariableOp?3sequential_17/batch_normalization_29/AssignNewValue?5sequential_17/batch_normalization_29/AssignNewValue_1?Dsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?Fsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?3sequential_17/batch_normalization_29/ReadVariableOp?5sequential_17/batch_normalization_29/ReadVariableOp_1?7sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp?@sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?3sequential_18/batch_normalization_30/AssignNewValue?5sequential_18/batch_normalization_30/AssignNewValue_1?Dsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?Fsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?3sequential_18/batch_normalization_30/ReadVariableOp?5sequential_18/batch_normalization_30/ReadVariableOp_1?3sequential_18/batch_normalization_31/AssignNewValue?5sequential_18/batch_normalization_31/AssignNewValue_1?Dsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?Fsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?3sequential_18/batch_normalization_31/ReadVariableOp?5sequential_18/batch_normalization_31/ReadVariableOp_1?.sequential_18/conv2d_25/BiasAdd/ReadVariableOp?-sequential_18/conv2d_25/Conv2D/ReadVariableOp?.sequential_18/conv2d_26/BiasAdd/ReadVariableOp?-sequential_18/conv2d_26/Conv2D/ReadVariableOp?3sequential_19/batch_normalization_32/AssignNewValue?5sequential_19/batch_normalization_32/AssignNewValue_1?Dsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?Fsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?3sequential_19/batch_normalization_32/ReadVariableOp?5sequential_19/batch_normalization_32/ReadVariableOp_1?7sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp?@sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?3sequential_20/batch_normalization_33/AssignNewValue?5sequential_20/batch_normalization_33/AssignNewValue_1?Dsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?Fsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?3sequential_20/batch_normalization_33/ReadVariableOp?5sequential_20/batch_normalization_33/ReadVariableOp_1?3sequential_20/batch_normalization_34/AssignNewValue?5sequential_20/batch_normalization_34/AssignNewValue_1?Dsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp?Fsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?3sequential_20/batch_normalization_34/ReadVariableOp?5sequential_20/batch_normalization_34/ReadVariableOp_1?.sequential_20/conv2d_27/BiasAdd/ReadVariableOp?-sequential_20/conv2d_27/Conv2D/ReadVariableOp?.sequential_20/conv2d_28/BiasAdd/ReadVariableOp?-sequential_20/conv2d_28/Conv2D/ReadVariableOp?3sequential_21/batch_normalization_35/AssignNewValue?5sequential_21/batch_normalization_35/AssignNewValue_1?Dsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp?Fsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?3sequential_21/batch_normalization_35/ReadVariableOp?5sequential_21/batch_normalization_35/ReadVariableOp_1?.sequential_21/conv2d_29/BiasAdd/ReadVariableOp?-sequential_21/conv2d_29/Conv2D/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0z
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????W
reshape_1/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_1/ReshapeReshapedense_1/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p
&sequential_15/conv2d_transpose_3/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:~
4sequential_15/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_15/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_15/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_15/conv2d_transpose_3/strided_sliceStridedSlice/sequential_15/conv2d_transpose_3/Shape:output:0=sequential_15/conv2d_transpose_3/strided_slice/stack:output:0?sequential_15/conv2d_transpose_3/strided_slice/stack_1:output:0?sequential_15/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_15/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_15/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_15/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_15/conv2d_transpose_3/stackPack7sequential_15/conv2d_transpose_3/strided_slice:output:01sequential_15/conv2d_transpose_3/stack/1:output:01sequential_15/conv2d_transpose_3/stack/2:output:01sequential_15/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_15/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_15/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_15/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_15/conv2d_transpose_3/strided_slice_1StridedSlice/sequential_15/conv2d_transpose_3/stack:output:0?sequential_15/conv2d_transpose_3/strided_slice_1/stack:output:0Asequential_15/conv2d_transpose_3/strided_slice_1/stack_1:output:0Asequential_15/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_15_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
1sequential_15/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput/sequential_15/conv2d_transpose_3/stack:output:0Hsequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
7sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp@sequential_15_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
(sequential_15/conv2d_transpose_3/BiasAddBiasAdd:sequential_15/conv2d_transpose_3/conv2d_transpose:output:0?sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_15/batch_normalization_26/ReadVariableOpReadVariableOp<sequential_15_batch_normalization_26_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_15/batch_normalization_26/ReadVariableOp_1ReadVariableOp>sequential_15_batch_normalization_26_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_15/batch_normalization_26/FusedBatchNormV3FusedBatchNormV31sequential_15/conv2d_transpose_3/BiasAdd:output:0;sequential_15/batch_normalization_26/ReadVariableOp:value:0=sequential_15/batch_normalization_26/ReadVariableOp_1:value:0Lsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_15/batch_normalization_26/AssignNewValueAssignVariableOpMsequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_resourceBsequential_15/batch_normalization_26/FusedBatchNormV3:batch_mean:0E^sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_15/batch_normalization_26/AssignNewValue_1AssignVariableOpOsequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resourceFsequential_15/batch_normalization_26/FusedBatchNormV3:batch_variance:0G^sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_15/leaky_re_lu_25/LeakyRelu	LeakyRelu9sequential_15/batch_normalization_26/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
-sequential_16/conv2d_23/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_16/conv2d_23/Conv2DConv2D4sequential_15/leaky_re_lu_25/LeakyRelu:activations:05sequential_16/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_16/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_16/conv2d_23/BiasAddBiasAdd'sequential_16/conv2d_23/Conv2D:output:06sequential_16/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_16/batch_normalization_27/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_16/batch_normalization_27/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_16/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3(sequential_16/conv2d_23/BiasAdd:output:0;sequential_16/batch_normalization_27/ReadVariableOp:value:0=sequential_16/batch_normalization_27/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_16/batch_normalization_27/AssignNewValueAssignVariableOpMsequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_resourceBsequential_16/batch_normalization_27/FusedBatchNormV3:batch_mean:0E^sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_16/batch_normalization_27/AssignNewValue_1AssignVariableOpOsequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resourceFsequential_16/batch_normalization_27/FusedBatchNormV3:batch_variance:0G^sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_16/leaky_re_lu_26/LeakyRelu	LeakyRelu9sequential_16/batch_normalization_27/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
-sequential_16/conv2d_24/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_16/conv2d_24/Conv2DConv2D4sequential_16/leaky_re_lu_26/LeakyRelu:activations:05sequential_16/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_16/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_16/conv2d_24/BiasAddBiasAdd'sequential_16/conv2d_24/Conv2D:output:06sequential_16/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_16/batch_normalization_28/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_16/batch_normalization_28/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_16/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3(sequential_16/conv2d_24/BiasAdd:output:0;sequential_16/batch_normalization_28/ReadVariableOp:value:0=sequential_16/batch_normalization_28/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_16/batch_normalization_28/AssignNewValueAssignVariableOpMsequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_resourceBsequential_16/batch_normalization_28/FusedBatchNormV3:batch_mean:0E^sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_16/batch_normalization_28/AssignNewValue_1AssignVariableOpOsequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resourceFsequential_16/batch_normalization_28/FusedBatchNormV3:batch_variance:0G^sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_16/leaky_re_lu_27/LeakyRelu	LeakyRelu9sequential_16/batch_normalization_28/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
&sequential_17/conv2d_transpose_4/ShapeShape4sequential_16/leaky_re_lu_27/LeakyRelu:activations:0*
T0*
_output_shapes
:~
4sequential_17/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_17/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_17/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_17/conv2d_transpose_4/strided_sliceStridedSlice/sequential_17/conv2d_transpose_4/Shape:output:0=sequential_17/conv2d_transpose_4/strided_slice/stack:output:0?sequential_17/conv2d_transpose_4/strided_slice/stack_1:output:0?sequential_17/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_17/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_17/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_17/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
&sequential_17/conv2d_transpose_4/stackPack7sequential_17/conv2d_transpose_4/strided_slice:output:01sequential_17/conv2d_transpose_4/stack/1:output:01sequential_17/conv2d_transpose_4/stack/2:output:01sequential_17/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_17/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_17/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_17/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_17/conv2d_transpose_4/strided_slice_1StridedSlice/sequential_17/conv2d_transpose_4/stack:output:0?sequential_17/conv2d_transpose_4/strided_slice_1/stack:output:0Asequential_17/conv2d_transpose_4/strided_slice_1/stack_1:output:0Asequential_17/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_17_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
1sequential_17/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput/sequential_17/conv2d_transpose_4/stack:output:0Hsequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:04sequential_16/leaky_re_lu_27/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
7sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp@sequential_17_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_17/conv2d_transpose_4/BiasAddBiasAdd:sequential_17/conv2d_transpose_4/conv2d_transpose:output:0?sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
3sequential_17/batch_normalization_29/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_29_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_17/batch_normalization_29/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_29_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_17/batch_normalization_29/FusedBatchNormV3FusedBatchNormV31sequential_17/conv2d_transpose_4/BiasAdd:output:0;sequential_17/batch_normalization_29/ReadVariableOp:value:0=sequential_17/batch_normalization_29/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_17/batch_normalization_29/AssignNewValueAssignVariableOpMsequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_resourceBsequential_17/batch_normalization_29/FusedBatchNormV3:batch_mean:0E^sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_17/batch_normalization_29/AssignNewValue_1AssignVariableOpOsequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resourceFsequential_17/batch_normalization_29/FusedBatchNormV3:batch_variance:0G^sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_17/leaky_re_lu_28/LeakyRelu	LeakyRelu9sequential_17/batch_normalization_29/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
-sequential_18/conv2d_25/Conv2D/ReadVariableOpReadVariableOp6sequential_18_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_18/conv2d_25/Conv2DConv2D4sequential_17/leaky_re_lu_28/LeakyRelu:activations:05sequential_18/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
.sequential_18/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp7sequential_18_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_18/conv2d_25/BiasAddBiasAdd'sequential_18/conv2d_25/Conv2D:output:06sequential_18/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
3sequential_18/batch_normalization_30/ReadVariableOpReadVariableOp<sequential_18_batch_normalization_30_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_18/batch_normalization_30/ReadVariableOp_1ReadVariableOp>sequential_18_batch_normalization_30_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_18/batch_normalization_30/FusedBatchNormV3FusedBatchNormV3(sequential_18/conv2d_25/BiasAdd:output:0;sequential_18/batch_normalization_30/ReadVariableOp:value:0=sequential_18/batch_normalization_30/ReadVariableOp_1:value:0Lsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_18/batch_normalization_30/AssignNewValueAssignVariableOpMsequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_resourceBsequential_18/batch_normalization_30/FusedBatchNormV3:batch_mean:0E^sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_18/batch_normalization_30/AssignNewValue_1AssignVariableOpOsequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resourceFsequential_18/batch_normalization_30/FusedBatchNormV3:batch_variance:0G^sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_18/leaky_re_lu_29/LeakyRelu	LeakyRelu9sequential_18/batch_normalization_30/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
-sequential_18/conv2d_26/Conv2D/ReadVariableOpReadVariableOp6sequential_18_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_18/conv2d_26/Conv2DConv2D4sequential_18/leaky_re_lu_29/LeakyRelu:activations:05sequential_18/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
.sequential_18/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp7sequential_18_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_18/conv2d_26/BiasAddBiasAdd'sequential_18/conv2d_26/Conv2D:output:06sequential_18/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
3sequential_18/batch_normalization_31/ReadVariableOpReadVariableOp<sequential_18_batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_18/batch_normalization_31/ReadVariableOp_1ReadVariableOp>sequential_18_batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_18/batch_normalization_31/FusedBatchNormV3FusedBatchNormV3(sequential_18/conv2d_26/BiasAdd:output:0;sequential_18/batch_normalization_31/ReadVariableOp:value:0=sequential_18/batch_normalization_31/ReadVariableOp_1:value:0Lsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_18/batch_normalization_31/AssignNewValueAssignVariableOpMsequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_resourceBsequential_18/batch_normalization_31/FusedBatchNormV3:batch_mean:0E^sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_18/batch_normalization_31/AssignNewValue_1AssignVariableOpOsequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resourceFsequential_18/batch_normalization_31/FusedBatchNormV3:batch_variance:0G^sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_18/leaky_re_lu_30/LeakyRelu	LeakyRelu9sequential_18/batch_normalization_31/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
&sequential_19/conv2d_transpose_5/ShapeShape4sequential_18/leaky_re_lu_30/LeakyRelu:activations:0*
T0*
_output_shapes
:~
4sequential_19/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_19/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_19/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_19/conv2d_transpose_5/strided_sliceStridedSlice/sequential_19/conv2d_transpose_5/Shape:output:0=sequential_19/conv2d_transpose_5/strided_slice/stack:output:0?sequential_19/conv2d_transpose_5/strided_slice/stack_1:output:0?sequential_19/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_19/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0j
(sequential_19/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0j
(sequential_19/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
&sequential_19/conv2d_transpose_5/stackPack7sequential_19/conv2d_transpose_5/strided_slice:output:01sequential_19/conv2d_transpose_5/stack/1:output:01sequential_19/conv2d_transpose_5/stack/2:output:01sequential_19/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_19/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_19/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_19/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_19/conv2d_transpose_5/strided_slice_1StridedSlice/sequential_19/conv2d_transpose_5/stack:output:0?sequential_19/conv2d_transpose_5/strided_slice_1/stack:output:0Asequential_19/conv2d_transpose_5/strided_slice_1/stack_1:output:0Asequential_19/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_19_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
1sequential_19/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput/sequential_19/conv2d_transpose_5/stack:output:0Hsequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:04sequential_18/leaky_re_lu_30/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00*
paddingVALID*
strides
?
7sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp@sequential_19_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_19/conv2d_transpose_5/BiasAddBiasAdd:sequential_19/conv2d_transpose_5/conv2d_transpose:output:0?sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_19/batch_normalization_32/ReadVariableOpReadVariableOp<sequential_19_batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_19/batch_normalization_32/ReadVariableOp_1ReadVariableOp>sequential_19_batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_19/batch_normalization_32/FusedBatchNormV3FusedBatchNormV31sequential_19/conv2d_transpose_5/BiasAdd:output:0;sequential_19/batch_normalization_32/ReadVariableOp:value:0=sequential_19/batch_normalization_32/ReadVariableOp_1:value:0Lsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_19/batch_normalization_32/AssignNewValueAssignVariableOpMsequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_resourceBsequential_19/batch_normalization_32/FusedBatchNormV3:batch_mean:0E^sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_19/batch_normalization_32/AssignNewValue_1AssignVariableOpOsequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resourceFsequential_19/batch_normalization_32/FusedBatchNormV3:batch_variance:0G^sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_19/leaky_re_lu_31/LeakyRelu	LeakyRelu9sequential_19/batch_normalization_32/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_20/conv2d_27/Conv2D/ReadVariableOpReadVariableOp6sequential_20_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_20/conv2d_27/Conv2DConv2D4sequential_19/leaky_re_lu_31/LeakyRelu:activations:05sequential_20/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_20/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp7sequential_20_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_20/conv2d_27/BiasAddBiasAdd'sequential_20/conv2d_27/Conv2D:output:06sequential_20/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_20/batch_normalization_33/ReadVariableOpReadVariableOp<sequential_20_batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_20/batch_normalization_33/ReadVariableOp_1ReadVariableOp>sequential_20_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_20/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3(sequential_20/conv2d_27/BiasAdd:output:0;sequential_20/batch_normalization_33/ReadVariableOp:value:0=sequential_20/batch_normalization_33/ReadVariableOp_1:value:0Lsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_20/batch_normalization_33/AssignNewValueAssignVariableOpMsequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_resourceBsequential_20/batch_normalization_33/FusedBatchNormV3:batch_mean:0E^sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_20/batch_normalization_33/AssignNewValue_1AssignVariableOpOsequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resourceFsequential_20/batch_normalization_33/FusedBatchNormV3:batch_variance:0G^sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_20/leaky_re_lu_32/LeakyRelu	LeakyRelu9sequential_20/batch_normalization_33/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_20/conv2d_28/Conv2D/ReadVariableOpReadVariableOp6sequential_20_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_20/conv2d_28/Conv2DConv2D4sequential_20/leaky_re_lu_32/LeakyRelu:activations:05sequential_20/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_20/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp7sequential_20_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_20/conv2d_28/BiasAddBiasAdd'sequential_20/conv2d_28/Conv2D:output:06sequential_20/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_20/batch_normalization_34/ReadVariableOpReadVariableOp<sequential_20_batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_20/batch_normalization_34/ReadVariableOp_1ReadVariableOp>sequential_20_batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_20/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3(sequential_20/conv2d_28/BiasAdd:output:0;sequential_20/batch_normalization_34/ReadVariableOp:value:0=sequential_20/batch_normalization_34/ReadVariableOp_1:value:0Lsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_20/batch_normalization_34/AssignNewValueAssignVariableOpMsequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_resourceBsequential_20/batch_normalization_34/FusedBatchNormV3:batch_mean:0E^sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_20/batch_normalization_34/AssignNewValue_1AssignVariableOpOsequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resourceFsequential_20/batch_normalization_34/FusedBatchNormV3:batch_variance:0G^sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_20/leaky_re_lu_33/LeakyRelu	LeakyRelu9sequential_20/batch_normalization_34/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_21/conv2d_29/Conv2D/ReadVariableOpReadVariableOp6sequential_21_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_21/conv2d_29/Conv2DConv2D4sequential_20/leaky_re_lu_33/LeakyRelu:activations:05sequential_21/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_21/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_21/conv2d_29/BiasAddBiasAdd'sequential_21/conv2d_29/Conv2D:output:06sequential_21/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_21/batch_normalization_35/ReadVariableOpReadVariableOp<sequential_21_batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_21/batch_normalization_35/ReadVariableOp_1ReadVariableOp>sequential_21_batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_21/batch_normalization_35/FusedBatchNormV3FusedBatchNormV3(sequential_21/conv2d_29/BiasAdd:output:0;sequential_21/batch_normalization_35/ReadVariableOp:value:0=sequential_21/batch_normalization_35/ReadVariableOp_1:value:0Lsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_21/batch_normalization_35/AssignNewValueAssignVariableOpMsequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_resourceBsequential_21/batch_normalization_35/FusedBatchNormV3:batch_mean:0E^sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_21/batch_normalization_35/AssignNewValue_1AssignVariableOpOsequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resourceFsequential_21/batch_normalization_35/FusedBatchNormV3:batch_variance:0G^sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
"sequential_21/activation_1/SigmoidSigmoid9sequential_21/batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????00}
IdentityIdentity&sequential_21/activation_1/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?%
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp4^sequential_15/batch_normalization_26/AssignNewValue6^sequential_15/batch_normalization_26/AssignNewValue_1E^sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpG^sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_14^sequential_15/batch_normalization_26/ReadVariableOp6^sequential_15/batch_normalization_26/ReadVariableOp_18^sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOpA^sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp4^sequential_16/batch_normalization_27/AssignNewValue6^sequential_16/batch_normalization_27/AssignNewValue_1E^sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_27/ReadVariableOp6^sequential_16/batch_normalization_27/ReadVariableOp_14^sequential_16/batch_normalization_28/AssignNewValue6^sequential_16/batch_normalization_28/AssignNewValue_1E^sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_28/ReadVariableOp6^sequential_16/batch_normalization_28/ReadVariableOp_1/^sequential_16/conv2d_23/BiasAdd/ReadVariableOp.^sequential_16/conv2d_23/Conv2D/ReadVariableOp/^sequential_16/conv2d_24/BiasAdd/ReadVariableOp.^sequential_16/conv2d_24/Conv2D/ReadVariableOp4^sequential_17/batch_normalization_29/AssignNewValue6^sequential_17/batch_normalization_29/AssignNewValue_1E^sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_29/ReadVariableOp6^sequential_17/batch_normalization_29/ReadVariableOp_18^sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOpA^sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp4^sequential_18/batch_normalization_30/AssignNewValue6^sequential_18/batch_normalization_30/AssignNewValue_1E^sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpG^sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_14^sequential_18/batch_normalization_30/ReadVariableOp6^sequential_18/batch_normalization_30/ReadVariableOp_14^sequential_18/batch_normalization_31/AssignNewValue6^sequential_18/batch_normalization_31/AssignNewValue_1E^sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpG^sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_14^sequential_18/batch_normalization_31/ReadVariableOp6^sequential_18/batch_normalization_31/ReadVariableOp_1/^sequential_18/conv2d_25/BiasAdd/ReadVariableOp.^sequential_18/conv2d_25/Conv2D/ReadVariableOp/^sequential_18/conv2d_26/BiasAdd/ReadVariableOp.^sequential_18/conv2d_26/Conv2D/ReadVariableOp4^sequential_19/batch_normalization_32/AssignNewValue6^sequential_19/batch_normalization_32/AssignNewValue_1E^sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpG^sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_14^sequential_19/batch_normalization_32/ReadVariableOp6^sequential_19/batch_normalization_32/ReadVariableOp_18^sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOpA^sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp4^sequential_20/batch_normalization_33/AssignNewValue6^sequential_20/batch_normalization_33/AssignNewValue_1E^sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpG^sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_14^sequential_20/batch_normalization_33/ReadVariableOp6^sequential_20/batch_normalization_33/ReadVariableOp_14^sequential_20/batch_normalization_34/AssignNewValue6^sequential_20/batch_normalization_34/AssignNewValue_1E^sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpG^sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_14^sequential_20/batch_normalization_34/ReadVariableOp6^sequential_20/batch_normalization_34/ReadVariableOp_1/^sequential_20/conv2d_27/BiasAdd/ReadVariableOp.^sequential_20/conv2d_27/Conv2D/ReadVariableOp/^sequential_20/conv2d_28/BiasAdd/ReadVariableOp.^sequential_20/conv2d_28/Conv2D/ReadVariableOp4^sequential_21/batch_normalization_35/AssignNewValue6^sequential_21/batch_normalization_35/AssignNewValue_1E^sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpG^sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_14^sequential_21/batch_normalization_35/ReadVariableOp6^sequential_21/batch_normalization_35/ReadVariableOp_1/^sequential_21/conv2d_29/BiasAdd/ReadVariableOp.^sequential_21/conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2j
3sequential_15/batch_normalization_26/AssignNewValue3sequential_15/batch_normalization_26/AssignNewValue2n
5sequential_15/batch_normalization_26/AssignNewValue_15sequential_15/batch_normalization_26/AssignNewValue_12?
Dsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpDsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Fsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Fsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12j
3sequential_15/batch_normalization_26/ReadVariableOp3sequential_15/batch_normalization_26/ReadVariableOp2n
5sequential_15/batch_normalization_26/ReadVariableOp_15sequential_15/batch_normalization_26/ReadVariableOp_12r
7sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp7sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp2?
@sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp@sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2j
3sequential_16/batch_normalization_27/AssignNewValue3sequential_16/batch_normalization_27/AssignNewValue2n
5sequential_16/batch_normalization_27/AssignNewValue_15sequential_16/batch_normalization_27/AssignNewValue_12?
Dsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Fsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_27/ReadVariableOp3sequential_16/batch_normalization_27/ReadVariableOp2n
5sequential_16/batch_normalization_27/ReadVariableOp_15sequential_16/batch_normalization_27/ReadVariableOp_12j
3sequential_16/batch_normalization_28/AssignNewValue3sequential_16/batch_normalization_28/AssignNewValue2n
5sequential_16/batch_normalization_28/AssignNewValue_15sequential_16/batch_normalization_28/AssignNewValue_12?
Dsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Fsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_28/ReadVariableOp3sequential_16/batch_normalization_28/ReadVariableOp2n
5sequential_16/batch_normalization_28/ReadVariableOp_15sequential_16/batch_normalization_28/ReadVariableOp_12`
.sequential_16/conv2d_23/BiasAdd/ReadVariableOp.sequential_16/conv2d_23/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_23/Conv2D/ReadVariableOp-sequential_16/conv2d_23/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_24/BiasAdd/ReadVariableOp.sequential_16/conv2d_24/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_24/Conv2D/ReadVariableOp-sequential_16/conv2d_24/Conv2D/ReadVariableOp2j
3sequential_17/batch_normalization_29/AssignNewValue3sequential_17/batch_normalization_29/AssignNewValue2n
5sequential_17/batch_normalization_29/AssignNewValue_15sequential_17/batch_normalization_29/AssignNewValue_12?
Dsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp2?
Fsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_29/ReadVariableOp3sequential_17/batch_normalization_29/ReadVariableOp2n
5sequential_17/batch_normalization_29/ReadVariableOp_15sequential_17/batch_normalization_29/ReadVariableOp_12r
7sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp7sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp2?
@sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp@sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2j
3sequential_18/batch_normalization_30/AssignNewValue3sequential_18/batch_normalization_30/AssignNewValue2n
5sequential_18/batch_normalization_30/AssignNewValue_15sequential_18/batch_normalization_30/AssignNewValue_12?
Dsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpDsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp2?
Fsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1Fsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12j
3sequential_18/batch_normalization_30/ReadVariableOp3sequential_18/batch_normalization_30/ReadVariableOp2n
5sequential_18/batch_normalization_30/ReadVariableOp_15sequential_18/batch_normalization_30/ReadVariableOp_12j
3sequential_18/batch_normalization_31/AssignNewValue3sequential_18/batch_normalization_31/AssignNewValue2n
5sequential_18/batch_normalization_31/AssignNewValue_15sequential_18/batch_normalization_31/AssignNewValue_12?
Dsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpDsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp2?
Fsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1Fsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12j
3sequential_18/batch_normalization_31/ReadVariableOp3sequential_18/batch_normalization_31/ReadVariableOp2n
5sequential_18/batch_normalization_31/ReadVariableOp_15sequential_18/batch_normalization_31/ReadVariableOp_12`
.sequential_18/conv2d_25/BiasAdd/ReadVariableOp.sequential_18/conv2d_25/BiasAdd/ReadVariableOp2^
-sequential_18/conv2d_25/Conv2D/ReadVariableOp-sequential_18/conv2d_25/Conv2D/ReadVariableOp2`
.sequential_18/conv2d_26/BiasAdd/ReadVariableOp.sequential_18/conv2d_26/BiasAdd/ReadVariableOp2^
-sequential_18/conv2d_26/Conv2D/ReadVariableOp-sequential_18/conv2d_26/Conv2D/ReadVariableOp2j
3sequential_19/batch_normalization_32/AssignNewValue3sequential_19/batch_normalization_32/AssignNewValue2n
5sequential_19/batch_normalization_32/AssignNewValue_15sequential_19/batch_normalization_32/AssignNewValue_12?
Dsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpDsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2?
Fsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Fsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12j
3sequential_19/batch_normalization_32/ReadVariableOp3sequential_19/batch_normalization_32/ReadVariableOp2n
5sequential_19/batch_normalization_32/ReadVariableOp_15sequential_19/batch_normalization_32/ReadVariableOp_12r
7sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp7sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp2?
@sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp@sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2j
3sequential_20/batch_normalization_33/AssignNewValue3sequential_20/batch_normalization_33/AssignNewValue2n
5sequential_20/batch_normalization_33/AssignNewValue_15sequential_20/batch_normalization_33/AssignNewValue_12?
Dsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpDsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2?
Fsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Fsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12j
3sequential_20/batch_normalization_33/ReadVariableOp3sequential_20/batch_normalization_33/ReadVariableOp2n
5sequential_20/batch_normalization_33/ReadVariableOp_15sequential_20/batch_normalization_33/ReadVariableOp_12j
3sequential_20/batch_normalization_34/AssignNewValue3sequential_20/batch_normalization_34/AssignNewValue2n
5sequential_20/batch_normalization_34/AssignNewValue_15sequential_20/batch_normalization_34/AssignNewValue_12?
Dsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpDsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2?
Fsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Fsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12j
3sequential_20/batch_normalization_34/ReadVariableOp3sequential_20/batch_normalization_34/ReadVariableOp2n
5sequential_20/batch_normalization_34/ReadVariableOp_15sequential_20/batch_normalization_34/ReadVariableOp_12`
.sequential_20/conv2d_27/BiasAdd/ReadVariableOp.sequential_20/conv2d_27/BiasAdd/ReadVariableOp2^
-sequential_20/conv2d_27/Conv2D/ReadVariableOp-sequential_20/conv2d_27/Conv2D/ReadVariableOp2`
.sequential_20/conv2d_28/BiasAdd/ReadVariableOp.sequential_20/conv2d_28/BiasAdd/ReadVariableOp2^
-sequential_20/conv2d_28/Conv2D/ReadVariableOp-sequential_20/conv2d_28/Conv2D/ReadVariableOp2j
3sequential_21/batch_normalization_35/AssignNewValue3sequential_21/batch_normalization_35/AssignNewValue2n
5sequential_21/batch_normalization_35/AssignNewValue_15sequential_21/batch_normalization_35/AssignNewValue_12?
Dsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpDsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2?
Fsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Fsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12j
3sequential_21/batch_normalization_35/ReadVariableOp3sequential_21/batch_normalization_35/ReadVariableOp2n
5sequential_21/batch_normalization_35/ReadVariableOp_15sequential_21/batch_normalization_35/ReadVariableOp_12`
.sequential_21/conv2d_29/BiasAdd/ReadVariableOp.sequential_21/conv2d_29/BiasAdd/ReadVariableOp2^
-sequential_21/conv2d_29/Conv2D/ReadVariableOp-sequential_21/conv2d_29/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_16_layer_call_fn_62087

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_57894w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64338

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
-__inference_sequential_21_layer_call_fn_62745

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_60224w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_63196

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_63014

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_61929

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
H__inference_sequential_18_layer_call_and_return_conditional_losses_59132
conv2d_25_input)
conv2d_25_59101:
conv2d_25_59103:*
batch_normalization_30_59106:*
batch_normalization_30_59108:*
batch_normalization_30_59110:*
batch_normalization_30_59112:)
conv2d_26_59116:
conv2d_26_59118:*
batch_normalization_31_59121:*
batch_normalization_31_59123:*
batch_normalization_31_59125:*
batch_normalization_31_59127:
identity??.batch_normalization_30/StatefulPartitionedCall?.batch_normalization_31/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputconv2d_25_59101conv2d_25_59103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_58739?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0batch_normalization_30_59106batch_normalization_30_59108batch_normalization_30_59110batch_normalization_30_59112*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58762?
leaky_re_lu_29/PartitionedCallPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_58777?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_29/PartitionedCall:output:0conv2d_26_59116conv2d_26_59118*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_58789?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_31_59121batch_normalization_31_59123batch_normalization_31_59125batch_normalization_31_59127*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58812?
leaky_re_lu_30/PartitionedCallPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_58827~
IdentityIdentity'leaky_re_lu_30/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_25_input
?	
?
-__inference_sequential_17_layer_call_fn_62225

inputs!
unknown: 
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_17_layer_call_and_return_conditional_losses_58412w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_35_layer_call_fn_64424

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60206w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
-__inference_sequential_18_layer_call_fn_58857
conv2d_25_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_18_layer_call_and_return_conditional_losses_58830w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_25_input
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64185

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_35_layer_call_fn_64398

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60124?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_26_layer_call_fn_63704

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_58789w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_28_layer_call_fn_64222

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_59725w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_60407
conv2d_29_input)
conv2d_29_60391:
conv2d_29_60393:*
batch_normalization_35_60396:*
batch_normalization_35_60398:*
batch_normalization_35_60400:*
batch_normalization_35_60402:
identity??.batch_normalization_35/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCallconv2d_29_inputconv2d_29_60391conv2d_29_60393*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_60183?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_35_60396batch_normalization_35_60398batch_normalization_35_60400batch_normalization_35_60402*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60278?
activation_1/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_60221|
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_35/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_29_input
?

?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_64385

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_63996

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?F
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_62728

inputsB
(conv2d_27_conv2d_readvariableop_resource:7
)conv2d_27_biasadd_readvariableop_resource:<
.batch_normalization_33_readvariableop_resource:>
0batch_normalization_33_readvariableop_1_resource:M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_28_conv2d_readvariableop_resource:7
)conv2d_28_biasadd_readvariableop_resource:<
.batch_normalization_34_readvariableop_resource:>
0batch_normalization_34_readvariableop_1_resource:M
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:
identity??%batch_normalization_33/AssignNewValue?'batch_normalization_33/AssignNewValue_1?6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_33/ReadVariableOp?'batch_normalization_33/ReadVariableOp_1?%batch_normalization_34/AssignNewValue?'batch_normalization_34/AssignNewValue_1?6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_34/ReadVariableOp?'batch_normalization_34/ReadVariableOp_1? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3conv2d_27/BiasAdd:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_33/AssignNewValueAssignVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource4batch_normalization_33/FusedBatchNormV3:batch_mean:07^batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_33/AssignNewValue_1AssignVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_33/FusedBatchNormV3:batch_variance:09^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_32/LeakyRelu	LeakyRelu+batch_normalization_33/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_28/Conv2DConv2D&leaky_re_lu_32/LeakyRelu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3conv2d_28/BiasAdd:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_34/AssignNewValueAssignVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource4batch_normalization_34/FusedBatchNormV3:batch_mean:07^batch_normalization_34/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_34/AssignNewValue_1AssignVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_34/FusedBatchNormV3:batch_variance:09^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_33/LeakyRelu	LeakyRelu+batch_normalization_34/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>}
IdentityIdentity&leaky_re_lu_33/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp&^batch_normalization_33/AssignNewValue(^batch_normalization_33/AssignNewValue_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1&^batch_normalization_34/AssignNewValue(^batch_normalization_34/AssignNewValue_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2N
%batch_normalization_33/AssignNewValue%batch_normalization_33/AssignNewValue2R
'batch_normalization_33/AssignNewValue_1'batch_normalization_33/AssignNewValue_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12N
%batch_normalization_34/AssignNewValue%batch_normalization_34/AssignNewValue2R
'batch_normalization_34/AssignNewValue_1'batch_normalization_34/AssignNewValue_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?"
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_58196
conv2d_23_input)
conv2d_23_58165:  
conv2d_23_58167: *
batch_normalization_27_58170: *
batch_normalization_27_58172: *
batch_normalization_27_58174: *
batch_normalization_27_58176: )
conv2d_24_58180:  
conv2d_24_58182: *
batch_normalization_28_58185: *
batch_normalization_28_58187: *
batch_normalization_28_58189: *
batch_normalization_28_58191: 
identity??.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallconv2d_23_inputconv2d_23_58165conv2d_23_58167*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_57803?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_27_58170batch_normalization_27_58172batch_normalization_27_58174batch_normalization_27_58176*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_57826?
leaky_re_lu_26/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_57841?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_26/PartitionedCall:output:0conv2d_24_58180conv2d_24_58182*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_57853?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_28_58185batch_normalization_28_58187batch_normalization_28_58189batch_normalization_28_58191*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57876?
leaky_re_lu_27/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_57891~
IdentityIdentity'leaky_re_lu_27/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall:` \
/
_output_shapes
:????????? 
)
_user_specified_nameconv2d_23_input
?
?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_62890

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
H__inference_sequential_18_layer_call_and_return_conditional_losses_59166
conv2d_25_input)
conv2d_25_59135:
conv2d_25_59137:*
batch_normalization_30_59140:*
batch_normalization_30_59142:*
batch_normalization_30_59144:*
batch_normalization_30_59146:)
conv2d_26_59150:
conv2d_26_59152:*
batch_normalization_31_59155:*
batch_normalization_31_59157:*
batch_normalization_31_59159:*
batch_normalization_31_59161:
identity??.batch_normalization_30/StatefulPartitionedCall?.batch_normalization_31/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputconv2d_25_59135conv2d_25_59137*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_58739?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0batch_normalization_30_59140batch_normalization_30_59142batch_normalization_30_59144batch_normalization_30_59146*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58956?
leaky_re_lu_29/PartitionedCallPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_58777?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_29/PartitionedCall:output:0conv2d_26_59150conv2d_26_59152*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_58789?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_31_59155batch_normalization_31_59157batch_normalization_31_59159batch_normalization_31_59161*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58896?
leaky_re_lu_30/PartitionedCallPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_58827~
IdentityIdentity'leaky_re_lu_30/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_25_input
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_60444

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59698

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
-__inference_sequential_20_layer_call_fn_62607

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_59766w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_32_layer_call_fn_63952

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59267?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_1_layer_call_fn_61934

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_60444h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_60337

inputs)
conv2d_29_60321:
conv2d_29_60323:*
batch_normalization_35_60326:*
batch_normalization_35_60328:*
batch_normalization_35_60330:*
batch_normalization_35_60332:
identity??.batch_normalization_35/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_29_60321conv2d_29_60323*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_60183?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_35_60326batch_normalization_35_60328batch_normalization_35_60330batch_normalization_35_60332*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60278?
activation_1/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_60221|
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_35/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
-__inference_sequential_19_layer_call_fn_62485

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_19_layer_call_and_return_conditional_losses_59348w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_57711

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_59345

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????00*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?8
?
H__inference_sequential_18_layer_call_and_return_conditional_losses_62422

inputsB
(conv2d_25_conv2d_readvariableop_resource:7
)conv2d_25_biasadd_readvariableop_resource:<
.batch_normalization_30_readvariableop_resource:>
0batch_normalization_30_readvariableop_1_resource:M
?batch_normalization_30_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_26_conv2d_readvariableop_resource:7
)conv2d_26_biasadd_readvariableop_resource:<
.batch_normalization_31_readvariableop_resource:>
0batch_normalization_31_readvariableop_1_resource:M
?batch_normalization_31_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:
identity??6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_30/ReadVariableOp?'batch_normalization_30/ReadVariableOp_1?6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_31/ReadVariableOp?'batch_normalization_31/ReadVariableOp_1? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_25/Conv2DConv2Dinputs'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_30/ReadVariableOpReadVariableOp.batch_normalization_30_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_30/ReadVariableOp_1ReadVariableOp0batch_normalization_30_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_30/FusedBatchNormV3FusedBatchNormV3conv2d_25/BiasAdd:output:0-batch_normalization_30/ReadVariableOp:value:0/batch_normalization_30/ReadVariableOp_1:value:0>batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_29/LeakyRelu	LeakyRelu+batch_normalization_30/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_26/Conv2DConv2D&leaky_re_lu_29/LeakyRelu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_31/ReadVariableOpReadVariableOp.batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_31/ReadVariableOp_1ReadVariableOp0batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_31/FusedBatchNormV3FusedBatchNormV3conv2d_26/BiasAdd:output:0-batch_normalization_31/ReadVariableOp:value:0/batch_normalization_31/ReadVariableOp_1:value:0>batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_30/LeakyRelu	LeakyRelu+batch_normalization_31/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>}
IdentityIdentity&leaky_re_lu_30/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp7^batch_normalization_30/FusedBatchNormV3/ReadVariableOp9^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_30/ReadVariableOp(^batch_normalization_30/ReadVariableOp_17^batch_normalization_31/FusedBatchNormV3/ReadVariableOp9^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_31/ReadVariableOp(^batch_normalization_31/ReadVariableOp_1!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2p
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp6batch_normalization_30/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_18batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_30/ReadVariableOp%batch_normalization_30/ReadVariableOp2R
'batch_normalization_30/ReadVariableOp_1'batch_normalization_30/ReadVariableOp_12p
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp6batch_normalization_31/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_18batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_31/ReadVariableOp%batch_normalization_31/ReadVariableOp2R
'batch_normalization_31/ReadVariableOp_1'batch_normalization_31/ReadVariableOp_12D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_32_layer_call_fn_63978

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59402w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_33_layer_call_fn_64131

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59892w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
-__inference_sequential_21_layer_call_fn_60369
conv2d_29_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_29_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_60337w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_29_input
?F
?
B__inference_model_4_layer_call_and_return_conditional_losses_60574

inputs!
dense_1_60425:
??
dense_1_60427:	?-
sequential_15_60446: !
sequential_15_60448: !
sequential_15_60450: !
sequential_15_60452: !
sequential_15_60454: !
sequential_15_60456: -
sequential_16_60459:  !
sequential_16_60461: !
sequential_16_60463: !
sequential_16_60465: !
sequential_16_60467: !
sequential_16_60469: -
sequential_16_60471:  !
sequential_16_60473: !
sequential_16_60475: !
sequential_16_60477: !
sequential_16_60479: !
sequential_16_60481: -
sequential_17_60484: !
sequential_17_60486:!
sequential_17_60488:!
sequential_17_60490:!
sequential_17_60492:!
sequential_17_60494:-
sequential_18_60497:!
sequential_18_60499:!
sequential_18_60501:!
sequential_18_60503:!
sequential_18_60505:!
sequential_18_60507:-
sequential_18_60509:!
sequential_18_60511:!
sequential_18_60513:!
sequential_18_60515:!
sequential_18_60517:!
sequential_18_60519:-
sequential_19_60522:!
sequential_19_60524:!
sequential_19_60526:!
sequential_19_60528:!
sequential_19_60530:!
sequential_19_60532:-
sequential_20_60535:!
sequential_20_60537:!
sequential_20_60539:!
sequential_20_60541:!
sequential_20_60543:!
sequential_20_60545:-
sequential_20_60547:!
sequential_20_60549:!
sequential_20_60551:!
sequential_20_60553:!
sequential_20_60555:!
sequential_20_60557:-
sequential_21_60560:!
sequential_21_60562:!
sequential_21_60564:!
sequential_21_60566:!
sequential_21_60568:!
sequential_21_60570:
identity??dense_1/StatefulPartitionedCall?%sequential_15/StatefulPartitionedCall?%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?%sequential_18/StatefulPartitionedCall?%sequential_19/StatefulPartitionedCall?%sequential_20/StatefulPartitionedCall?%sequential_21/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_60425dense_1_60427*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_60424?
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_60444?
%sequential_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0sequential_15_60446sequential_15_60448sequential_15_60450sequential_15_60452sequential_15_60454sequential_15_60456*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_15_layer_call_and_return_conditional_losses_57476?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCall.sequential_15/StatefulPartitionedCall:output:0sequential_16_60459sequential_16_60461sequential_16_60463sequential_16_60465sequential_16_60467sequential_16_60469sequential_16_60471sequential_16_60473sequential_16_60475sequential_16_60477sequential_16_60479sequential_16_60481*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_57894?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_60484sequential_17_60486sequential_17_60488sequential_17_60490sequential_17_60492sequential_17_60494*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_17_layer_call_and_return_conditional_losses_58412?
%sequential_18/StatefulPartitionedCallStatefulPartitionedCall.sequential_17/StatefulPartitionedCall:output:0sequential_18_60497sequential_18_60499sequential_18_60501sequential_18_60503sequential_18_60505sequential_18_60507sequential_18_60509sequential_18_60511sequential_18_60513sequential_18_60515sequential_18_60517sequential_18_60519*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_18_layer_call_and_return_conditional_losses_58830?
%sequential_19/StatefulPartitionedCallStatefulPartitionedCall.sequential_18/StatefulPartitionedCall:output:0sequential_19_60522sequential_19_60524sequential_19_60526sequential_19_60528sequential_19_60530sequential_19_60532*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_19_layer_call_and_return_conditional_losses_59348?
%sequential_20/StatefulPartitionedCallStatefulPartitionedCall.sequential_19/StatefulPartitionedCall:output:0sequential_20_60535sequential_20_60537sequential_20_60539sequential_20_60541sequential_20_60543sequential_20_60545sequential_20_60547sequential_20_60549sequential_20_60551sequential_20_60553sequential_20_60555sequential_20_60557*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_59766?
%sequential_21/StatefulPartitionedCallStatefulPartitionedCall.sequential_20/StatefulPartitionedCall:output:0sequential_21_60560sequential_21_60562sequential_21_60564sequential_21_60566sequential_21_60568sequential_21_60570*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_60224?
IdentityIdentity.sequential_21/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^dense_1/StatefulPartitionedCall&^sequential_15/StatefulPartitionedCall&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall&^sequential_18/StatefulPartitionedCall&^sequential_19/StatefulPartitionedCall&^sequential_20/StatefulPartitionedCall&^sequential_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%sequential_15/StatefulPartitionedCall%sequential_15/StatefulPartitionedCall2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall2N
%sequential_18/StatefulPartitionedCall%sequential_18/StatefulPartitionedCall2N
%sequential_19/StatefulPartitionedCall%sequential_19/StatefulPartitionedCall2N
%sequential_20/StatefulPartitionedCall%sequential_20/StatefulPartitionedCall2N
%sequential_21/StatefulPartitionedCall%sequential_21/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63320

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
G__inference_activation_1_layer_call_and_return_conditional_losses_60221

inputs
identityT
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????00[
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_5_layer_call_fn_63857

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_59207?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_15_layer_call_and_return_conditional_losses_57476

inputs2
conv2d_transpose_3_57436: &
conv2d_transpose_3_57438: *
batch_normalization_26_57459: *
batch_normalization_26_57461: *
batch_normalization_26_57463: *
batch_normalization_26_57465: 
identity??.batch_normalization_26/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_3_57436conv2d_transpose_3_57438*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_57435?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_26_57459batch_normalization_26_57461batch_normalization_26_57463batch_normalization_26_57465*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57458?
leaky_re_lu_25/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_57473~
IdentityIdentity'leaky_re_lu_25/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_26/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60124

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_60224

inputs)
conv2d_29_60184:
conv2d_29_60186:*
batch_normalization_35_60207:*
batch_normalization_35_60209:*
batch_normalization_35_60211:*
batch_normalization_35_60213:
identity??.batch_normalization_35/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_29_60184conv2d_29_60186*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_60183?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_35_60207batch_normalization_35_60209batch_normalization_35_60211batch_normalization_35_60213*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60206?
activation_1/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_60221|
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_35/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_58789

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_60424

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_5_layer_call_fn_63866

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_59307w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_31_layer_call_fn_63766

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58896w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_57473

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_64079

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
)__inference_conv2d_23_layer_call_fn_63033

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_57803w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
-__inference_sequential_19_layer_call_fn_59492
conv2d_transpose_5_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_19_layer_call_and_return_conditional_losses_59460w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:?????????
2
_user_specified_nameconv2d_transpose_5_input
?	
?
-__inference_sequential_17_layer_call_fn_58427
conv2d_transpose_4_input!
unknown: 
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_17_layer_call_and_return_conditional_losses_58412w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:????????? 
2
_user_specified_nameconv2d_transpose_4_input
?
?
H__inference_sequential_19_layer_call_and_return_conditional_losses_59511
conv2d_transpose_5_input2
conv2d_transpose_5_59495:&
conv2d_transpose_5_59497:*
batch_normalization_32_59500:*
batch_normalization_32_59502:*
batch_normalization_32_59504:*
batch_normalization_32_59506:
identity??.batch_normalization_32/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_5_inputconv2d_transpose_5_59495conv2d_transpose_5_59497*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_59307?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_32_59500batch_normalization_32_59502batch_normalization_32_59504batch_normalization_32_59506*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59330?
leaky_re_lu_31/PartitionedCallPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_59345~
IdentityIdentity'leaky_re_lu_31/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:i e
/
_output_shapes
:?????????
2
_user_specified_nameconv2d_transpose_5_input
?	
?
6__inference_batch_normalization_30_layer_call_fn_63587

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58647?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?!
?
H__inference_sequential_18_layer_call_and_return_conditional_losses_59042

inputs)
conv2d_25_59011:
conv2d_25_59013:*
batch_normalization_30_59016:*
batch_normalization_30_59018:*
batch_normalization_30_59020:*
batch_normalization_30_59022:)
conv2d_26_59026:
conv2d_26_59028:*
batch_normalization_31_59031:*
batch_normalization_31_59033:*
batch_normalization_31_59035:*
batch_normalization_31_59037:
identity??.batch_normalization_30/StatefulPartitionedCall?.batch_normalization_31/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_25_59011conv2d_25_59013*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_58739?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0batch_normalization_30_59016batch_normalization_30_59018batch_normalization_30_59020batch_normalization_30_59022*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58956?
leaky_re_lu_29/PartitionedCallPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_58777?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_29/PartitionedCall:output:0conv2d_26_59026conv2d_26_59028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_58789?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_31_59031batch_normalization_31_59033batch_normalization_31_59035batch_normalization_31_59037*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58896?
leaky_re_lu_30/PartitionedCallPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_58827~
IdentityIdentity'leaky_re_lu_30/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62978

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_61919

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_60424p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_60388
conv2d_29_input)
conv2d_29_60372:
conv2d_29_60374:*
batch_normalization_35_60377:*
batch_normalization_35_60379:*
batch_normalization_35_60381:*
batch_normalization_35_60383:
identity??.batch_normalization_35/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCallconv2d_29_inputconv2d_29_60372conv2d_29_60374*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_60183?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0batch_normalization_35_60377batch_normalization_35_60379batch_normalization_35_60381batch_normalization_35_60383*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60206?
activation_1/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_60221|
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_35/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_29_input
?
?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63266

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?8
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_62162

inputsB
(conv2d_23_conv2d_readvariableop_resource:  7
)conv2d_23_biasadd_readvariableop_resource: <
.batch_normalization_27_readvariableop_resource: >
0batch_normalization_27_readvariableop_1_resource: M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_24_conv2d_readvariableop_resource:  7
)conv2d_24_biasadd_readvariableop_resource: <
.batch_normalization_28_readvariableop_resource: >
0batch_normalization_28_readvariableop_1_resource: M
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource: 
identity??6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_23/Conv2DConv2Dinputs'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_23/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
leaky_re_lu_26/LeakyRelu	LeakyRelu+batch_normalization_27/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_24/Conv2DConv2D&leaky_re_lu_26/LeakyRelu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_24/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
leaky_re_lu_27/LeakyRelu	LeakyRelu+batch_normalization_28/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>}
IdentityIdentity&leaky_re_lu_27/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp7^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_16_layer_call_fn_57921
conv2d_23_input!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_23_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_57894w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:????????? 
)
_user_specified_nameconv2d_23_input
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64473

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
H__inference_sequential_17_layer_call_and_return_conditional_losses_62280

inputsU
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_4_biasadd_readvariableop_resource:<
.batch_normalization_29_readvariableop_resource:>
0batch_normalization_29_readvariableop_1_resource:M
?batch_normalization_29_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:
identity??6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_29/ReadVariableOp?'batch_normalization_29/ReadVariableOp_1?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOpN
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_29/ReadVariableOpReadVariableOp.batch_normalization_29_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_29/ReadVariableOp_1ReadVariableOp0batch_normalization_29_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_29/FusedBatchNormV3FusedBatchNormV3#conv2d_transpose_4/BiasAdd:output:0-batch_normalization_29/ReadVariableOp:value:0/batch_normalization_29/ReadVariableOp_1:value:0>batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_28/LeakyRelu	LeakyRelu+batch_normalization_29/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>}
IdentityIdentity&leaky_re_lu_28/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp7^batch_normalization_29/FusedBatchNormV3/ReadVariableOp9^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_29/ReadVariableOp(^batch_normalization_29/ReadVariableOp_1*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 2p
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp6batch_normalization_29/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_18batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_29/ReadVariableOp%batch_normalization_29/ReadVariableOp2R
'batch_normalization_29/ReadVariableOp_1'batch_normalization_29/ReadVariableOp_12V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?F
?
B__inference_model_4_layer_call_and_return_conditional_losses_60856

inputs!
dense_1_60722:
??
dense_1_60724:	?-
sequential_15_60728: !
sequential_15_60730: !
sequential_15_60732: !
sequential_15_60734: !
sequential_15_60736: !
sequential_15_60738: -
sequential_16_60741:  !
sequential_16_60743: !
sequential_16_60745: !
sequential_16_60747: !
sequential_16_60749: !
sequential_16_60751: -
sequential_16_60753:  !
sequential_16_60755: !
sequential_16_60757: !
sequential_16_60759: !
sequential_16_60761: !
sequential_16_60763: -
sequential_17_60766: !
sequential_17_60768:!
sequential_17_60770:!
sequential_17_60772:!
sequential_17_60774:!
sequential_17_60776:-
sequential_18_60779:!
sequential_18_60781:!
sequential_18_60783:!
sequential_18_60785:!
sequential_18_60787:!
sequential_18_60789:-
sequential_18_60791:!
sequential_18_60793:!
sequential_18_60795:!
sequential_18_60797:!
sequential_18_60799:!
sequential_18_60801:-
sequential_19_60804:!
sequential_19_60806:!
sequential_19_60808:!
sequential_19_60810:!
sequential_19_60812:!
sequential_19_60814:-
sequential_20_60817:!
sequential_20_60819:!
sequential_20_60821:!
sequential_20_60823:!
sequential_20_60825:!
sequential_20_60827:-
sequential_20_60829:!
sequential_20_60831:!
sequential_20_60833:!
sequential_20_60835:!
sequential_20_60837:!
sequential_20_60839:-
sequential_21_60842:!
sequential_21_60844:!
sequential_21_60846:!
sequential_21_60848:!
sequential_21_60850:!
sequential_21_60852:
identity??dense_1/StatefulPartitionedCall?%sequential_15/StatefulPartitionedCall?%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?%sequential_18/StatefulPartitionedCall?%sequential_19/StatefulPartitionedCall?%sequential_20/StatefulPartitionedCall?%sequential_21/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_60722dense_1_60724*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_60424?
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_60444?
%sequential_15/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0sequential_15_60728sequential_15_60730sequential_15_60732sequential_15_60734sequential_15_60736sequential_15_60738*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_15_layer_call_and_return_conditional_losses_57588?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCall.sequential_15/StatefulPartitionedCall:output:0sequential_16_60741sequential_16_60743sequential_16_60745sequential_16_60747sequential_16_60749sequential_16_60751sequential_16_60753sequential_16_60755sequential_16_60757sequential_16_60759sequential_16_60761sequential_16_60763*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_58106?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_60766sequential_17_60768sequential_17_60770sequential_17_60772sequential_17_60774sequential_17_60776*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_17_layer_call_and_return_conditional_losses_58524?
%sequential_18/StatefulPartitionedCallStatefulPartitionedCall.sequential_17/StatefulPartitionedCall:output:0sequential_18_60779sequential_18_60781sequential_18_60783sequential_18_60785sequential_18_60787sequential_18_60789sequential_18_60791sequential_18_60793sequential_18_60795sequential_18_60797sequential_18_60799sequential_18_60801*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_18_layer_call_and_return_conditional_losses_59042?
%sequential_19/StatefulPartitionedCallStatefulPartitionedCall.sequential_18/StatefulPartitionedCall:output:0sequential_19_60804sequential_19_60806sequential_19_60808sequential_19_60810sequential_19_60812sequential_19_60814*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_19_layer_call_and_return_conditional_losses_59460?
%sequential_20/StatefulPartitionedCallStatefulPartitionedCall.sequential_19/StatefulPartitionedCall:output:0sequential_20_60817sequential_20_60819sequential_20_60821sequential_20_60823sequential_20_60825sequential_20_60827sequential_20_60829sequential_20_60831sequential_20_60833sequential_20_60835sequential_20_60837sequential_20_60839*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_59978?
%sequential_21/StatefulPartitionedCallStatefulPartitionedCall.sequential_20/StatefulPartitionedCall:output:0sequential_21_60842sequential_21_60844sequential_21_60846sequential_21_60848sequential_21_60850sequential_21_60852*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_60337?
IdentityIdentity.sequential_21/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^dense_1/StatefulPartitionedCall&^sequential_15/StatefulPartitionedCall&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall&^sequential_18/StatefulPartitionedCall&^sequential_19/StatefulPartitionedCall&^sequential_20/StatefulPartitionedCall&^sequential_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%sequential_15/StatefulPartitionedCall%sequential_15/StatefulPartitionedCall2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall2N
%sequential_18/StatefulPartitionedCall%sequential_18/StatefulPartitionedCall2N
%sequential_19/StatefulPartitionedCall%sequential_19/StatefulPartitionedCall2N
%sequential_20/StatefulPartitionedCall%sequential_20/StatefulPartitionedCall2N
%sequential_21/StatefulPartitionedCall%sequential_21/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ɚ
?E
B__inference_model_4_layer_call_and_return_conditional_losses_61641

inputs:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?c
Isequential_15_conv2d_transpose_3_conv2d_transpose_readvariableop_resource: N
@sequential_15_conv2d_transpose_3_biasadd_readvariableop_resource: J
<sequential_15_batch_normalization_26_readvariableop_resource: L
>sequential_15_batch_normalization_26_readvariableop_1_resource: [
Msequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_resource: ]
Osequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource: P
6sequential_16_conv2d_23_conv2d_readvariableop_resource:  E
7sequential_16_conv2d_23_biasadd_readvariableop_resource: J
<sequential_16_batch_normalization_27_readvariableop_resource: L
>sequential_16_batch_normalization_27_readvariableop_1_resource: [
Msequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_resource: ]
Osequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource: P
6sequential_16_conv2d_24_conv2d_readvariableop_resource:  E
7sequential_16_conv2d_24_biasadd_readvariableop_resource: J
<sequential_16_batch_normalization_28_readvariableop_resource: L
>sequential_16_batch_normalization_28_readvariableop_1_resource: [
Msequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_resource: ]
Osequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource: c
Isequential_17_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: N
@sequential_17_conv2d_transpose_4_biasadd_readvariableop_resource:J
<sequential_17_batch_normalization_29_readvariableop_resource:L
>sequential_17_batch_normalization_29_readvariableop_1_resource:[
Msequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_resource:]
Osequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_18_conv2d_25_conv2d_readvariableop_resource:E
7sequential_18_conv2d_25_biasadd_readvariableop_resource:J
<sequential_18_batch_normalization_30_readvariableop_resource:L
>sequential_18_batch_normalization_30_readvariableop_1_resource:[
Msequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_resource:]
Osequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_18_conv2d_26_conv2d_readvariableop_resource:E
7sequential_18_conv2d_26_biasadd_readvariableop_resource:J
<sequential_18_batch_normalization_31_readvariableop_resource:L
>sequential_18_batch_normalization_31_readvariableop_1_resource:[
Msequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_resource:]
Osequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:c
Isequential_19_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:N
@sequential_19_conv2d_transpose_5_biasadd_readvariableop_resource:J
<sequential_19_batch_normalization_32_readvariableop_resource:L
>sequential_19_batch_normalization_32_readvariableop_1_resource:[
Msequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:]
Osequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_20_conv2d_27_conv2d_readvariableop_resource:E
7sequential_20_conv2d_27_biasadd_readvariableop_resource:J
<sequential_20_batch_normalization_33_readvariableop_resource:L
>sequential_20_batch_normalization_33_readvariableop_1_resource:[
Msequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:]
Osequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_20_conv2d_28_conv2d_readvariableop_resource:E
7sequential_20_conv2d_28_biasadd_readvariableop_resource:J
<sequential_20_batch_normalization_34_readvariableop_resource:L
>sequential_20_batch_normalization_34_readvariableop_1_resource:[
Msequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:]
Osequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_21_conv2d_29_conv2d_readvariableop_resource:E
7sequential_21_conv2d_29_biasadd_readvariableop_resource:J
<sequential_21_batch_normalization_35_readvariableop_resource:L
>sequential_21_batch_normalization_35_readvariableop_1_resource:[
Msequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_resource:]
Osequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?Dsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Fsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?3sequential_15/batch_normalization_26/ReadVariableOp?5sequential_15/batch_normalization_26/ReadVariableOp_1?7sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp?@sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?Dsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Fsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?3sequential_16/batch_normalization_27/ReadVariableOp?5sequential_16/batch_normalization_27/ReadVariableOp_1?Dsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Fsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?3sequential_16/batch_normalization_28/ReadVariableOp?5sequential_16/batch_normalization_28/ReadVariableOp_1?.sequential_16/conv2d_23/BiasAdd/ReadVariableOp?-sequential_16/conv2d_23/Conv2D/ReadVariableOp?.sequential_16/conv2d_24/BiasAdd/ReadVariableOp?-sequential_16/conv2d_24/Conv2D/ReadVariableOp?Dsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?Fsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?3sequential_17/batch_normalization_29/ReadVariableOp?5sequential_17/batch_normalization_29/ReadVariableOp_1?7sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp?@sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?Dsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?Fsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?3sequential_18/batch_normalization_30/ReadVariableOp?5sequential_18/batch_normalization_30/ReadVariableOp_1?Dsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?Fsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?3sequential_18/batch_normalization_31/ReadVariableOp?5sequential_18/batch_normalization_31/ReadVariableOp_1?.sequential_18/conv2d_25/BiasAdd/ReadVariableOp?-sequential_18/conv2d_25/Conv2D/ReadVariableOp?.sequential_18/conv2d_26/BiasAdd/ReadVariableOp?-sequential_18/conv2d_26/Conv2D/ReadVariableOp?Dsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?Fsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?3sequential_19/batch_normalization_32/ReadVariableOp?5sequential_19/batch_normalization_32/ReadVariableOp_1?7sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp?@sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?Dsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?Fsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?3sequential_20/batch_normalization_33/ReadVariableOp?5sequential_20/batch_normalization_33/ReadVariableOp_1?Dsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp?Fsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?3sequential_20/batch_normalization_34/ReadVariableOp?5sequential_20/batch_normalization_34/ReadVariableOp_1?.sequential_20/conv2d_27/BiasAdd/ReadVariableOp?-sequential_20/conv2d_27/Conv2D/ReadVariableOp?.sequential_20/conv2d_28/BiasAdd/ReadVariableOp?-sequential_20/conv2d_28/Conv2D/ReadVariableOp?Dsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp?Fsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?3sequential_21/batch_normalization_35/ReadVariableOp?5sequential_21/batch_normalization_35/ReadVariableOp_1?.sequential_21/conv2d_29/BiasAdd/ReadVariableOp?-sequential_21/conv2d_29/Conv2D/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0z
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????W
reshape_1/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_1/ReshapeReshapedense_1/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????p
&sequential_15/conv2d_transpose_3/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:~
4sequential_15/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_15/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_15/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_15/conv2d_transpose_3/strided_sliceStridedSlice/sequential_15/conv2d_transpose_3/Shape:output:0=sequential_15/conv2d_transpose_3/strided_slice/stack:output:0?sequential_15/conv2d_transpose_3/strided_slice/stack_1:output:0?sequential_15/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_15/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_15/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_15/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_15/conv2d_transpose_3/stackPack7sequential_15/conv2d_transpose_3/strided_slice:output:01sequential_15/conv2d_transpose_3/stack/1:output:01sequential_15/conv2d_transpose_3/stack/2:output:01sequential_15/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_15/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_15/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_15/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_15/conv2d_transpose_3/strided_slice_1StridedSlice/sequential_15/conv2d_transpose_3/stack:output:0?sequential_15/conv2d_transpose_3/strided_slice_1/stack:output:0Asequential_15/conv2d_transpose_3/strided_slice_1/stack_1:output:0Asequential_15/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_15_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
1sequential_15/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput/sequential_15/conv2d_transpose_3/stack:output:0Hsequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
7sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp@sequential_15_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
(sequential_15/conv2d_transpose_3/BiasAddBiasAdd:sequential_15/conv2d_transpose_3/conv2d_transpose:output:0?sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_15/batch_normalization_26/ReadVariableOpReadVariableOp<sequential_15_batch_normalization_26_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_15/batch_normalization_26/ReadVariableOp_1ReadVariableOp>sequential_15_batch_normalization_26_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_15/batch_normalization_26/FusedBatchNormV3FusedBatchNormV31sequential_15/conv2d_transpose_3/BiasAdd:output:0;sequential_15/batch_normalization_26/ReadVariableOp:value:0=sequential_15/batch_normalization_26/ReadVariableOp_1:value:0Lsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
&sequential_15/leaky_re_lu_25/LeakyRelu	LeakyRelu9sequential_15/batch_normalization_26/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
-sequential_16/conv2d_23/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_16/conv2d_23/Conv2DConv2D4sequential_15/leaky_re_lu_25/LeakyRelu:activations:05sequential_16/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_16/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_16/conv2d_23/BiasAddBiasAdd'sequential_16/conv2d_23/Conv2D:output:06sequential_16/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_16/batch_normalization_27/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_16/batch_normalization_27/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_16/batch_normalization_27/FusedBatchNormV3FusedBatchNormV3(sequential_16/conv2d_23/BiasAdd:output:0;sequential_16/batch_normalization_27/ReadVariableOp:value:0=sequential_16/batch_normalization_27/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
&sequential_16/leaky_re_lu_26/LeakyRelu	LeakyRelu9sequential_16/batch_normalization_27/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
-sequential_16/conv2d_24/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_16/conv2d_24/Conv2DConv2D4sequential_16/leaky_re_lu_26/LeakyRelu:activations:05sequential_16/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_16/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_16/conv2d_24/BiasAddBiasAdd'sequential_16/conv2d_24/Conv2D:output:06sequential_16/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_16/batch_normalization_28/ReadVariableOpReadVariableOp<sequential_16_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_16/batch_normalization_28/ReadVariableOp_1ReadVariableOp>sequential_16_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_16/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3(sequential_16/conv2d_24/BiasAdd:output:0;sequential_16/batch_normalization_28/ReadVariableOp:value:0=sequential_16/batch_normalization_28/ReadVariableOp_1:value:0Lsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
&sequential_16/leaky_re_lu_27/LeakyRelu	LeakyRelu9sequential_16/batch_normalization_28/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
&sequential_17/conv2d_transpose_4/ShapeShape4sequential_16/leaky_re_lu_27/LeakyRelu:activations:0*
T0*
_output_shapes
:~
4sequential_17/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_17/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_17/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_17/conv2d_transpose_4/strided_sliceStridedSlice/sequential_17/conv2d_transpose_4/Shape:output:0=sequential_17/conv2d_transpose_4/strided_slice/stack:output:0?sequential_17/conv2d_transpose_4/strided_slice/stack_1:output:0?sequential_17/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_17/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_17/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_17/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
&sequential_17/conv2d_transpose_4/stackPack7sequential_17/conv2d_transpose_4/strided_slice:output:01sequential_17/conv2d_transpose_4/stack/1:output:01sequential_17/conv2d_transpose_4/stack/2:output:01sequential_17/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_17/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_17/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_17/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_17/conv2d_transpose_4/strided_slice_1StridedSlice/sequential_17/conv2d_transpose_4/stack:output:0?sequential_17/conv2d_transpose_4/strided_slice_1/stack:output:0Asequential_17/conv2d_transpose_4/strided_slice_1/stack_1:output:0Asequential_17/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_17_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
1sequential_17/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput/sequential_17/conv2d_transpose_4/stack:output:0Hsequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:04sequential_16/leaky_re_lu_27/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
7sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp@sequential_17_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_17/conv2d_transpose_4/BiasAddBiasAdd:sequential_17/conv2d_transpose_4/conv2d_transpose:output:0?sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
3sequential_17/batch_normalization_29/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_29_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_17/batch_normalization_29/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_29_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_17/batch_normalization_29/FusedBatchNormV3FusedBatchNormV31sequential_17/conv2d_transpose_4/BiasAdd:output:0;sequential_17/batch_normalization_29/ReadVariableOp:value:0=sequential_17/batch_normalization_29/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
&sequential_17/leaky_re_lu_28/LeakyRelu	LeakyRelu9sequential_17/batch_normalization_29/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
-sequential_18/conv2d_25/Conv2D/ReadVariableOpReadVariableOp6sequential_18_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_18/conv2d_25/Conv2DConv2D4sequential_17/leaky_re_lu_28/LeakyRelu:activations:05sequential_18/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
.sequential_18/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp7sequential_18_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_18/conv2d_25/BiasAddBiasAdd'sequential_18/conv2d_25/Conv2D:output:06sequential_18/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
3sequential_18/batch_normalization_30/ReadVariableOpReadVariableOp<sequential_18_batch_normalization_30_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_18/batch_normalization_30/ReadVariableOp_1ReadVariableOp>sequential_18_batch_normalization_30_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_18/batch_normalization_30/FusedBatchNormV3FusedBatchNormV3(sequential_18/conv2d_25/BiasAdd:output:0;sequential_18/batch_normalization_30/ReadVariableOp:value:0=sequential_18/batch_normalization_30/ReadVariableOp_1:value:0Lsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
&sequential_18/leaky_re_lu_29/LeakyRelu	LeakyRelu9sequential_18/batch_normalization_30/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
-sequential_18/conv2d_26/Conv2D/ReadVariableOpReadVariableOp6sequential_18_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_18/conv2d_26/Conv2DConv2D4sequential_18/leaky_re_lu_29/LeakyRelu:activations:05sequential_18/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
.sequential_18/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp7sequential_18_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_18/conv2d_26/BiasAddBiasAdd'sequential_18/conv2d_26/Conv2D:output:06sequential_18/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
3sequential_18/batch_normalization_31/ReadVariableOpReadVariableOp<sequential_18_batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_18/batch_normalization_31/ReadVariableOp_1ReadVariableOp>sequential_18_batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_18/batch_normalization_31/FusedBatchNormV3FusedBatchNormV3(sequential_18/conv2d_26/BiasAdd:output:0;sequential_18/batch_normalization_31/ReadVariableOp:value:0=sequential_18/batch_normalization_31/ReadVariableOp_1:value:0Lsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
&sequential_18/leaky_re_lu_30/LeakyRelu	LeakyRelu9sequential_18/batch_normalization_31/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
&sequential_19/conv2d_transpose_5/ShapeShape4sequential_18/leaky_re_lu_30/LeakyRelu:activations:0*
T0*
_output_shapes
:~
4sequential_19/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6sequential_19/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential_19/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.sequential_19/conv2d_transpose_5/strided_sliceStridedSlice/sequential_19/conv2d_transpose_5/Shape:output:0=sequential_19/conv2d_transpose_5/strided_slice/stack:output:0?sequential_19/conv2d_transpose_5/strided_slice/stack_1:output:0?sequential_19/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_19/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0j
(sequential_19/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0j
(sequential_19/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
&sequential_19/conv2d_transpose_5/stackPack7sequential_19/conv2d_transpose_5/strided_slice:output:01sequential_19/conv2d_transpose_5/stack/1:output:01sequential_19/conv2d_transpose_5/stack/2:output:01sequential_19/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:?
6sequential_19/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8sequential_19/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential_19/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0sequential_19/conv2d_transpose_5/strided_slice_1StridedSlice/sequential_19/conv2d_transpose_5/stack:output:0?sequential_19/conv2d_transpose_5/strided_slice_1/stack:output:0Asequential_19/conv2d_transpose_5/strided_slice_1/stack_1:output:0Asequential_19/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
@sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_19_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
1sequential_19/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput/sequential_19/conv2d_transpose_5/stack:output:0Hsequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:04sequential_18/leaky_re_lu_30/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00*
paddingVALID*
strides
?
7sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp@sequential_19_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_19/conv2d_transpose_5/BiasAddBiasAdd:sequential_19/conv2d_transpose_5/conv2d_transpose:output:0?sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_19/batch_normalization_32/ReadVariableOpReadVariableOp<sequential_19_batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_19/batch_normalization_32/ReadVariableOp_1ReadVariableOp>sequential_19_batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_19/batch_normalization_32/FusedBatchNormV3FusedBatchNormV31sequential_19/conv2d_transpose_5/BiasAdd:output:0;sequential_19/batch_normalization_32/ReadVariableOp:value:0=sequential_19/batch_normalization_32/ReadVariableOp_1:value:0Lsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
&sequential_19/leaky_re_lu_31/LeakyRelu	LeakyRelu9sequential_19/batch_normalization_32/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_20/conv2d_27/Conv2D/ReadVariableOpReadVariableOp6sequential_20_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_20/conv2d_27/Conv2DConv2D4sequential_19/leaky_re_lu_31/LeakyRelu:activations:05sequential_20/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_20/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp7sequential_20_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_20/conv2d_27/BiasAddBiasAdd'sequential_20/conv2d_27/Conv2D:output:06sequential_20/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_20/batch_normalization_33/ReadVariableOpReadVariableOp<sequential_20_batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_20/batch_normalization_33/ReadVariableOp_1ReadVariableOp>sequential_20_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_20/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3(sequential_20/conv2d_27/BiasAdd:output:0;sequential_20/batch_normalization_33/ReadVariableOp:value:0=sequential_20/batch_normalization_33/ReadVariableOp_1:value:0Lsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
&sequential_20/leaky_re_lu_32/LeakyRelu	LeakyRelu9sequential_20/batch_normalization_33/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_20/conv2d_28/Conv2D/ReadVariableOpReadVariableOp6sequential_20_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_20/conv2d_28/Conv2DConv2D4sequential_20/leaky_re_lu_32/LeakyRelu:activations:05sequential_20/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_20/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp7sequential_20_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_20/conv2d_28/BiasAddBiasAdd'sequential_20/conv2d_28/Conv2D:output:06sequential_20/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_20/batch_normalization_34/ReadVariableOpReadVariableOp<sequential_20_batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_20/batch_normalization_34/ReadVariableOp_1ReadVariableOp>sequential_20_batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_20/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3(sequential_20/conv2d_28/BiasAdd:output:0;sequential_20/batch_normalization_34/ReadVariableOp:value:0=sequential_20/batch_normalization_34/ReadVariableOp_1:value:0Lsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
&sequential_20/leaky_re_lu_33/LeakyRelu	LeakyRelu9sequential_20/batch_normalization_34/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_21/conv2d_29/Conv2D/ReadVariableOpReadVariableOp6sequential_21_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_21/conv2d_29/Conv2DConv2D4sequential_20/leaky_re_lu_33/LeakyRelu:activations:05sequential_21/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_21/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_21/conv2d_29/BiasAddBiasAdd'sequential_21/conv2d_29/Conv2D:output:06sequential_21/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_21/batch_normalization_35/ReadVariableOpReadVariableOp<sequential_21_batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_21/batch_normalization_35/ReadVariableOp_1ReadVariableOp>sequential_21_batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_21/batch_normalization_35/FusedBatchNormV3FusedBatchNormV3(sequential_21/conv2d_29/BiasAdd:output:0;sequential_21/batch_normalization_35/ReadVariableOp:value:0=sequential_21/batch_normalization_35/ReadVariableOp_1:value:0Lsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
"sequential_21/activation_1/SigmoidSigmoid9sequential_21/batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????00}
IdentityIdentity&sequential_21/activation_1/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOpE^sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpG^sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_14^sequential_15/batch_normalization_26/ReadVariableOp6^sequential_15/batch_normalization_26/ReadVariableOp_18^sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOpA^sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOpE^sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_27/ReadVariableOp6^sequential_16/batch_normalization_27/ReadVariableOp_1E^sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpG^sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_14^sequential_16/batch_normalization_28/ReadVariableOp6^sequential_16/batch_normalization_28/ReadVariableOp_1/^sequential_16/conv2d_23/BiasAdd/ReadVariableOp.^sequential_16/conv2d_23/Conv2D/ReadVariableOp/^sequential_16/conv2d_24/BiasAdd/ReadVariableOp.^sequential_16/conv2d_24/Conv2D/ReadVariableOpE^sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_29/ReadVariableOp6^sequential_17/batch_normalization_29/ReadVariableOp_18^sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOpA^sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOpE^sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpG^sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_14^sequential_18/batch_normalization_30/ReadVariableOp6^sequential_18/batch_normalization_30/ReadVariableOp_1E^sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpG^sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_14^sequential_18/batch_normalization_31/ReadVariableOp6^sequential_18/batch_normalization_31/ReadVariableOp_1/^sequential_18/conv2d_25/BiasAdd/ReadVariableOp.^sequential_18/conv2d_25/Conv2D/ReadVariableOp/^sequential_18/conv2d_26/BiasAdd/ReadVariableOp.^sequential_18/conv2d_26/Conv2D/ReadVariableOpE^sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpG^sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_14^sequential_19/batch_normalization_32/ReadVariableOp6^sequential_19/batch_normalization_32/ReadVariableOp_18^sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOpA^sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOpE^sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpG^sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_14^sequential_20/batch_normalization_33/ReadVariableOp6^sequential_20/batch_normalization_33/ReadVariableOp_1E^sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpG^sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_14^sequential_20/batch_normalization_34/ReadVariableOp6^sequential_20/batch_normalization_34/ReadVariableOp_1/^sequential_20/conv2d_27/BiasAdd/ReadVariableOp.^sequential_20/conv2d_27/Conv2D/ReadVariableOp/^sequential_20/conv2d_28/BiasAdd/ReadVariableOp.^sequential_20/conv2d_28/Conv2D/ReadVariableOpE^sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpG^sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_14^sequential_21/batch_normalization_35/ReadVariableOp6^sequential_21/batch_normalization_35/ReadVariableOp_1/^sequential_21/conv2d_29/BiasAdd/ReadVariableOp.^sequential_21/conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2?
Dsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpDsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Fsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Fsequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12j
3sequential_15/batch_normalization_26/ReadVariableOp3sequential_15/batch_normalization_26/ReadVariableOp2n
5sequential_15/batch_normalization_26/ReadVariableOp_15sequential_15/batch_normalization_26/ReadVariableOp_12r
7sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp7sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp2?
@sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp@sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2?
Dsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Fsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_27/ReadVariableOp3sequential_16/batch_normalization_27/ReadVariableOp2n
5sequential_16/batch_normalization_27/ReadVariableOp_15sequential_16/batch_normalization_27/ReadVariableOp_12?
Dsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpDsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Fsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Fsequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12j
3sequential_16/batch_normalization_28/ReadVariableOp3sequential_16/batch_normalization_28/ReadVariableOp2n
5sequential_16/batch_normalization_28/ReadVariableOp_15sequential_16/batch_normalization_28/ReadVariableOp_12`
.sequential_16/conv2d_23/BiasAdd/ReadVariableOp.sequential_16/conv2d_23/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_23/Conv2D/ReadVariableOp-sequential_16/conv2d_23/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_24/BiasAdd/ReadVariableOp.sequential_16/conv2d_24/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_24/Conv2D/ReadVariableOp-sequential_16/conv2d_24/Conv2D/ReadVariableOp2?
Dsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp2?
Fsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_29/ReadVariableOp3sequential_17/batch_normalization_29/ReadVariableOp2n
5sequential_17/batch_normalization_29/ReadVariableOp_15sequential_17/batch_normalization_29/ReadVariableOp_12r
7sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp7sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp2?
@sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp@sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2?
Dsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpDsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp2?
Fsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1Fsequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12j
3sequential_18/batch_normalization_30/ReadVariableOp3sequential_18/batch_normalization_30/ReadVariableOp2n
5sequential_18/batch_normalization_30/ReadVariableOp_15sequential_18/batch_normalization_30/ReadVariableOp_12?
Dsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpDsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp2?
Fsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1Fsequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12j
3sequential_18/batch_normalization_31/ReadVariableOp3sequential_18/batch_normalization_31/ReadVariableOp2n
5sequential_18/batch_normalization_31/ReadVariableOp_15sequential_18/batch_normalization_31/ReadVariableOp_12`
.sequential_18/conv2d_25/BiasAdd/ReadVariableOp.sequential_18/conv2d_25/BiasAdd/ReadVariableOp2^
-sequential_18/conv2d_25/Conv2D/ReadVariableOp-sequential_18/conv2d_25/Conv2D/ReadVariableOp2`
.sequential_18/conv2d_26/BiasAdd/ReadVariableOp.sequential_18/conv2d_26/BiasAdd/ReadVariableOp2^
-sequential_18/conv2d_26/Conv2D/ReadVariableOp-sequential_18/conv2d_26/Conv2D/ReadVariableOp2?
Dsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpDsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2?
Fsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Fsequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12j
3sequential_19/batch_normalization_32/ReadVariableOp3sequential_19/batch_normalization_32/ReadVariableOp2n
5sequential_19/batch_normalization_32/ReadVariableOp_15sequential_19/batch_normalization_32/ReadVariableOp_12r
7sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp7sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp2?
@sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp@sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2?
Dsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpDsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2?
Fsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Fsequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12j
3sequential_20/batch_normalization_33/ReadVariableOp3sequential_20/batch_normalization_33/ReadVariableOp2n
5sequential_20/batch_normalization_33/ReadVariableOp_15sequential_20/batch_normalization_33/ReadVariableOp_12?
Dsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpDsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2?
Fsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Fsequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12j
3sequential_20/batch_normalization_34/ReadVariableOp3sequential_20/batch_normalization_34/ReadVariableOp2n
5sequential_20/batch_normalization_34/ReadVariableOp_15sequential_20/batch_normalization_34/ReadVariableOp_12`
.sequential_20/conv2d_27/BiasAdd/ReadVariableOp.sequential_20/conv2d_27/BiasAdd/ReadVariableOp2^
-sequential_20/conv2d_27/Conv2D/ReadVariableOp-sequential_20/conv2d_27/Conv2D/ReadVariableOp2`
.sequential_20/conv2d_28/BiasAdd/ReadVariableOp.sequential_20/conv2d_28/BiasAdd/ReadVariableOp2^
-sequential_20/conv2d_28/Conv2D/ReadVariableOp-sequential_20/conv2d_28/Conv2D/ReadVariableOp2?
Dsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpDsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2?
Fsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Fsequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12j
3sequential_21/batch_normalization_35/ReadVariableOp3sequential_21/batch_normalization_35/ReadVariableOp2n
5sequential_21/batch_normalization_35/ReadVariableOp_15sequential_21/batch_normalization_35/ReadVariableOp_12`
.sequential_21/conv2d_29/BiasAdd/ReadVariableOp.sequential_21/conv2d_29/BiasAdd/ReadVariableOp2^
-sequential_21/conv2d_29/Conv2D/ReadVariableOp-sequential_21/conv2d_29/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_33_layer_call_fn_64105

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59583?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59647

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60155

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_29_layer_call_fn_63460

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58466w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63496

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_29_layer_call_fn_63421

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58300?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_31_layer_call_fn_63753

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58812w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_27_layer_call_fn_63082

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_57826w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
-__inference_sequential_21_layer_call_fn_62762

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_60337w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59832

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59267

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_28_layer_call_fn_63222

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57775?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?8
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_62682

inputsB
(conv2d_27_conv2d_readvariableop_resource:7
)conv2d_27_biasadd_readvariableop_resource:<
.batch_normalization_33_readvariableop_resource:>
0batch_normalization_33_readvariableop_1_resource:M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_28_conv2d_readvariableop_resource:7
)conv2d_28_biasadd_readvariableop_resource:<
.batch_normalization_34_readvariableop_resource:>
0batch_normalization_34_readvariableop_1_resource:M
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:
identity??6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_33/ReadVariableOp?'batch_normalization_33/ReadVariableOp_1?6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_34/ReadVariableOp?'batch_normalization_34/ReadVariableOp_1? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3conv2d_27/BiasAdd:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_32/LeakyRelu	LeakyRelu+batch_normalization_33/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_28/Conv2DConv2D&leaky_re_lu_32/LeakyRelu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3conv2d_28/BiasAdd:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_33/LeakyRelu	LeakyRelu+batch_normalization_34/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>}
IdentityIdentity&leaky_re_lu_33/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp7^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_29_layer_call_fn_63434

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58331?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_58409

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_58371

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
H
,__inference_activation_1_layer_call_fn_64514

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_60221h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57775

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_58777

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_16_layer_call_fn_62116

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_58106w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64149

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63649

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?!
?
H__inference_sequential_18_layer_call_and_return_conditional_losses_58830

inputs)
conv2d_25_58740:
conv2d_25_58742:*
batch_normalization_30_58763:*
batch_normalization_30_58765:*
batch_normalization_30_58767:*
batch_normalization_30_58769:)
conv2d_26_58790:
conv2d_26_58792:*
batch_normalization_31_58813:*
batch_normalization_31_58815:*
batch_normalization_31_58817:*
batch_normalization_31_58819:
identity??.batch_normalization_30/StatefulPartitionedCall?.batch_normalization_31/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_25_58740conv2d_25_58742*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_58739?
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0batch_normalization_30_58763batch_normalization_30_58765batch_normalization_30_58767batch_normalization_30_58769*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58762?
leaky_re_lu_29/PartitionedCallPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_58777?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_29/PartitionedCall:output:0conv2d_26_58790conv2d_26_58792*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_58789?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_31_58813batch_normalization_31_58815batch_normalization_31_58817batch_normalization_31_58819*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58812?
leaky_re_lu_30/PartitionedCallPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_58827~
IdentityIdentity'leaky_re_lu_30/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58812

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_3_layer_call_fn_62821

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_57335?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57458

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64203

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_58020

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_64232

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64302

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_31_layer_call_fn_63740

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58711?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_61948

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_15_layer_call_fn_57620
conv2d_transpose_3_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_15_layer_call_and_return_conditional_losses_57588w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:?????????
2
_user_specified_nameconv2d_transpose_3_input
?	
?
6__inference_batch_normalization_34_layer_call_fn_64258

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59647?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63131

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_29_layer_call_fn_63690

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_58777h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_32_layer_call_fn_63939

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59236?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?*
!__inference__traced_restore_64924
file_prefix3
assignvariableop_dense_1_kernel:
??.
assignvariableop_1_dense_1_bias:	?F
,assignvariableop_2_conv2d_transpose_3_kernel: 8
*assignvariableop_3_conv2d_transpose_3_bias: =
/assignvariableop_4_batch_normalization_26_gamma: <
.assignvariableop_5_batch_normalization_26_beta: C
5assignvariableop_6_batch_normalization_26_moving_mean: G
9assignvariableop_7_batch_normalization_26_moving_variance: =
#assignvariableop_8_conv2d_23_kernel:  /
!assignvariableop_9_conv2d_23_bias: >
0assignvariableop_10_batch_normalization_27_gamma: =
/assignvariableop_11_batch_normalization_27_beta: D
6assignvariableop_12_batch_normalization_27_moving_mean: H
:assignvariableop_13_batch_normalization_27_moving_variance: >
$assignvariableop_14_conv2d_24_kernel:  0
"assignvariableop_15_conv2d_24_bias: >
0assignvariableop_16_batch_normalization_28_gamma: =
/assignvariableop_17_batch_normalization_28_beta: D
6assignvariableop_18_batch_normalization_28_moving_mean: H
:assignvariableop_19_batch_normalization_28_moving_variance: G
-assignvariableop_20_conv2d_transpose_4_kernel: 9
+assignvariableop_21_conv2d_transpose_4_bias:>
0assignvariableop_22_batch_normalization_29_gamma:=
/assignvariableop_23_batch_normalization_29_beta:D
6assignvariableop_24_batch_normalization_29_moving_mean:H
:assignvariableop_25_batch_normalization_29_moving_variance:>
$assignvariableop_26_conv2d_25_kernel:0
"assignvariableop_27_conv2d_25_bias:>
0assignvariableop_28_batch_normalization_30_gamma:=
/assignvariableop_29_batch_normalization_30_beta:D
6assignvariableop_30_batch_normalization_30_moving_mean:H
:assignvariableop_31_batch_normalization_30_moving_variance:>
$assignvariableop_32_conv2d_26_kernel:0
"assignvariableop_33_conv2d_26_bias:>
0assignvariableop_34_batch_normalization_31_gamma:=
/assignvariableop_35_batch_normalization_31_beta:D
6assignvariableop_36_batch_normalization_31_moving_mean:H
:assignvariableop_37_batch_normalization_31_moving_variance:G
-assignvariableop_38_conv2d_transpose_5_kernel:9
+assignvariableop_39_conv2d_transpose_5_bias:>
0assignvariableop_40_batch_normalization_32_gamma:=
/assignvariableop_41_batch_normalization_32_beta:D
6assignvariableop_42_batch_normalization_32_moving_mean:H
:assignvariableop_43_batch_normalization_32_moving_variance:>
$assignvariableop_44_conv2d_27_kernel:0
"assignvariableop_45_conv2d_27_bias:>
0assignvariableop_46_batch_normalization_33_gamma:=
/assignvariableop_47_batch_normalization_33_beta:D
6assignvariableop_48_batch_normalization_33_moving_mean:H
:assignvariableop_49_batch_normalization_33_moving_variance:>
$assignvariableop_50_conv2d_28_kernel:0
"assignvariableop_51_conv2d_28_bias:>
0assignvariableop_52_batch_normalization_34_gamma:=
/assignvariableop_53_batch_normalization_34_beta:D
6assignvariableop_54_batch_normalization_34_moving_mean:H
:assignvariableop_55_batch_normalization_34_moving_variance:>
$assignvariableop_56_conv2d_29_kernel:0
"assignvariableop_57_conv2d_29_bias:>
0assignvariableop_58_batch_normalization_35_gamma:=
/assignvariableop_59_batch_normalization_35_beta:D
6assignvariableop_60_batch_normalization_35_moving_mean:H
:assignvariableop_61_batch_normalization_35_moving_variance:
identity_63??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_conv2d_transpose_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_conv2d_transpose_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_26_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_26_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_26_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_26_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_23_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_23_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_27_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_27_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_27_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_27_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_24_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_24_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_28_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_28_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_28_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_28_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_conv2d_transpose_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_conv2d_transpose_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_29_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_29_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_29_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_29_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_25_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_25_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_30_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_30_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_30_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_30_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_26_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_26_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp0assignvariableop_34_batch_normalization_31_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batch_normalization_31_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp6assignvariableop_36_batch_normalization_31_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp:assignvariableop_37_batch_normalization_31_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp-assignvariableop_38_conv2d_transpose_5_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_conv2d_transpose_5_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp0assignvariableop_40_batch_normalization_32_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp/assignvariableop_41_batch_normalization_32_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_batch_normalization_32_moving_meanIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp:assignvariableop_43_batch_normalization_32_moving_varianceIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp$assignvariableop_44_conv2d_27_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp"assignvariableop_45_conv2d_27_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp0assignvariableop_46_batch_normalization_33_gammaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp/assignvariableop_47_batch_normalization_33_betaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp6assignvariableop_48_batch_normalization_33_moving_meanIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp:assignvariableop_49_batch_normalization_33_moving_varianceIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp$assignvariableop_50_conv2d_28_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp"assignvariableop_51_conv2d_28_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp0assignvariableop_52_batch_normalization_34_gammaIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp/assignvariableop_53_batch_normalization_34_betaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp6assignvariableop_54_batch_normalization_34_moving_meanIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp:assignvariableop_55_batch_normalization_34_moving_varianceIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp$assignvariableop_56_conv2d_29_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp"assignvariableop_57_conv2d_29_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp0assignvariableop_58_batch_normalization_35_gammaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp/assignvariableop_59_batch_normalization_35_betaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_batch_normalization_35_moving_meanIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp:assignvariableop_61_batch_normalization_35_moving_varianceIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_63IdentityIdentity_62:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_63Identity_63:output:0*?
_input_shapes?
~: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
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
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59892

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
H__inference_sequential_15_layer_call_and_return_conditional_losses_57588

inputs2
conv2d_transpose_3_57572: &
conv2d_transpose_3_57574: *
batch_normalization_26_57577: *
batch_normalization_26_57579: *
batch_normalization_26_57581: *
batch_normalization_26_57583: 
identity??.batch_normalization_26/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_3_57572conv2d_transpose_3_57574*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_57435?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_26_57577batch_normalization_26_57579batch_normalization_26_57581batch_normalization_26_57583*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57530?
leaky_re_lu_25/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_57473~
IdentityIdentity'leaky_re_lu_25/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_26/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_59763

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????00*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_26_layer_call_fn_62903

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57364?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_62787

inputsB
(conv2d_29_conv2d_readvariableop_resource:7
)conv2d_29_biasadd_readvariableop_resource:<
.batch_normalization_35_readvariableop_resource:>
0batch_normalization_35_readvariableop_1_resource:M
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:
identity??6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_35/ReadVariableOp?'batch_normalization_35/ReadVariableOp_1? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_29/Conv2DConv2Dinputs'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3conv2d_29/BiasAdd:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
activation_1/SigmoidSigmoid+batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????00o
IdentityIdentityactivation_1/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp7^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 2p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_59307

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :0I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :0I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????00*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_31_layer_call_fn_64055

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_59345h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63685

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64032

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
??
?M
 __inference__wrapped_model_57294

args_0B
.model_4_dense_1_matmul_readvariableop_resource:
??>
/model_4_dense_1_biasadd_readvariableop_resource:	?k
Qmodel_4_sequential_15_conv2d_transpose_3_conv2d_transpose_readvariableop_resource: V
Hmodel_4_sequential_15_conv2d_transpose_3_biasadd_readvariableop_resource: R
Dmodel_4_sequential_15_batch_normalization_26_readvariableop_resource: T
Fmodel_4_sequential_15_batch_normalization_26_readvariableop_1_resource: c
Umodel_4_sequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_resource: e
Wmodel_4_sequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource: X
>model_4_sequential_16_conv2d_23_conv2d_readvariableop_resource:  M
?model_4_sequential_16_conv2d_23_biasadd_readvariableop_resource: R
Dmodel_4_sequential_16_batch_normalization_27_readvariableop_resource: T
Fmodel_4_sequential_16_batch_normalization_27_readvariableop_1_resource: c
Umodel_4_sequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_resource: e
Wmodel_4_sequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource: X
>model_4_sequential_16_conv2d_24_conv2d_readvariableop_resource:  M
?model_4_sequential_16_conv2d_24_biasadd_readvariableop_resource: R
Dmodel_4_sequential_16_batch_normalization_28_readvariableop_resource: T
Fmodel_4_sequential_16_batch_normalization_28_readvariableop_1_resource: c
Umodel_4_sequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_resource: e
Wmodel_4_sequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource: k
Qmodel_4_sequential_17_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: V
Hmodel_4_sequential_17_conv2d_transpose_4_biasadd_readvariableop_resource:R
Dmodel_4_sequential_17_batch_normalization_29_readvariableop_resource:T
Fmodel_4_sequential_17_batch_normalization_29_readvariableop_1_resource:c
Umodel_4_sequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_resource:e
Wmodel_4_sequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:X
>model_4_sequential_18_conv2d_25_conv2d_readvariableop_resource:M
?model_4_sequential_18_conv2d_25_biasadd_readvariableop_resource:R
Dmodel_4_sequential_18_batch_normalization_30_readvariableop_resource:T
Fmodel_4_sequential_18_batch_normalization_30_readvariableop_1_resource:c
Umodel_4_sequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_resource:e
Wmodel_4_sequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:X
>model_4_sequential_18_conv2d_26_conv2d_readvariableop_resource:M
?model_4_sequential_18_conv2d_26_biasadd_readvariableop_resource:R
Dmodel_4_sequential_18_batch_normalization_31_readvariableop_resource:T
Fmodel_4_sequential_18_batch_normalization_31_readvariableop_1_resource:c
Umodel_4_sequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_resource:e
Wmodel_4_sequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:k
Qmodel_4_sequential_19_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:V
Hmodel_4_sequential_19_conv2d_transpose_5_biasadd_readvariableop_resource:R
Dmodel_4_sequential_19_batch_normalization_32_readvariableop_resource:T
Fmodel_4_sequential_19_batch_normalization_32_readvariableop_1_resource:c
Umodel_4_sequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:e
Wmodel_4_sequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:X
>model_4_sequential_20_conv2d_27_conv2d_readvariableop_resource:M
?model_4_sequential_20_conv2d_27_biasadd_readvariableop_resource:R
Dmodel_4_sequential_20_batch_normalization_33_readvariableop_resource:T
Fmodel_4_sequential_20_batch_normalization_33_readvariableop_1_resource:c
Umodel_4_sequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:e
Wmodel_4_sequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:X
>model_4_sequential_20_conv2d_28_conv2d_readvariableop_resource:M
?model_4_sequential_20_conv2d_28_biasadd_readvariableop_resource:R
Dmodel_4_sequential_20_batch_normalization_34_readvariableop_resource:T
Fmodel_4_sequential_20_batch_normalization_34_readvariableop_1_resource:c
Umodel_4_sequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:e
Wmodel_4_sequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:X
>model_4_sequential_21_conv2d_29_conv2d_readvariableop_resource:M
?model_4_sequential_21_conv2d_29_biasadd_readvariableop_resource:R
Dmodel_4_sequential_21_batch_normalization_35_readvariableop_resource:T
Fmodel_4_sequential_21_batch_normalization_35_readvariableop_1_resource:c
Umodel_4_sequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_resource:e
Wmodel_4_sequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:
identity??&model_4/dense_1/BiasAdd/ReadVariableOp?%model_4/dense_1/MatMul/ReadVariableOp?Lmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_15/batch_normalization_26/ReadVariableOp?=model_4/sequential_15/batch_normalization_26/ReadVariableOp_1??model_4/sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp?Hmodel_4/sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?Lmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_16/batch_normalization_27/ReadVariableOp?=model_4/sequential_16/batch_normalization_27/ReadVariableOp_1?Lmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_16/batch_normalization_28/ReadVariableOp?=model_4/sequential_16/batch_normalization_28/ReadVariableOp_1?6model_4/sequential_16/conv2d_23/BiasAdd/ReadVariableOp?5model_4/sequential_16/conv2d_23/Conv2D/ReadVariableOp?6model_4/sequential_16/conv2d_24/BiasAdd/ReadVariableOp?5model_4/sequential_16/conv2d_24/Conv2D/ReadVariableOp?Lmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_17/batch_normalization_29/ReadVariableOp?=model_4/sequential_17/batch_normalization_29/ReadVariableOp_1??model_4/sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp?Hmodel_4/sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?Lmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_18/batch_normalization_30/ReadVariableOp?=model_4/sequential_18/batch_normalization_30/ReadVariableOp_1?Lmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_18/batch_normalization_31/ReadVariableOp?=model_4/sequential_18/batch_normalization_31/ReadVariableOp_1?6model_4/sequential_18/conv2d_25/BiasAdd/ReadVariableOp?5model_4/sequential_18/conv2d_25/Conv2D/ReadVariableOp?6model_4/sequential_18/conv2d_26/BiasAdd/ReadVariableOp?5model_4/sequential_18/conv2d_26/Conv2D/ReadVariableOp?Lmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_19/batch_normalization_32/ReadVariableOp?=model_4/sequential_19/batch_normalization_32/ReadVariableOp_1??model_4/sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp?Hmodel_4/sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?Lmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_20/batch_normalization_33/ReadVariableOp?=model_4/sequential_20/batch_normalization_33/ReadVariableOp_1?Lmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_20/batch_normalization_34/ReadVariableOp?=model_4/sequential_20/batch_normalization_34/ReadVariableOp_1?6model_4/sequential_20/conv2d_27/BiasAdd/ReadVariableOp?5model_4/sequential_20/conv2d_27/Conv2D/ReadVariableOp?6model_4/sequential_20/conv2d_28/BiasAdd/ReadVariableOp?5model_4/sequential_20/conv2d_28/Conv2D/ReadVariableOp?Lmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp?Nmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?;model_4/sequential_21/batch_normalization_35/ReadVariableOp?=model_4/sequential_21/batch_normalization_35/ReadVariableOp_1?6model_4/sequential_21/conv2d_29/BiasAdd/ReadVariableOp?5model_4/sequential_21/conv2d_29/Conv2D/ReadVariableOp?
%model_4/dense_1/MatMul/ReadVariableOpReadVariableOp.model_4_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_4/dense_1/MatMulMatMulargs_0-model_4/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&model_4/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_4_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_4/dense_1/BiasAddBiasAdd model_4/dense_1/MatMul:product:0.model_4/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????g
model_4/reshape_1/ShapeShape model_4/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:o
%model_4/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_4/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_4/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model_4/reshape_1/strided_sliceStridedSlice model_4/reshape_1/Shape:output:0.model_4/reshape_1/strided_slice/stack:output:00model_4/reshape_1/strided_slice/stack_1:output:00model_4/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_4/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_4/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model_4/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
model_4/reshape_1/Reshape/shapePack(model_4/reshape_1/strided_slice:output:0*model_4/reshape_1/Reshape/shape/1:output:0*model_4/reshape_1/Reshape/shape/2:output:0*model_4/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
model_4/reshape_1/ReshapeReshape model_4/dense_1/BiasAdd:output:0(model_4/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
.model_4/sequential_15/conv2d_transpose_3/ShapeShape"model_4/reshape_1/Reshape:output:0*
T0*
_output_shapes
:?
<model_4/sequential_15/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>model_4/sequential_15/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>model_4/sequential_15/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6model_4/sequential_15/conv2d_transpose_3/strided_sliceStridedSlice7model_4/sequential_15/conv2d_transpose_3/Shape:output:0Emodel_4/sequential_15/conv2d_transpose_3/strided_slice/stack:output:0Gmodel_4/sequential_15/conv2d_transpose_3/strided_slice/stack_1:output:0Gmodel_4/sequential_15/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0model_4/sequential_15/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :r
0model_4/sequential_15/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :r
0model_4/sequential_15/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
.model_4/sequential_15/conv2d_transpose_3/stackPack?model_4/sequential_15/conv2d_transpose_3/strided_slice:output:09model_4/sequential_15/conv2d_transpose_3/stack/1:output:09model_4/sequential_15/conv2d_transpose_3/stack/2:output:09model_4/sequential_15/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:?
>model_4/sequential_15/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_4/sequential_15/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@model_4/sequential_15/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8model_4/sequential_15/conv2d_transpose_3/strided_slice_1StridedSlice7model_4/sequential_15/conv2d_transpose_3/stack:output:0Gmodel_4/sequential_15/conv2d_transpose_3/strided_slice_1/stack:output:0Imodel_4/sequential_15/conv2d_transpose_3/strided_slice_1/stack_1:output:0Imodel_4/sequential_15/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Hmodel_4/sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpQmodel_4_sequential_15_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
9model_4/sequential_15/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput7model_4/sequential_15/conv2d_transpose_3/stack:output:0Pmodel_4/sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0"model_4/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
?model_4/sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpHmodel_4_sequential_15_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
0model_4/sequential_15/conv2d_transpose_3/BiasAddBiasAddBmodel_4/sequential_15/conv2d_transpose_3/conv2d_transpose:output:0Gmodel_4/sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
;model_4/sequential_15/batch_normalization_26/ReadVariableOpReadVariableOpDmodel_4_sequential_15_batch_normalization_26_readvariableop_resource*
_output_shapes
: *
dtype0?
=model_4/sequential_15/batch_normalization_26/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_15_batch_normalization_26_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Lmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Nmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_15_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
=model_4/sequential_15/batch_normalization_26/FusedBatchNormV3FusedBatchNormV39model_4/sequential_15/conv2d_transpose_3/BiasAdd:output:0Cmodel_4/sequential_15/batch_normalization_26/ReadVariableOp:value:0Emodel_4/sequential_15/batch_normalization_26/ReadVariableOp_1:value:0Tmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
.model_4/sequential_15/leaky_re_lu_25/LeakyRelu	LeakyReluAmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
5model_4/sequential_16/conv2d_23/Conv2D/ReadVariableOpReadVariableOp>model_4_sequential_16_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
&model_4/sequential_16/conv2d_23/Conv2DConv2D<model_4/sequential_15/leaky_re_lu_25/LeakyRelu:activations:0=model_4/sequential_16/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
6model_4/sequential_16/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp?model_4_sequential_16_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
'model_4/sequential_16/conv2d_23/BiasAddBiasAdd/model_4/sequential_16/conv2d_23/Conv2D:output:0>model_4/sequential_16/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
;model_4/sequential_16/batch_normalization_27/ReadVariableOpReadVariableOpDmodel_4_sequential_16_batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype0?
=model_4/sequential_16/batch_normalization_27/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_16_batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Lmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Nmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_16_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
=model_4/sequential_16/batch_normalization_27/FusedBatchNormV3FusedBatchNormV30model_4/sequential_16/conv2d_23/BiasAdd:output:0Cmodel_4/sequential_16/batch_normalization_27/ReadVariableOp:value:0Emodel_4/sequential_16/batch_normalization_27/ReadVariableOp_1:value:0Tmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
.model_4/sequential_16/leaky_re_lu_26/LeakyRelu	LeakyReluAmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
5model_4/sequential_16/conv2d_24/Conv2D/ReadVariableOpReadVariableOp>model_4_sequential_16_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
&model_4/sequential_16/conv2d_24/Conv2DConv2D<model_4/sequential_16/leaky_re_lu_26/LeakyRelu:activations:0=model_4/sequential_16/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
6model_4/sequential_16/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp?model_4_sequential_16_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
'model_4/sequential_16/conv2d_24/BiasAddBiasAdd/model_4/sequential_16/conv2d_24/Conv2D:output:0>model_4/sequential_16/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
;model_4/sequential_16/batch_normalization_28/ReadVariableOpReadVariableOpDmodel_4_sequential_16_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype0?
=model_4/sequential_16/batch_normalization_28/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_16_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Lmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Nmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_16_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
=model_4/sequential_16/batch_normalization_28/FusedBatchNormV3FusedBatchNormV30model_4/sequential_16/conv2d_24/BiasAdd:output:0Cmodel_4/sequential_16/batch_normalization_28/ReadVariableOp:value:0Emodel_4/sequential_16/batch_normalization_28/ReadVariableOp_1:value:0Tmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
.model_4/sequential_16/leaky_re_lu_27/LeakyRelu	LeakyReluAmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
.model_4/sequential_17/conv2d_transpose_4/ShapeShape<model_4/sequential_16/leaky_re_lu_27/LeakyRelu:activations:0*
T0*
_output_shapes
:?
<model_4/sequential_17/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>model_4/sequential_17/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>model_4/sequential_17/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6model_4/sequential_17/conv2d_transpose_4/strided_sliceStridedSlice7model_4/sequential_17/conv2d_transpose_4/Shape:output:0Emodel_4/sequential_17/conv2d_transpose_4/strided_slice/stack:output:0Gmodel_4/sequential_17/conv2d_transpose_4/strided_slice/stack_1:output:0Gmodel_4/sequential_17/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0model_4/sequential_17/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :r
0model_4/sequential_17/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :r
0model_4/sequential_17/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
.model_4/sequential_17/conv2d_transpose_4/stackPack?model_4/sequential_17/conv2d_transpose_4/strided_slice:output:09model_4/sequential_17/conv2d_transpose_4/stack/1:output:09model_4/sequential_17/conv2d_transpose_4/stack/2:output:09model_4/sequential_17/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:?
>model_4/sequential_17/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_4/sequential_17/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@model_4/sequential_17/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8model_4/sequential_17/conv2d_transpose_4/strided_slice_1StridedSlice7model_4/sequential_17/conv2d_transpose_4/stack:output:0Gmodel_4/sequential_17/conv2d_transpose_4/strided_slice_1/stack:output:0Imodel_4/sequential_17/conv2d_transpose_4/strided_slice_1/stack_1:output:0Imodel_4/sequential_17/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Hmodel_4/sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpQmodel_4_sequential_17_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
9model_4/sequential_17/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput7model_4/sequential_17/conv2d_transpose_4/stack:output:0Pmodel_4/sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0<model_4/sequential_16/leaky_re_lu_27/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
?model_4/sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpHmodel_4_sequential_17_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
0model_4/sequential_17/conv2d_transpose_4/BiasAddBiasAddBmodel_4/sequential_17/conv2d_transpose_4/conv2d_transpose:output:0Gmodel_4/sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
;model_4/sequential_17/batch_normalization_29/ReadVariableOpReadVariableOpDmodel_4_sequential_17_batch_normalization_29_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_17/batch_normalization_29/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_17_batch_normalization_29_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_17_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_17/batch_normalization_29/FusedBatchNormV3FusedBatchNormV39model_4/sequential_17/conv2d_transpose_4/BiasAdd:output:0Cmodel_4/sequential_17/batch_normalization_29/ReadVariableOp:value:0Emodel_4/sequential_17/batch_normalization_29/ReadVariableOp_1:value:0Tmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
.model_4/sequential_17/leaky_re_lu_28/LeakyRelu	LeakyReluAmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
5model_4/sequential_18/conv2d_25/Conv2D/ReadVariableOpReadVariableOp>model_4_sequential_18_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_4/sequential_18/conv2d_25/Conv2DConv2D<model_4/sequential_17/leaky_re_lu_28/LeakyRelu:activations:0=model_4/sequential_18/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
6model_4/sequential_18/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp?model_4_sequential_18_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_4/sequential_18/conv2d_25/BiasAddBiasAdd/model_4/sequential_18/conv2d_25/Conv2D:output:0>model_4/sequential_18/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
;model_4/sequential_18/batch_normalization_30/ReadVariableOpReadVariableOpDmodel_4_sequential_18_batch_normalization_30_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_18/batch_normalization_30/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_18_batch_normalization_30_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_18_batch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_18/batch_normalization_30/FusedBatchNormV3FusedBatchNormV30model_4/sequential_18/conv2d_25/BiasAdd:output:0Cmodel_4/sequential_18/batch_normalization_30/ReadVariableOp:value:0Emodel_4/sequential_18/batch_normalization_30/ReadVariableOp_1:value:0Tmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
.model_4/sequential_18/leaky_re_lu_29/LeakyRelu	LeakyReluAmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
5model_4/sequential_18/conv2d_26/Conv2D/ReadVariableOpReadVariableOp>model_4_sequential_18_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_4/sequential_18/conv2d_26/Conv2DConv2D<model_4/sequential_18/leaky_re_lu_29/LeakyRelu:activations:0=model_4/sequential_18/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
6model_4/sequential_18/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp?model_4_sequential_18_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_4/sequential_18/conv2d_26/BiasAddBiasAdd/model_4/sequential_18/conv2d_26/Conv2D:output:0>model_4/sequential_18/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
;model_4/sequential_18/batch_normalization_31/ReadVariableOpReadVariableOpDmodel_4_sequential_18_batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_18/batch_normalization_31/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_18_batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_18_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_18/batch_normalization_31/FusedBatchNormV3FusedBatchNormV30model_4/sequential_18/conv2d_26/BiasAdd:output:0Cmodel_4/sequential_18/batch_normalization_31/ReadVariableOp:value:0Emodel_4/sequential_18/batch_normalization_31/ReadVariableOp_1:value:0Tmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
.model_4/sequential_18/leaky_re_lu_30/LeakyRelu	LeakyReluAmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
.model_4/sequential_19/conv2d_transpose_5/ShapeShape<model_4/sequential_18/leaky_re_lu_30/LeakyRelu:activations:0*
T0*
_output_shapes
:?
<model_4/sequential_19/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>model_4/sequential_19/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>model_4/sequential_19/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6model_4/sequential_19/conv2d_transpose_5/strided_sliceStridedSlice7model_4/sequential_19/conv2d_transpose_5/Shape:output:0Emodel_4/sequential_19/conv2d_transpose_5/strided_slice/stack:output:0Gmodel_4/sequential_19/conv2d_transpose_5/strided_slice/stack_1:output:0Gmodel_4/sequential_19/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0model_4/sequential_19/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0r
0model_4/sequential_19/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0r
0model_4/sequential_19/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
.model_4/sequential_19/conv2d_transpose_5/stackPack?model_4/sequential_19/conv2d_transpose_5/strided_slice:output:09model_4/sequential_19/conv2d_transpose_5/stack/1:output:09model_4/sequential_19/conv2d_transpose_5/stack/2:output:09model_4/sequential_19/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:?
>model_4/sequential_19/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_4/sequential_19/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@model_4/sequential_19/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8model_4/sequential_19/conv2d_transpose_5/strided_slice_1StridedSlice7model_4/sequential_19/conv2d_transpose_5/stack:output:0Gmodel_4/sequential_19/conv2d_transpose_5/strided_slice_1/stack:output:0Imodel_4/sequential_19/conv2d_transpose_5/strided_slice_1/stack_1:output:0Imodel_4/sequential_19/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Hmodel_4/sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpQmodel_4_sequential_19_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
9model_4/sequential_19/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput7model_4/sequential_19/conv2d_transpose_5/stack:output:0Pmodel_4/sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0<model_4/sequential_18/leaky_re_lu_30/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00*
paddingVALID*
strides
?
?model_4/sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpHmodel_4_sequential_19_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
0model_4/sequential_19/conv2d_transpose_5/BiasAddBiasAddBmodel_4/sequential_19/conv2d_transpose_5/conv2d_transpose:output:0Gmodel_4/sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
;model_4/sequential_19/batch_normalization_32/ReadVariableOpReadVariableOpDmodel_4_sequential_19_batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_19/batch_normalization_32/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_19_batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_19_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_19/batch_normalization_32/FusedBatchNormV3FusedBatchNormV39model_4/sequential_19/conv2d_transpose_5/BiasAdd:output:0Cmodel_4/sequential_19/batch_normalization_32/ReadVariableOp:value:0Emodel_4/sequential_19/batch_normalization_32/ReadVariableOp_1:value:0Tmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
.model_4/sequential_19/leaky_re_lu_31/LeakyRelu	LeakyReluAmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
5model_4/sequential_20/conv2d_27/Conv2D/ReadVariableOpReadVariableOp>model_4_sequential_20_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_4/sequential_20/conv2d_27/Conv2DConv2D<model_4/sequential_19/leaky_re_lu_31/LeakyRelu:activations:0=model_4/sequential_20/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
6model_4/sequential_20/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp?model_4_sequential_20_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_4/sequential_20/conv2d_27/BiasAddBiasAdd/model_4/sequential_20/conv2d_27/Conv2D:output:0>model_4/sequential_20/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
;model_4/sequential_20/batch_normalization_33/ReadVariableOpReadVariableOpDmodel_4_sequential_20_batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_20/batch_normalization_33/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_20_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_20_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_20/batch_normalization_33/FusedBatchNormV3FusedBatchNormV30model_4/sequential_20/conv2d_27/BiasAdd:output:0Cmodel_4/sequential_20/batch_normalization_33/ReadVariableOp:value:0Emodel_4/sequential_20/batch_normalization_33/ReadVariableOp_1:value:0Tmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
.model_4/sequential_20/leaky_re_lu_32/LeakyRelu	LeakyReluAmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
5model_4/sequential_20/conv2d_28/Conv2D/ReadVariableOpReadVariableOp>model_4_sequential_20_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_4/sequential_20/conv2d_28/Conv2DConv2D<model_4/sequential_20/leaky_re_lu_32/LeakyRelu:activations:0=model_4/sequential_20/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
6model_4/sequential_20/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp?model_4_sequential_20_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_4/sequential_20/conv2d_28/BiasAddBiasAdd/model_4/sequential_20/conv2d_28/Conv2D:output:0>model_4/sequential_20/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
;model_4/sequential_20/batch_normalization_34/ReadVariableOpReadVariableOpDmodel_4_sequential_20_batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_20/batch_normalization_34/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_20_batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_20_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_20/batch_normalization_34/FusedBatchNormV3FusedBatchNormV30model_4/sequential_20/conv2d_28/BiasAdd:output:0Cmodel_4/sequential_20/batch_normalization_34/ReadVariableOp:value:0Emodel_4/sequential_20/batch_normalization_34/ReadVariableOp_1:value:0Tmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
.model_4/sequential_20/leaky_re_lu_33/LeakyRelu	LeakyReluAmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
5model_4/sequential_21/conv2d_29/Conv2D/ReadVariableOpReadVariableOp>model_4_sequential_21_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_4/sequential_21/conv2d_29/Conv2DConv2D<model_4/sequential_20/leaky_re_lu_33/LeakyRelu:activations:0=model_4/sequential_21/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
6model_4/sequential_21/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp?model_4_sequential_21_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_4/sequential_21/conv2d_29/BiasAddBiasAdd/model_4/sequential_21/conv2d_29/Conv2D:output:0>model_4/sequential_21/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
;model_4/sequential_21/batch_normalization_35/ReadVariableOpReadVariableOpDmodel_4_sequential_21_batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_21/batch_normalization_35/ReadVariableOp_1ReadVariableOpFmodel_4_sequential_21_batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_4_sequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_4_sequential_21_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_4/sequential_21/batch_normalization_35/FusedBatchNormV3FusedBatchNormV30model_4/sequential_21/conv2d_29/BiasAdd:output:0Cmodel_4/sequential_21/batch_normalization_35/ReadVariableOp:value:0Emodel_4/sequential_21/batch_normalization_35/ReadVariableOp_1:value:0Tmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
*model_4/sequential_21/activation_1/SigmoidSigmoidAmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????00?
IdentityIdentity.model_4/sequential_21/activation_1/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?!
NoOpNoOp'^model_4/dense_1/BiasAdd/ReadVariableOp&^model_4/dense_1/MatMul/ReadVariableOpM^model_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_15/batch_normalization_26/ReadVariableOp>^model_4/sequential_15/batch_normalization_26/ReadVariableOp_1@^model_4/sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOpI^model_4/sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOpM^model_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_16/batch_normalization_27/ReadVariableOp>^model_4/sequential_16/batch_normalization_27/ReadVariableOp_1M^model_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_16/batch_normalization_28/ReadVariableOp>^model_4/sequential_16/batch_normalization_28/ReadVariableOp_17^model_4/sequential_16/conv2d_23/BiasAdd/ReadVariableOp6^model_4/sequential_16/conv2d_23/Conv2D/ReadVariableOp7^model_4/sequential_16/conv2d_24/BiasAdd/ReadVariableOp6^model_4/sequential_16/conv2d_24/Conv2D/ReadVariableOpM^model_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_17/batch_normalization_29/ReadVariableOp>^model_4/sequential_17/batch_normalization_29/ReadVariableOp_1@^model_4/sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOpI^model_4/sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOpM^model_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_18/batch_normalization_30/ReadVariableOp>^model_4/sequential_18/batch_normalization_30/ReadVariableOp_1M^model_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_18/batch_normalization_31/ReadVariableOp>^model_4/sequential_18/batch_normalization_31/ReadVariableOp_17^model_4/sequential_18/conv2d_25/BiasAdd/ReadVariableOp6^model_4/sequential_18/conv2d_25/Conv2D/ReadVariableOp7^model_4/sequential_18/conv2d_26/BiasAdd/ReadVariableOp6^model_4/sequential_18/conv2d_26/Conv2D/ReadVariableOpM^model_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_19/batch_normalization_32/ReadVariableOp>^model_4/sequential_19/batch_normalization_32/ReadVariableOp_1@^model_4/sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOpI^model_4/sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOpM^model_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_20/batch_normalization_33/ReadVariableOp>^model_4/sequential_20/batch_normalization_33/ReadVariableOp_1M^model_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_20/batch_normalization_34/ReadVariableOp>^model_4/sequential_20/batch_normalization_34/ReadVariableOp_17^model_4/sequential_20/conv2d_27/BiasAdd/ReadVariableOp6^model_4/sequential_20/conv2d_27/Conv2D/ReadVariableOp7^model_4/sequential_20/conv2d_28/BiasAdd/ReadVariableOp6^model_4/sequential_20/conv2d_28/Conv2D/ReadVariableOpM^model_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpO^model_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1<^model_4/sequential_21/batch_normalization_35/ReadVariableOp>^model_4/sequential_21/batch_normalization_35/ReadVariableOp_17^model_4/sequential_21/conv2d_29/BiasAdd/ReadVariableOp6^model_4/sequential_21/conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_4/dense_1/BiasAdd/ReadVariableOp&model_4/dense_1/BiasAdd/ReadVariableOp2N
%model_4/dense_1/MatMul/ReadVariableOp%model_4/dense_1/MatMul/ReadVariableOp2?
Lmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_15/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_15/batch_normalization_26/ReadVariableOp;model_4/sequential_15/batch_normalization_26/ReadVariableOp2~
=model_4/sequential_15/batch_normalization_26/ReadVariableOp_1=model_4/sequential_15/batch_normalization_26/ReadVariableOp_12?
?model_4/sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp?model_4/sequential_15/conv2d_transpose_3/BiasAdd/ReadVariableOp2?
Hmodel_4/sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOpHmodel_4/sequential_15/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2?
Lmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_16/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_16/batch_normalization_27/ReadVariableOp;model_4/sequential_16/batch_normalization_27/ReadVariableOp2~
=model_4/sequential_16/batch_normalization_27/ReadVariableOp_1=model_4/sequential_16/batch_normalization_27/ReadVariableOp_12?
Lmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_16/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_16/batch_normalization_28/ReadVariableOp;model_4/sequential_16/batch_normalization_28/ReadVariableOp2~
=model_4/sequential_16/batch_normalization_28/ReadVariableOp_1=model_4/sequential_16/batch_normalization_28/ReadVariableOp_12p
6model_4/sequential_16/conv2d_23/BiasAdd/ReadVariableOp6model_4/sequential_16/conv2d_23/BiasAdd/ReadVariableOp2n
5model_4/sequential_16/conv2d_23/Conv2D/ReadVariableOp5model_4/sequential_16/conv2d_23/Conv2D/ReadVariableOp2p
6model_4/sequential_16/conv2d_24/BiasAdd/ReadVariableOp6model_4/sequential_16/conv2d_24/BiasAdd/ReadVariableOp2n
5model_4/sequential_16/conv2d_24/Conv2D/ReadVariableOp5model_4/sequential_16/conv2d_24/Conv2D/ReadVariableOp2?
Lmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_17/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_17/batch_normalization_29/ReadVariableOp;model_4/sequential_17/batch_normalization_29/ReadVariableOp2~
=model_4/sequential_17/batch_normalization_29/ReadVariableOp_1=model_4/sequential_17/batch_normalization_29/ReadVariableOp_12?
?model_4/sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp?model_4/sequential_17/conv2d_transpose_4/BiasAdd/ReadVariableOp2?
Hmodel_4/sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOpHmodel_4/sequential_17/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2?
Lmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_18/batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_18/batch_normalization_30/ReadVariableOp;model_4/sequential_18/batch_normalization_30/ReadVariableOp2~
=model_4/sequential_18/batch_normalization_30/ReadVariableOp_1=model_4/sequential_18/batch_normalization_30/ReadVariableOp_12?
Lmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_18/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_18/batch_normalization_31/ReadVariableOp;model_4/sequential_18/batch_normalization_31/ReadVariableOp2~
=model_4/sequential_18/batch_normalization_31/ReadVariableOp_1=model_4/sequential_18/batch_normalization_31/ReadVariableOp_12p
6model_4/sequential_18/conv2d_25/BiasAdd/ReadVariableOp6model_4/sequential_18/conv2d_25/BiasAdd/ReadVariableOp2n
5model_4/sequential_18/conv2d_25/Conv2D/ReadVariableOp5model_4/sequential_18/conv2d_25/Conv2D/ReadVariableOp2p
6model_4/sequential_18/conv2d_26/BiasAdd/ReadVariableOp6model_4/sequential_18/conv2d_26/BiasAdd/ReadVariableOp2n
5model_4/sequential_18/conv2d_26/Conv2D/ReadVariableOp5model_4/sequential_18/conv2d_26/Conv2D/ReadVariableOp2?
Lmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_19/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_19/batch_normalization_32/ReadVariableOp;model_4/sequential_19/batch_normalization_32/ReadVariableOp2~
=model_4/sequential_19/batch_normalization_32/ReadVariableOp_1=model_4/sequential_19/batch_normalization_32/ReadVariableOp_12?
?model_4/sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp?model_4/sequential_19/conv2d_transpose_5/BiasAdd/ReadVariableOp2?
Hmodel_4/sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOpHmodel_4/sequential_19/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2?
Lmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_20/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_20/batch_normalization_33/ReadVariableOp;model_4/sequential_20/batch_normalization_33/ReadVariableOp2~
=model_4/sequential_20/batch_normalization_33/ReadVariableOp_1=model_4/sequential_20/batch_normalization_33/ReadVariableOp_12?
Lmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_20/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_20/batch_normalization_34/ReadVariableOp;model_4/sequential_20/batch_normalization_34/ReadVariableOp2~
=model_4/sequential_20/batch_normalization_34/ReadVariableOp_1=model_4/sequential_20/batch_normalization_34/ReadVariableOp_12p
6model_4/sequential_20/conv2d_27/BiasAdd/ReadVariableOp6model_4/sequential_20/conv2d_27/BiasAdd/ReadVariableOp2n
5model_4/sequential_20/conv2d_27/Conv2D/ReadVariableOp5model_4/sequential_20/conv2d_27/Conv2D/ReadVariableOp2p
6model_4/sequential_20/conv2d_28/BiasAdd/ReadVariableOp6model_4/sequential_20/conv2d_28/BiasAdd/ReadVariableOp2n
5model_4/sequential_20/conv2d_28/Conv2D/ReadVariableOp5model_4/sequential_20/conv2d_28/Conv2D/ReadVariableOp2?
Lmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOpLmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2?
Nmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Nmodel_4/sequential_21/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12z
;model_4/sequential_21/batch_normalization_35/ReadVariableOp;model_4/sequential_21/batch_normalization_35/ReadVariableOp2~
=model_4/sequential_21/batch_normalization_35/ReadVariableOp_1=model_4/sequential_21/batch_normalization_35/ReadVariableOp_12p
6model_4/sequential_21/conv2d_29/BiasAdd/ReadVariableOp6model_4/sequential_21/conv2d_29/BiasAdd/ReadVariableOp2n
5model_4/sequential_21/conv2d_29/Conv2D/ReadVariableOp5model_4/sequential_21/conv2d_29/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_57680

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?6
?
H__inference_sequential_17_layer_call_and_return_conditional_losses_62318

inputsU
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_4_biasadd_readvariableop_resource:<
.batch_normalization_29_readvariableop_resource:>
0batch_normalization_29_readvariableop_1_resource:M
?batch_normalization_29_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource:
identity??%batch_normalization_29/AssignNewValue?'batch_normalization_29/AssignNewValue_1?6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_29/ReadVariableOp?'batch_normalization_29/ReadVariableOp_1?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOpN
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_29/ReadVariableOpReadVariableOp.batch_normalization_29_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_29/ReadVariableOp_1ReadVariableOp0batch_normalization_29_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_29/FusedBatchNormV3FusedBatchNormV3#conv2d_transpose_4/BiasAdd:output:0-batch_normalization_29/ReadVariableOp:value:0/batch_normalization_29/ReadVariableOp_1:value:0>batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_29/AssignNewValueAssignVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource4batch_normalization_29/FusedBatchNormV3:batch_mean:07^batch_normalization_29/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_29/AssignNewValue_1AssignVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_29/FusedBatchNormV3:batch_variance:09^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_28/LeakyRelu	LeakyRelu+batch_normalization_29/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>}
IdentityIdentity&leaky_re_lu_28/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_29/AssignNewValue(^batch_normalization_29/AssignNewValue_17^batch_normalization_29/FusedBatchNormV3/ReadVariableOp9^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_29/ReadVariableOp(^batch_normalization_29/ReadVariableOp_1*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 2N
%batch_normalization_29/AssignNewValue%batch_normalization_29/AssignNewValue2R
'batch_normalization_29/AssignNewValue_1'batch_normalization_29/AssignNewValue_12p
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp6batch_normalization_29/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_18batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_29/ReadVariableOp%batch_normalization_29/ReadVariableOp2R
'batch_normalization_29/ReadVariableOp_1'batch_normalization_29/ReadVariableOp_12V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58680

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57364

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_58739

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_model_4_layer_call_fn_61243

inputs
unknown:
??
	unknown_0:	?#
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:$

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:$

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:$

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:$

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:$

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:
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
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_60574w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_58827

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_64060

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????00*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_57853

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_28_layer_call_fn_63248

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57960w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59402

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_30_layer_call_fn_63843

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_58827h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63514

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64050

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_4_layer_call_fn_63348

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_58371w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_33_layer_call_fn_64118

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59698w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?"
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_60102
conv2d_27_input)
conv2d_27_60071:
conv2d_27_60073:*
batch_normalization_33_60076:*
batch_normalization_33_60078:*
batch_normalization_33_60080:*
batch_normalization_33_60082:)
conv2d_28_60086:
conv2d_28_60088:*
batch_normalization_34_60091:*
batch_normalization_34_60093:*
batch_normalization_34_60095:*
batch_normalization_34_60097:
identity??.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallconv2d_27_inputconv2d_27_60071conv2d_27_60073*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_59675?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_33_60076batch_normalization_33_60078batch_normalization_33_60080batch_normalization_33_60082*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59892?
leaky_re_lu_32/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_59713?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0conv2d_28_60086conv2d_28_60088*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_59725?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_34_60091batch_normalization_34_60093batch_normalization_34_60095batch_normalization_34_60097*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59832?
leaky_re_lu_33/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_59763~
IdentityIdentity'leaky_re_lu_33/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_27_input
?6
?
H__inference_sequential_15_layer_call_and_return_conditional_losses_62058

inputsU
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_3_biasadd_readvariableop_resource: <
.batch_normalization_26_readvariableop_resource: >
0batch_normalization_26_readvariableop_1_resource: M
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource: 
identity??%batch_normalization_26/AssignNewValue?'batch_normalization_26/AssignNewValue_1?6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOpN
conv2d_transpose_3/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3#conv2d_transpose_3/BiasAdd:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_25/LeakyRelu	LeakyRelu+batch_normalization_26/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>}
IdentityIdentity&leaky_re_lu_25/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_30_layer_call_fn_63574

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58616?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?"
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_60068
conv2d_27_input)
conv2d_27_60037:
conv2d_27_60039:*
batch_normalization_33_60042:*
batch_normalization_33_60044:*
batch_normalization_33_60046:*
batch_normalization_33_60048:)
conv2d_28_60052:
conv2d_28_60054:*
batch_normalization_34_60057:*
batch_normalization_34_60059:*
batch_normalization_34_60061:*
batch_normalization_34_60063:
identity??.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallconv2d_27_inputconv2d_27_60037conv2d_27_60039*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_59675?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_33_60042batch_normalization_33_60044batch_normalization_33_60046batch_normalization_33_60048*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59698?
leaky_re_lu_32/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_59713?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0conv2d_28_60052conv2d_28_60054*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_59725?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_34_60057batch_normalization_34_60059batch_normalization_34_60061batch_normalization_34_60063*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59748?
leaky_re_lu_33/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_59763~
IdentityIdentity'leaky_re_lu_33/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_27_input
?
c
G__inference_activation_1_layer_call_and_return_conditional_losses_64519

inputs
identityT
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????00[
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_59713

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????00*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_28_layer_call_fn_63537

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_58409h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_58271

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
H__inference_sequential_19_layer_call_and_return_conditional_losses_59348

inputs2
conv2d_transpose_5_59308:&
conv2d_transpose_5_59310:*
batch_normalization_32_59331:*
batch_normalization_32_59333:*
batch_normalization_32_59335:*
batch_normalization_32_59337:
identity??.batch_normalization_32/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_5_59308conv2d_transpose_5_59310*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_59307?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_32_59331batch_normalization_32_59333batch_normalization_32_59335batch_normalization_32_59337*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59330?
leaky_re_lu_31/PartitionedCallPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_59345~
IdentityIdentity'leaky_re_lu_31/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_27_layer_call_fn_63069

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_57711?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_29_layer_call_fn_63447

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58394w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58711

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?s
?
__inference__traced_save_64728
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop;
7savev2_batch_normalization_26_gamma_read_readvariableop:
6savev2_batch_normalization_26_beta_read_readvariableopA
=savev2_batch_normalization_26_moving_mean_read_readvariableopE
Asavev2_batch_normalization_26_moving_variance_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop;
7savev2_batch_normalization_28_gamma_read_readvariableop:
6savev2_batch_normalization_28_beta_read_readvariableopA
=savev2_batch_normalization_28_moving_mean_read_readvariableopE
Asavev2_batch_normalization_28_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop;
7savev2_batch_normalization_29_gamma_read_readvariableop:
6savev2_batch_normalization_29_beta_read_readvariableopA
=savev2_batch_normalization_29_moving_mean_read_readvariableopE
Asavev2_batch_normalization_29_moving_variance_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop;
7savev2_batch_normalization_30_gamma_read_readvariableop:
6savev2_batch_normalization_30_beta_read_readvariableopA
=savev2_batch_normalization_30_moving_mean_read_readvariableopE
Asavev2_batch_normalization_30_moving_variance_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop;
7savev2_batch_normalization_31_gamma_read_readvariableop:
6savev2_batch_normalization_31_beta_read_readvariableopA
=savev2_batch_normalization_31_moving_mean_read_readvariableopE
Asavev2_batch_normalization_31_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop;
7savev2_batch_normalization_32_gamma_read_readvariableop:
6savev2_batch_normalization_32_beta_read_readvariableopA
=savev2_batch_normalization_32_moving_mean_read_readvariableopE
Asavev2_batch_normalization_32_moving_variance_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop;
7savev2_batch_normalization_33_gamma_read_readvariableop:
6savev2_batch_normalization_33_beta_read_readvariableopA
=savev2_batch_normalization_33_moving_mean_read_readvariableopE
Asavev2_batch_normalization_33_moving_variance_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop;
7savev2_batch_normalization_34_gamma_read_readvariableop:
6savev2_batch_normalization_34_beta_read_readvariableopA
=savev2_batch_normalization_34_moving_mean_read_readvariableopE
Asavev2_batch_normalization_34_moving_variance_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop;
7savev2_batch_normalization_35_gamma_read_readvariableop:
6savev2_batch_normalization_35_beta_read_readvariableopA
=savev2_batch_normalization_35_moving_mean_read_readvariableopE
Asavev2_batch_normalization_35_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop7savev2_batch_normalization_26_gamma_read_readvariableop6savev2_batch_normalization_26_beta_read_readvariableop=savev2_batch_normalization_26_moving_mean_read_readvariableopAsavev2_batch_normalization_26_moving_variance_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop7savev2_batch_normalization_28_gamma_read_readvariableop6savev2_batch_normalization_28_beta_read_readvariableop=savev2_batch_normalization_28_moving_mean_read_readvariableopAsavev2_batch_normalization_28_moving_variance_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop7savev2_batch_normalization_29_gamma_read_readvariableop6savev2_batch_normalization_29_beta_read_readvariableop=savev2_batch_normalization_29_moving_mean_read_readvariableopAsavev2_batch_normalization_29_moving_variance_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop7savev2_batch_normalization_30_gamma_read_readvariableop6savev2_batch_normalization_30_beta_read_readvariableop=savev2_batch_normalization_30_moving_mean_read_readvariableopAsavev2_batch_normalization_30_moving_variance_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop7savev2_batch_normalization_31_gamma_read_readvariableop6savev2_batch_normalization_31_beta_read_readvariableop=savev2_batch_normalization_31_moving_mean_read_readvariableopAsavev2_batch_normalization_31_moving_variance_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableop7savev2_batch_normalization_32_gamma_read_readvariableop6savev2_batch_normalization_32_beta_read_readvariableop=savev2_batch_normalization_32_moving_mean_read_readvariableopAsavev2_batch_normalization_32_moving_variance_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop7savev2_batch_normalization_33_gamma_read_readvariableop6savev2_batch_normalization_33_beta_read_readvariableop=savev2_batch_normalization_33_moving_mean_read_readvariableopAsavev2_batch_normalization_33_moving_variance_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop7savev2_batch_normalization_34_gamma_read_readvariableop6savev2_batch_normalization_34_beta_read_readvariableop=savev2_batch_normalization_34_moving_mean_read_readvariableopAsavev2_batch_normalization_34_moving_variance_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop7savev2_batch_normalization_35_gamma_read_readvariableop6savev2_batch_normalization_35_beta_read_readvariableop=savev2_batch_normalization_35_moving_mean_read_readvariableopAsavev2_batch_normalization_35_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *M
dtypesC
A2??
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?: : : : : : :  : : : : : :  : : : : : : :::::::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,	(
&
_output_shapes
:  : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
::,!(
&
_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
::,'(
&
_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
::,-(
&
_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
::,3(
&
_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
::,9(
&
_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
::?

_output_shapes
: 
?
e
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_63848

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_34_layer_call_fn_64284

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59832w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_63177

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
-__inference_sequential_15_layer_call_fn_61982

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_15_layer_call_and_return_conditional_losses_57588w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_29_layer_call_fn_64375

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_60183w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?6
?
H__inference_sequential_19_layer_call_and_return_conditional_losses_62578

inputsU
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_5_biasadd_readvariableop_resource:<
.batch_normalization_32_readvariableop_resource:>
0batch_normalization_32_readvariableop_1_resource:M
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:
identity??%batch_normalization_32/AssignNewValue?'batch_normalization_32/AssignNewValue_1?6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_32/ReadVariableOp?'batch_normalization_32/ReadVariableOp_1?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOpN
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????00*
paddingVALID*
strides
?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3#conv2d_transpose_5/BiasAdd:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_32/AssignNewValueAssignVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource4batch_normalization_32/FusedBatchNormV3:batch_mean:07^batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_32/AssignNewValue_1AssignVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_32/FusedBatchNormV3:batch_variance:09^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_31/LeakyRelu	LeakyRelu+batch_normalization_32/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>}
IdentityIdentity&leaky_re_lu_31/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp&^batch_normalization_32/AssignNewValue(^batch_normalization_32/AssignNewValue_17^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2N
%batch_normalization_32/AssignNewValue%batch_normalization_32/AssignNewValue2R
'batch_normalization_32/AssignNewValue_1'batch_normalization_32/AssignNewValue_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_3_layer_call_fn_62830

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_57435w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_27_layer_call_fn_64069

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_59675w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62996

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64491

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60206

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
H__inference_sequential_19_layer_call_and_return_conditional_losses_59460

inputs2
conv2d_transpose_5_59444:&
conv2d_transpose_5_59446:*
batch_normalization_32_59449:*
batch_normalization_32_59451:*
batch_normalization_32_59453:*
batch_normalization_32_59455:
identity??.batch_normalization_32/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_5_59444conv2d_transpose_5_59446*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_59307?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_32_59449batch_normalization_32_59451batch_normalization_32_59453batch_normalization_32_59455*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59402?
leaky_re_lu_31/PartitionedCallPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_59345~
IdentityIdentity'leaky_re_lu_31/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_17_layer_call_fn_58556
conv2d_transpose_4_input!
unknown: 
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_17_layer_call_and_return_conditional_losses_58524w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:????????? 
2
_user_specified_nameconv2d_transpose_4_input
?
?
H__inference_sequential_17_layer_call_and_return_conditional_losses_58412

inputs2
conv2d_transpose_4_58372: &
conv2d_transpose_4_58374:*
batch_normalization_29_58395:*
batch_normalization_29_58397:*
batch_normalization_29_58399:*
batch_normalization_29_58401:
identity??.batch_normalization_29/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_4_58372conv2d_transpose_4_58374*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_58371?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0batch_normalization_29_58395batch_normalization_29_58397batch_normalization_29_58399batch_normalization_29_58401*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58394?
leaky_re_lu_28/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_58409~
IdentityIdentity'leaky_re_lu_28/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_29/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_18_layer_call_fn_62347

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_18_layer_call_and_return_conditional_losses_58830w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59330

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
-__inference_sequential_20_layer_call_fn_62636

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_59978w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
-__inference_sequential_20_layer_call_fn_60034
conv2d_27_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_27_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_59978w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_27_input
?!
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_58106

inputs)
conv2d_23_58075:  
conv2d_23_58077: *
batch_normalization_27_58080: *
batch_normalization_27_58082: *
batch_normalization_27_58084: *
batch_normalization_27_58086: )
conv2d_24_58090:  
conv2d_24_58092: *
batch_normalization_28_58095: *
batch_normalization_28_58097: *
batch_normalization_28_58099: *
batch_normalization_28_58101: 
identity??.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_23_58075conv2d_23_58077*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_57803?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_27_58080batch_normalization_27_58082batch_normalization_27_58084batch_normalization_27_58086*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_58020?
leaky_re_lu_26/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_57841?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_26/PartitionedCall:output:0conv2d_24_58090conv2d_24_58092*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_57853?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_28_58095batch_normalization_28_58097batch_normalization_28_58099batch_normalization_28_58101*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57960?
leaky_re_lu_27/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_57891~
IdentityIdentity'leaky_re_lu_27/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
H__inference_sequential_15_layer_call_and_return_conditional_losses_57639
conv2d_transpose_3_input2
conv2d_transpose_3_57623: &
conv2d_transpose_3_57625: *
batch_normalization_26_57628: *
batch_normalization_26_57630: *
batch_normalization_26_57632: *
batch_normalization_26_57634: 
identity??.batch_normalization_26/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_3_inputconv2d_transpose_3_57623conv2d_transpose_3_57625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_57435?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_26_57628batch_normalization_26_57630batch_normalization_26_57632batch_normalization_26_57634*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57458?
leaky_re_lu_25/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_57473~
IdentityIdentity'leaky_re_lu_25/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_26/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:i e
/
_output_shapes
:?????????
2
_user_specified_nameconv2d_transpose_3_input
?
e
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_57841

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_30_layer_call_fn_63600

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58762w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64014

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64356

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_63330

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63149

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63478

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
H__inference_sequential_15_layer_call_and_return_conditional_losses_62020

inputsU
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_3_biasadd_readvariableop_resource: <
.batch_normalization_26_readvariableop_resource: >
0batch_normalization_26_readvariableop_1_resource: M
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource: 
identity??6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOpN
conv2d_transpose_3/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3#conv2d_transpose_3/BiasAdd:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
leaky_re_lu_25/LeakyRelu	LeakyRelu+batch_normalization_26/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>}
IdentityIdentity&leaky_re_lu_25/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp7^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63784

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58331

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_32_layer_call_fn_63965

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59330w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
-__inference_sequential_17_layer_call_fn_62242

inputs!
unknown: 
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_17_layer_call_and_return_conditional_losses_58524w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_18_layer_call_fn_62376

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_18_layer_call_and_return_conditional_losses_59042w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58896

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64167

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_59725

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_4_layer_call_fn_63339

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_58271?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63667

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_62208

inputsB
(conv2d_23_conv2d_readvariableop_resource:  7
)conv2d_23_biasadd_readvariableop_resource: <
.batch_normalization_27_readvariableop_resource: >
0batch_normalization_27_readvariableop_1_resource: M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_24_conv2d_readvariableop_resource:  7
)conv2d_24_biasadd_readvariableop_resource: <
.batch_normalization_28_readvariableop_resource: >
0batch_normalization_28_readvariableop_1_resource: M
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource: 
identity??%batch_normalization_27/AssignNewValue?'batch_normalization_27/AssignNewValue_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?%batch_normalization_28/AssignNewValue?'batch_normalization_28/AssignNewValue_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_23/Conv2DConv2Dinputs'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3conv2d_23/BiasAdd:output:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_26/LeakyRelu	LeakyRelu+batch_normalization_27/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_24/Conv2DConv2D&leaky_re_lu_26/LeakyRelu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_24/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_28/AssignNewValueAssignVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource4batch_normalization_28/FusedBatchNormV3:batch_mean:07^batch_normalization_28/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_28/AssignNewValue_1AssignVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_28/FusedBatchNormV3:batch_variance:09^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_27/LeakyRelu	LeakyRelu+batch_normalization_28/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>}
IdentityIdentity&leaky_re_lu_27/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1&^batch_normalization_28/AssignNewValue(^batch_normalization_28/AssignNewValue_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12N
%batch_normalization_28/AssignNewValue%batch_normalization_28/AssignNewValue2R
'batch_normalization_28/AssignNewValue_1'batch_normalization_28/AssignNewValue_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59552

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_60183

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_63561

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_63408

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59583

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58394

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63167

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63631

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_57335

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_32_layer_call_fn_64208

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_59713h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_63024

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?%
?
H__inference_sequential_21_layer_call_and_return_conditional_losses_62812

inputsB
(conv2d_29_conv2d_readvariableop_resource:7
)conv2d_29_biasadd_readvariableop_resource:<
.batch_normalization_35_readvariableop_resource:>
0batch_normalization_35_readvariableop_1_resource:M
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:
identity??%batch_normalization_35/AssignNewValue?'batch_normalization_35/AssignNewValue_1?6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_35/ReadVariableOp?'batch_normalization_35/ReadVariableOp_1? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_29/Conv2DConv2Dinputs'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3conv2d_29/BiasAdd:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_35/AssignNewValueAssignVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource4batch_normalization_35/FusedBatchNormV3:batch_mean:07^batch_normalization_35/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_35/AssignNewValue_1AssignVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_35/FusedBatchNormV3:batch_variance:09^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
activation_1/SigmoidSigmoid+batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????00o
IdentityIdentityactivation_1/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp&^batch_normalization_35/AssignNewValue(^batch_normalization_35/AssignNewValue_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 2N
%batch_normalization_35/AssignNewValue%batch_normalization_35/AssignNewValue2R
'batch_normalization_35/AssignNewValue_1'batch_normalization_35/AssignNewValue_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_26_layer_call_fn_62942

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57530w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_28_layer_call_fn_63235

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57876w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
-__inference_sequential_15_layer_call_fn_57491
conv2d_transpose_3_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_15_layer_call_and_return_conditional_losses_57476w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:?????????
2
_user_specified_nameconv2d_transpose_3_input
?!
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_57894

inputs)
conv2d_23_57804:  
conv2d_23_57806: *
batch_normalization_27_57827: *
batch_normalization_27_57829: *
batch_normalization_27_57831: *
batch_normalization_27_57833: )
conv2d_24_57854:  
conv2d_24_57856: *
batch_normalization_28_57877: *
batch_normalization_28_57879: *
batch_normalization_28_57881: *
batch_normalization_28_57883: 
identity??.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_23_57804conv2d_23_57806*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_57803?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_27_57827batch_normalization_27_57829batch_normalization_27_57831batch_normalization_27_57833*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_57826?
leaky_re_lu_26/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_57841?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_26/PartitionedCall:output:0conv2d_24_57854conv2d_24_57856*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_57853?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_28_57877batch_normalization_28_57879batch_normalization_28_57881batch_normalization_28_57883*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57876?
leaky_re_lu_27/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_57891~
IdentityIdentity'leaky_re_lu_27/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59748

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_62867

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_17_layer_call_and_return_conditional_losses_58594
conv2d_transpose_4_input2
conv2d_transpose_4_58578: &
conv2d_transpose_4_58580:*
batch_normalization_29_58583:*
batch_normalization_29_58585:*
batch_normalization_29_58587:*
batch_normalization_29_58589:
identity??.batch_normalization_29/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_4_inputconv2d_transpose_4_58578conv2d_transpose_4_58580*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_58371?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0batch_normalization_29_58583batch_normalization_29_58585batch_normalization_29_58587batch_normalization_29_58589*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58466?
leaky_re_lu_28/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_58409~
IdentityIdentity'leaky_re_lu_28/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_29/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall:i e
/
_output_shapes
:????????? 
2
_user_specified_nameconv2d_transpose_4_input
?
?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57876

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58300

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_27_layer_call_fn_63056

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_57680?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_26_layer_call_fn_62929

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57458w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_conv2d_25_layer_call_fn_63551

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_58739w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_63926

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :0I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :0I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????00*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64455

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_19_layer_call_fn_62502

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_19_layer_call_and_return_conditional_losses_59460w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_25_layer_call_fn_63019

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_57473h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63820

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_59207

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_21_layer_call_fn_60239
conv2d_29_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_29_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_21_layer_call_and_return_conditional_losses_60224w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????00: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_29_input
?	
?
-__inference_sequential_19_layer_call_fn_59363
conv2d_transpose_5_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_19_layer_call_and_return_conditional_losses_59348w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:?????????
2
_user_specified_nameconv2d_transpose_5_input
?
?
-__inference_sequential_16_layer_call_fn_58162
conv2d_23_input!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_23_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? **
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_58106w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:????????? 
)
_user_specified_nameconv2d_23_input
?
?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63302

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57395

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_63903

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_64366

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????00*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64320

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_20_layer_call_fn_59793
conv2d_27_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_27_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_59766w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_27_input
?
?
)__inference_conv2d_24_layer_call_fn_63186

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_57853w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_33_layer_call_fn_64361

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_59763h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57530

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
H__inference_sequential_17_layer_call_and_return_conditional_losses_58575
conv2d_transpose_4_input2
conv2d_transpose_4_58559: &
conv2d_transpose_4_58561:*
batch_normalization_29_58564:*
batch_normalization_29_58566:*
batch_normalization_29_58568:*
batch_normalization_29_58570:
identity??.batch_normalization_29/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_4_inputconv2d_transpose_4_58559conv2d_transpose_4_58561*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_58371?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0batch_normalization_29_58564batch_normalization_29_58566batch_normalization_29_58568batch_normalization_29_58570*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58394?
leaky_re_lu_28/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_58409~
IdentityIdentity'leaky_re_lu_28/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_29/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall:i e
/
_output_shapes
:????????? 
2
_user_specified_nameconv2d_transpose_4_input
?
?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58762

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_35_layer_call_fn_64411

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60155?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_57826

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
H__inference_sequential_15_layer_call_and_return_conditional_losses_57658
conv2d_transpose_3_input2
conv2d_transpose_3_57642: &
conv2d_transpose_3_57644: *
batch_normalization_26_57647: *
batch_normalization_26_57649: *
batch_normalization_26_57651: *
batch_normalization_26_57653: 
identity??.batch_normalization_26/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_3_inputconv2d_transpose_3_57642conv2d_transpose_3_57644*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_57435?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_26_57647batch_normalization_26_57649batch_normalization_26_57651batch_normalization_26_57653*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57530?
leaky_re_lu_25/PartitionedCallPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_57473~
IdentityIdentity'leaky_re_lu_25/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_26/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:i e
/
_output_shapes
:?????????
2
_user_specified_nameconv2d_transpose_3_input
?
?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63284

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_34_layer_call_fn_64271

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59748w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_63714

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_57891

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_35_layer_call_fn_64437

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60278w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_28_layer_call_fn_63209

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57744?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_26_layer_call_fn_62916

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_57395?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58647

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_31_layer_call_fn_63727

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_58680?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_33_layer_call_fn_64092

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59552?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_15_layer_call_fn_61965

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_15_layer_call_and_return_conditional_losses_57476w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_57435

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
stack/1Const*
_output_shapes
: *
dtype0*
value	B :I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59236

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59616

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_model_4_layer_call_fn_61372

inputs
unknown:
??
	unknown_0:	?#
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:$

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:$

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:$

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:$

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:$

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:
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
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*L
_read_only_resource_inputs.
,*	
!"#$'()*-./034569:;<*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_4_layer_call_and_return_conditional_losses_60856w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58466

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
H__inference_sequential_18_layer_call_and_return_conditional_losses_62468

inputsB
(conv2d_25_conv2d_readvariableop_resource:7
)conv2d_25_biasadd_readvariableop_resource:<
.batch_normalization_30_readvariableop_resource:>
0batch_normalization_30_readvariableop_1_resource:M
?batch_normalization_30_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_26_conv2d_readvariableop_resource:7
)conv2d_26_biasadd_readvariableop_resource:<
.batch_normalization_31_readvariableop_resource:>
0batch_normalization_31_readvariableop_1_resource:M
?batch_normalization_31_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:
identity??%batch_normalization_30/AssignNewValue?'batch_normalization_30/AssignNewValue_1?6batch_normalization_30/FusedBatchNormV3/ReadVariableOp?8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_30/ReadVariableOp?'batch_normalization_30/ReadVariableOp_1?%batch_normalization_31/AssignNewValue?'batch_normalization_31/AssignNewValue_1?6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_31/ReadVariableOp?'batch_normalization_31/ReadVariableOp_1? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_25/Conv2DConv2Dinputs'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_30/ReadVariableOpReadVariableOp.batch_normalization_30_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_30/ReadVariableOp_1ReadVariableOp0batch_normalization_30_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_30/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_30/FusedBatchNormV3FusedBatchNormV3conv2d_25/BiasAdd:output:0-batch_normalization_30/ReadVariableOp:value:0/batch_normalization_30/ReadVariableOp_1:value:0>batch_normalization_30/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_30/AssignNewValueAssignVariableOp?batch_normalization_30_fusedbatchnormv3_readvariableop_resource4batch_normalization_30/FusedBatchNormV3:batch_mean:07^batch_normalization_30/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_30/AssignNewValue_1AssignVariableOpAbatch_normalization_30_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_30/FusedBatchNormV3:batch_variance:09^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_29/LeakyRelu	LeakyRelu+batch_normalization_30/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_26/Conv2DConv2D&leaky_re_lu_29/LeakyRelu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_31/ReadVariableOpReadVariableOp.batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_31/ReadVariableOp_1ReadVariableOp0batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_31/FusedBatchNormV3FusedBatchNormV3conv2d_26/BiasAdd:output:0-batch_normalization_31/ReadVariableOp:value:0/batch_normalization_31/ReadVariableOp_1:value:0>batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_31/AssignNewValueAssignVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource4batch_normalization_31/FusedBatchNormV3:batch_mean:07^batch_normalization_31/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_31/AssignNewValue_1AssignVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_31/FusedBatchNormV3:batch_variance:09^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_30/LeakyRelu	LeakyRelu+batch_normalization_31/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>}
IdentityIdentity&leaky_re_lu_30/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_30/AssignNewValue(^batch_normalization_30/AssignNewValue_17^batch_normalization_30/FusedBatchNormV3/ReadVariableOp9^batch_normalization_30/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_30/ReadVariableOp(^batch_normalization_30/ReadVariableOp_1&^batch_normalization_31/AssignNewValue(^batch_normalization_31/AssignNewValue_17^batch_normalization_31/FusedBatchNormV3/ReadVariableOp9^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_31/ReadVariableOp(^batch_normalization_31/ReadVariableOp_1!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2N
%batch_normalization_30/AssignNewValue%batch_normalization_30/AssignNewValue2R
'batch_normalization_30/AssignNewValue_1'batch_normalization_30/AssignNewValue_12p
6batch_normalization_30/FusedBatchNormV3/ReadVariableOp6batch_normalization_30/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_30/FusedBatchNormV3/ReadVariableOp_18batch_normalization_30/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_30/ReadVariableOp%batch_normalization_30/ReadVariableOp2R
'batch_normalization_30/ReadVariableOp_1'batch_normalization_30/ReadVariableOp_12N
%batch_normalization_31/AssignNewValue%batch_normalization_31/AssignNewValue2R
'batch_normalization_31/AssignNewValue_1'batch_normalization_31/AssignNewValue_12p
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp6batch_normalization_31/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_18batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_31/ReadVariableOp%batch_normalization_31/ReadVariableOp2R
'batch_normalization_31/ReadVariableOp_1'batch_normalization_31/ReadVariableOp_12D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_30_layer_call_fn_63613

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58956w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63532

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_61114

args_0
unknown:
??
	unknown_0:	?#
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:$

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:$

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:$

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:$

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:$

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_57294w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
H__inference_sequential_17_layer_call_and_return_conditional_losses_58524

inputs2
conv2d_transpose_4_58508: &
conv2d_transpose_4_58510:*
batch_normalization_29_58513:*
batch_normalization_29_58515:*
batch_normalization_29_58517:*
batch_normalization_29_58519:
identity??.batch_normalization_29/StatefulPartitionedCall?*conv2d_transpose_4/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_4_58508conv2d_transpose_4_58510*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_58371?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0batch_normalization_29_58513batch_normalization_29_58515batch_normalization_29_58517batch_normalization_29_58519*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_58466?
leaky_re_lu_28/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_58409~
IdentityIdentity'leaky_re_lu_28/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_29/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':????????? : : : : : : 2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_63542

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58956

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_60278

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_57803

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?!
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_59766

inputs)
conv2d_27_59676:
conv2d_27_59678:*
batch_normalization_33_59699:*
batch_normalization_33_59701:*
batch_normalization_33_59703:*
batch_normalization_33_59705:)
conv2d_28_59726:
conv2d_28_59728:*
batch_normalization_34_59749:*
batch_normalization_34_59751:*
batch_normalization_34_59753:*
batch_normalization_34_59755:
identity??.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_59676conv2d_27_59678*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_59675?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_33_59699batch_normalization_33_59701batch_normalization_33_59703batch_normalization_33_59705*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59698?
leaky_re_lu_32/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_59713?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0conv2d_28_59726conv2d_28_59728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_59725?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_34_59749batch_normalization_34_59751batch_normalization_34_59753batch_normalization_34_59755*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59748?
leaky_re_lu_33/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_59763~
IdentityIdentity'leaky_re_lu_33/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_59675

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
H__inference_sequential_19_layer_call_and_return_conditional_losses_59530
conv2d_transpose_5_input2
conv2d_transpose_5_59514:&
conv2d_transpose_5_59516:*
batch_normalization_32_59519:*
batch_normalization_32_59521:*
batch_normalization_32_59523:*
batch_normalization_32_59525:
identity??.batch_normalization_32/StatefulPartitionedCall?*conv2d_transpose_5/StatefulPartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_5_inputconv2d_transpose_5_59514conv2d_transpose_5_59516*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_59307?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0batch_normalization_32_59519batch_normalization_32_59521batch_normalization_32_59523batch_normalization_32_59525*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_59402?
leaky_re_lu_31/PartitionedCallPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_59345~
IdentityIdentity'leaky_re_lu_31/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:i e
/
_output_shapes
:?????????
2
_user_specified_nameconv2d_transpose_5_input
?/
?
H__inference_sequential_19_layer_call_and_return_conditional_losses_62540

inputsU
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_5_biasadd_readvariableop_resource:<
.batch_normalization_32_readvariableop_resource:>
0batch_normalization_32_readvariableop_1_resource:M
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:
identity??6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_32/ReadVariableOp?'batch_normalization_32/ReadVariableOp_1?)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOpN
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :0\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :0\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????00*
paddingVALID*
strides
?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3#conv2d_transpose_5/BiasAdd:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_31/LeakyRelu	LeakyRelu+batch_normalization_32/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>}
IdentityIdentity&leaky_re_lu_31/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp7^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_34_layer_call_fn_64245

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59616?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57744

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_18_layer_call_fn_59098
conv2d_25_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_18_layer_call_and_return_conditional_losses_59042w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_25_input
?#
?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_63385

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?"
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_58230
conv2d_23_input)
conv2d_23_58199:  
conv2d_23_58201: *
batch_normalization_27_58204: *
batch_normalization_27_58206: *
batch_normalization_27_58208: *
batch_normalization_27_58210: )
conv2d_24_58214:  
conv2d_24_58216: *
batch_normalization_28_58219: *
batch_normalization_28_58221: *
batch_normalization_28_58223: *
batch_normalization_28_58225: 
identity??.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallconv2d_23_inputconv2d_23_58199conv2d_23_58201*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_57803?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_27_58204batch_normalization_27_58206batch_normalization_27_58208batch_normalization_27_58210*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_58020?
leaky_re_lu_26/PartitionedCallPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_57841?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_26/PartitionedCall:output:0conv2d_24_58214conv2d_24_58216*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_57853?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_28_58219batch_normalization_28_58221batch_normalization_28_58223batch_normalization_28_58225*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_57960?
leaky_re_lu_27/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_57891~
IdentityIdentity'leaky_re_lu_27/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall:` \
/
_output_shapes
:????????? 
)
_user_specified_nameconv2d_23_input
?
?
6__inference_batch_normalization_27_layer_call_fn_63095

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_58020w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?!
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_59978

inputs)
conv2d_27_59947:
conv2d_27_59949:*
batch_normalization_33_59952:*
batch_normalization_33_59954:*
batch_normalization_33_59956:*
batch_normalization_33_59958:)
conv2d_28_59962:
conv2d_28_59964:*
batch_normalization_34_59967:*
batch_normalization_34_59969:*
batch_normalization_34_59971:*
batch_normalization_34_59973:
identity??.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_59947conv2d_27_59949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_59675?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_33_59952batch_normalization_33_59954batch_normalization_33_59956batch_normalization_33_59958*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_59892?
leaky_re_lu_32/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_59713?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0conv2d_28_59962conv2d_28_59964*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_59725?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0batch_normalization_34_59967batch_normalization_34_59969batch_normalization_34_59971batch_normalization_34_59973*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_59832?
leaky_re_lu_33/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_59763~
IdentityIdentity'leaky_re_lu_33/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_63695

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_27_layer_call_fn_63325

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_57891h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_58616

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_64213

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????00*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
:
args_00
serving_default_args_0:0??????????I
sequential_218
StatefulPartitionedCall:0?????????00tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
!layer_with_weights-0
!layer-0
"layer_with_weights-1
"layer-1
#layer-2
$layer_with_weights-2
$layer-3
%layer_with_weights-3
%layer-4
&layer-5
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
+layer_with_weights-0
+layer-0
,layer_with_weights-1
,layer-1
-layer-2
.	variables
/trainable_variables
0regularization_losses
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
2layer_with_weights-0
2layer-0
3layer_with_weights-1
3layer-1
4layer-2
5layer_with_weights-2
5layer-3
6layer_with_weights-3
6layer-4
7layer-5
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
<layer_with_weights-0
<layer-0
=layer_with_weights-1
=layer-1
>layer-2
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
Clayer_with_weights-0
Clayer-0
Dlayer_with_weights-1
Dlayer-1
Elayer-2
Flayer_with_weights-2
Flayer-3
Glayer_with_weights-3
Glayer-4
Hlayer-5
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
Mlayer_with_weights-0
Mlayer-0
Nlayer_with_weights-1
Nlayer-1
Olayer-2
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
0
1
T2
U3
V4
W5
X6
Y7
Z8
[9
\10
]11
^12
_13
`14
a15
b16
c17
d18
e19
f20
g21
h22
i23
j24
k25
l26
m27
n28
o29
p30
q31
r32
s33
t34
u35
v36
w37
x38
y39
z40
{41
|42
}43
~44
45
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
?60
?61"
trackable_list_wrapper
?
0
1
T2
U3
V4
W5
Z6
[7
\8
]9
`10
a11
b12
c13
f14
g15
h16
i17
l18
m19
n20
o21
r22
s23
t24
u25
x26
y27
z28
{29
~30
31
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
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
": 
??2dense_1/kernel
:?2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Tkernel
Ubias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
T0
U1
V2
W3
X4
Y5"
trackable_list_wrapper
<
T0
U1
V2
W3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Zkernel
[bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	\gamma
]beta
^moving_mean
_moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

`kernel
abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	bgamma
cbeta
dmoving_mean
emoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
v
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11"
trackable_list_wrapper
X
Z0
[1
\2
]3
`4
a5
b6
c7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

fkernel
gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	hgamma
ibeta
jmoving_mean
kmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
f0
g1
h2
i3
j4
k5"
trackable_list_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

lkernel
mbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	ngamma
obeta
pmoving_mean
qmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

rkernel
sbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	tgamma
ubeta
vmoving_mean
wmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
v
l0
m1
n2
o3
p4
q5
r6
s7
t8
u9
v10
w11"
trackable_list_wrapper
X
l0
m1
n2
o3
r4
s5
t6
u7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

xkernel
ybias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	zgamma
{beta
|moving_mean
}moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
x0
y1
z2
{3
|4
}5"
trackable_list_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

~kernel
bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
~0
1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11"
trackable_list_wrapper
^
~0
1
?2
?3
?4
?5
?6
?7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1 2conv2d_transpose_3/kernel
%:# 2conv2d_transpose_3/bias
*:( 2batch_normalization_26/gamma
):' 2batch_normalization_26/beta
2:0  (2"batch_normalization_26/moving_mean
6:4  (2&batch_normalization_26/moving_variance
*:(  2conv2d_23/kernel
: 2conv2d_23/bias
*:( 2batch_normalization_27/gamma
):' 2batch_normalization_27/beta
2:0  (2"batch_normalization_27/moving_mean
6:4  (2&batch_normalization_27/moving_variance
*:(  2conv2d_24/kernel
: 2conv2d_24/bias
*:( 2batch_normalization_28/gamma
):' 2batch_normalization_28/beta
2:0  (2"batch_normalization_28/moving_mean
6:4  (2&batch_normalization_28/moving_variance
3:1 2conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
*:(2batch_normalization_29/gamma
):'2batch_normalization_29/beta
2:0 (2"batch_normalization_29/moving_mean
6:4 (2&batch_normalization_29/moving_variance
*:(2conv2d_25/kernel
:2conv2d_25/bias
*:(2batch_normalization_30/gamma
):'2batch_normalization_30/beta
2:0 (2"batch_normalization_30/moving_mean
6:4 (2&batch_normalization_30/moving_variance
*:(2conv2d_26/kernel
:2conv2d_26/bias
*:(2batch_normalization_31/gamma
):'2batch_normalization_31/beta
2:0 (2"batch_normalization_31/moving_mean
6:4 (2&batch_normalization_31/moving_variance
3:12conv2d_transpose_5/kernel
%:#2conv2d_transpose_5/bias
*:(2batch_normalization_32/gamma
):'2batch_normalization_32/beta
2:0 (2"batch_normalization_32/moving_mean
6:4 (2&batch_normalization_32/moving_variance
*:(2conv2d_27/kernel
:2conv2d_27/bias
*:(2batch_normalization_33/gamma
):'2batch_normalization_33/beta
2:0 (2"batch_normalization_33/moving_mean
6:4 (2&batch_normalization_33/moving_variance
*:(2conv2d_28/kernel
:2conv2d_28/bias
*:(2batch_normalization_34/gamma
):'2batch_normalization_34/beta
2:0 (2"batch_normalization_34/moving_mean
6:4 (2&batch_normalization_34/moving_variance
*:(2conv2d_29/kernel
:2conv2d_29/bias
*:(2batch_normalization_35/gamma
):'2batch_normalization_35/beta
2:0 (2"batch_normalization_35/moving_mean
6:4 (2&batch_normalization_35/moving_variance
?
X0
Y1
^2
_3
d4
e5
j6
k7
p8
q9
v10
w11
|12
}13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
f
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
9"
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
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
V0
W1
X2
Y3"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
X0
Y1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
\0
]1
^2
_3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
b0
c1
d2
e3"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
<
^0
_1
d2
e3"
trackable_list_wrapper
J
!0
"1
#2
$3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
h0
i1
j2
k3"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
j0
k1"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
n0
o1
p2
q3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
t0
u1
v2
w3"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
<
p0
q1
v2
w3"
trackable_list_wrapper
J
20
31
42
53
64
75"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
z0
{1
|2
}3"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
|0
}1"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
@
?0
?1
?2
?3"
trackable_list_wrapper
J
C0
D1
E2
F3
G4
H5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
5
M0
N1
O2"
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
.
X0
Y1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
^0
_1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
d0
e1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
j0
k1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
p0
q1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
v0
w1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
|0
}1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
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
?2?
'__inference_model_4_layer_call_fn_61243
'__inference_model_4_layer_call_fn_61372?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_4_layer_call_and_return_conditional_losses_61641
B__inference_model_4_layer_call_and_return_conditional_losses_61910?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_57294args_0"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_61919?
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
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_61929?
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
?2?
)__inference_reshape_1_layer_call_fn_61934?
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
?2?
D__inference_reshape_1_layer_call_and_return_conditional_losses_61948?
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
?2?
-__inference_sequential_15_layer_call_fn_57491
-__inference_sequential_15_layer_call_fn_61965
-__inference_sequential_15_layer_call_fn_61982
-__inference_sequential_15_layer_call_fn_57620?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_15_layer_call_and_return_conditional_losses_62020
H__inference_sequential_15_layer_call_and_return_conditional_losses_62058
H__inference_sequential_15_layer_call_and_return_conditional_losses_57639
H__inference_sequential_15_layer_call_and_return_conditional_losses_57658?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_16_layer_call_fn_57921
-__inference_sequential_16_layer_call_fn_62087
-__inference_sequential_16_layer_call_fn_62116
-__inference_sequential_16_layer_call_fn_58162?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_16_layer_call_and_return_conditional_losses_62162
H__inference_sequential_16_layer_call_and_return_conditional_losses_62208
H__inference_sequential_16_layer_call_and_return_conditional_losses_58196
H__inference_sequential_16_layer_call_and_return_conditional_losses_58230?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_17_layer_call_fn_58427
-__inference_sequential_17_layer_call_fn_62225
-__inference_sequential_17_layer_call_fn_62242
-__inference_sequential_17_layer_call_fn_58556?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_17_layer_call_and_return_conditional_losses_62280
H__inference_sequential_17_layer_call_and_return_conditional_losses_62318
H__inference_sequential_17_layer_call_and_return_conditional_losses_58575
H__inference_sequential_17_layer_call_and_return_conditional_losses_58594?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_18_layer_call_fn_58857
-__inference_sequential_18_layer_call_fn_62347
-__inference_sequential_18_layer_call_fn_62376
-__inference_sequential_18_layer_call_fn_59098?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_18_layer_call_and_return_conditional_losses_62422
H__inference_sequential_18_layer_call_and_return_conditional_losses_62468
H__inference_sequential_18_layer_call_and_return_conditional_losses_59132
H__inference_sequential_18_layer_call_and_return_conditional_losses_59166?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_19_layer_call_fn_59363
-__inference_sequential_19_layer_call_fn_62485
-__inference_sequential_19_layer_call_fn_62502
-__inference_sequential_19_layer_call_fn_59492?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_19_layer_call_and_return_conditional_losses_62540
H__inference_sequential_19_layer_call_and_return_conditional_losses_62578
H__inference_sequential_19_layer_call_and_return_conditional_losses_59511
H__inference_sequential_19_layer_call_and_return_conditional_losses_59530?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_20_layer_call_fn_59793
-__inference_sequential_20_layer_call_fn_62607
-__inference_sequential_20_layer_call_fn_62636
-__inference_sequential_20_layer_call_fn_60034?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_20_layer_call_and_return_conditional_losses_62682
H__inference_sequential_20_layer_call_and_return_conditional_losses_62728
H__inference_sequential_20_layer_call_and_return_conditional_losses_60068
H__inference_sequential_20_layer_call_and_return_conditional_losses_60102?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_21_layer_call_fn_60239
-__inference_sequential_21_layer_call_fn_62745
-__inference_sequential_21_layer_call_fn_62762
-__inference_sequential_21_layer_call_fn_60369?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_21_layer_call_and_return_conditional_losses_62787
H__inference_sequential_21_layer_call_and_return_conditional_losses_62812
H__inference_sequential_21_layer_call_and_return_conditional_losses_60388
H__inference_sequential_21_layer_call_and_return_conditional_losses_60407?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_61114args_0"?
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
?2?
2__inference_conv2d_transpose_3_layer_call_fn_62821
2__inference_conv2d_transpose_3_layer_call_fn_62830?
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
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_62867
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_62890?
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
?2?
6__inference_batch_normalization_26_layer_call_fn_62903
6__inference_batch_normalization_26_layer_call_fn_62916
6__inference_batch_normalization_26_layer_call_fn_62929
6__inference_batch_normalization_26_layer_call_fn_62942?
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
?2?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62960
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62978
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62996
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_63014?
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
?2?
.__inference_leaky_re_lu_25_layer_call_fn_63019?
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
?2?
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_63024?
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
?2?
)__inference_conv2d_23_layer_call_fn_63033?
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
?2?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_63043?
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
?2?
6__inference_batch_normalization_27_layer_call_fn_63056
6__inference_batch_normalization_27_layer_call_fn_63069
6__inference_batch_normalization_27_layer_call_fn_63082
6__inference_batch_normalization_27_layer_call_fn_63095?
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
?2?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63113
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63131
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63149
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63167?
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
?2?
.__inference_leaky_re_lu_26_layer_call_fn_63172?
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
?2?
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_63177?
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
?2?
)__inference_conv2d_24_layer_call_fn_63186?
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
?2?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_63196?
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
?2?
6__inference_batch_normalization_28_layer_call_fn_63209
6__inference_batch_normalization_28_layer_call_fn_63222
6__inference_batch_normalization_28_layer_call_fn_63235
6__inference_batch_normalization_28_layer_call_fn_63248?
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
?2?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63266
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63284
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63302
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63320?
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
?2?
.__inference_leaky_re_lu_27_layer_call_fn_63325?
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
?2?
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_63330?
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
2__inference_conv2d_transpose_4_layer_call_fn_63339
2__inference_conv2d_transpose_4_layer_call_fn_63348?
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
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_63385
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_63408?
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
?2?
6__inference_batch_normalization_29_layer_call_fn_63421
6__inference_batch_normalization_29_layer_call_fn_63434
6__inference_batch_normalization_29_layer_call_fn_63447
6__inference_batch_normalization_29_layer_call_fn_63460?
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
?2?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63478
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63496
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63514
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63532?
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
?2?
.__inference_leaky_re_lu_28_layer_call_fn_63537?
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
?2?
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_63542?
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
?2?
)__inference_conv2d_25_layer_call_fn_63551?
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
?2?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_63561?
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
?2?
6__inference_batch_normalization_30_layer_call_fn_63574
6__inference_batch_normalization_30_layer_call_fn_63587
6__inference_batch_normalization_30_layer_call_fn_63600
6__inference_batch_normalization_30_layer_call_fn_63613?
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
?2?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63631
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63649
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63667
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63685?
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
?2?
.__inference_leaky_re_lu_29_layer_call_fn_63690?
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
?2?
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_63695?
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
?2?
)__inference_conv2d_26_layer_call_fn_63704?
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
?2?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_63714?
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
?2?
6__inference_batch_normalization_31_layer_call_fn_63727
6__inference_batch_normalization_31_layer_call_fn_63740
6__inference_batch_normalization_31_layer_call_fn_63753
6__inference_batch_normalization_31_layer_call_fn_63766?
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
?2?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63784
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63802
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63820
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63838?
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
?2?
.__inference_leaky_re_lu_30_layer_call_fn_63843?
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
?2?
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_63848?
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
2__inference_conv2d_transpose_5_layer_call_fn_63857
2__inference_conv2d_transpose_5_layer_call_fn_63866?
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
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_63903
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_63926?
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
?2?
6__inference_batch_normalization_32_layer_call_fn_63939
6__inference_batch_normalization_32_layer_call_fn_63952
6__inference_batch_normalization_32_layer_call_fn_63965
6__inference_batch_normalization_32_layer_call_fn_63978?
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
?2?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_63996
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64014
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64032
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64050?
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
?2?
.__inference_leaky_re_lu_31_layer_call_fn_64055?
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
?2?
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_64060?
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
?2?
)__inference_conv2d_27_layer_call_fn_64069?
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
?2?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_64079?
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
?2?
6__inference_batch_normalization_33_layer_call_fn_64092
6__inference_batch_normalization_33_layer_call_fn_64105
6__inference_batch_normalization_33_layer_call_fn_64118
6__inference_batch_normalization_33_layer_call_fn_64131?
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
?2?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64149
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64167
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64185
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64203?
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
?2?
.__inference_leaky_re_lu_32_layer_call_fn_64208?
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
?2?
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_64213?
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
?2?
)__inference_conv2d_28_layer_call_fn_64222?
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
?2?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_64232?
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
?2?
6__inference_batch_normalization_34_layer_call_fn_64245
6__inference_batch_normalization_34_layer_call_fn_64258
6__inference_batch_normalization_34_layer_call_fn_64271
6__inference_batch_normalization_34_layer_call_fn_64284?
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
?2?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64302
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64320
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64338
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64356?
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
?2?
.__inference_leaky_re_lu_33_layer_call_fn_64361?
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
?2?
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_64366?
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
?2?
)__inference_conv2d_29_layer_call_fn_64375?
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
?2?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_64385?
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
?2?
6__inference_batch_normalization_35_layer_call_fn_64398
6__inference_batch_normalization_35_layer_call_fn_64411
6__inference_batch_normalization_35_layer_call_fn_64424
6__inference_batch_normalization_35_layer_call_fn_64437?
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
?2?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64455
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64473
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64491
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64509?
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
?2?
,__inference_activation_1_layer_call_fn_64514?
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
?2?
G__inference_activation_1_layer_call_and_return_conditional_losses_64519?
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
 ?
 __inference__wrapped_model_57294?NTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~????????????????0?-
&?#
!?
args_0??????????
? "E?B
@
sequential_21/?,
sequential_21?????????00?
G__inference_activation_1_layer_call_and_return_conditional_losses_64519h7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
,__inference_activation_1_layer_call_fn_64514[7?4
-?*
(?%
inputs?????????00
? " ??????????00?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62960?VWXYM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62978?VWXYM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_62996rVWXY;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
Q__inference_batch_normalization_26_layer_call_and_return_conditional_losses_63014rVWXY;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
6__inference_batch_normalization_26_layer_call_fn_62903?VWXYM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_26_layer_call_fn_62916?VWXYM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_26_layer_call_fn_62929eVWXY;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
6__inference_batch_normalization_26_layer_call_fn_62942eVWXY;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63113?\]^_M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63131?\]^_M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63149r\]^_;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
Q__inference_batch_normalization_27_layer_call_and_return_conditional_losses_63167r\]^_;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
6__inference_batch_normalization_27_layer_call_fn_63056?\]^_M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_27_layer_call_fn_63069?\]^_M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_27_layer_call_fn_63082e\]^_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
6__inference_batch_normalization_27_layer_call_fn_63095e\]^_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63266?bcdeM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63284?bcdeM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63302rbcde;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
Q__inference_batch_normalization_28_layer_call_and_return_conditional_losses_63320rbcde;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
6__inference_batch_normalization_28_layer_call_fn_63209?bcdeM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_28_layer_call_fn_63222?bcdeM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_28_layer_call_fn_63235ebcde;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
6__inference_batch_normalization_28_layer_call_fn_63248ebcde;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63478?hijkM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63496?hijkM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63514rhijk;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
Q__inference_batch_normalization_29_layer_call_and_return_conditional_losses_63532rhijk;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
6__inference_batch_normalization_29_layer_call_fn_63421?hijkM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_29_layer_call_fn_63434?hijkM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_29_layer_call_fn_63447ehijk;?8
1?.
(?%
inputs?????????
p 
? " ???????????
6__inference_batch_normalization_29_layer_call_fn_63460ehijk;?8
1?.
(?%
inputs?????????
p
? " ???????????
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63631?nopqM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63649?nopqM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63667rnopq;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
Q__inference_batch_normalization_30_layer_call_and_return_conditional_losses_63685rnopq;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
6__inference_batch_normalization_30_layer_call_fn_63574?nopqM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_30_layer_call_fn_63587?nopqM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_30_layer_call_fn_63600enopq;?8
1?.
(?%
inputs?????????
p 
? " ???????????
6__inference_batch_normalization_30_layer_call_fn_63613enopq;?8
1?.
(?%
inputs?????????
p
? " ???????????
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63784?tuvwM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63802?tuvwM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63820rtuvw;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
Q__inference_batch_normalization_31_layer_call_and_return_conditional_losses_63838rtuvw;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
6__inference_batch_normalization_31_layer_call_fn_63727?tuvwM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_31_layer_call_fn_63740?tuvwM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_31_layer_call_fn_63753etuvw;?8
1?.
(?%
inputs?????????
p 
? " ???????????
6__inference_batch_normalization_31_layer_call_fn_63766etuvw;?8
1?.
(?%
inputs?????????
p
? " ???????????
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_63996?z{|}M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64014?z{|}M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64032rz{|};?8
1?.
(?%
inputs?????????00
p 
? "-?*
#? 
0?????????00
? ?
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_64050rz{|};?8
1?.
(?%
inputs?????????00
p
? "-?*
#? 
0?????????00
? ?
6__inference_batch_normalization_32_layer_call_fn_63939?z{|}M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_32_layer_call_fn_63952?z{|}M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_32_layer_call_fn_63965ez{|};?8
1?.
(?%
inputs?????????00
p 
? " ??????????00?
6__inference_batch_normalization_32_layer_call_fn_63978ez{|};?8
1?.
(?%
inputs?????????00
p
? " ??????????00?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64149?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64167?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64185v????;?8
1?.
(?%
inputs?????????00
p 
? "-?*
#? 
0?????????00
? ?
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_64203v????;?8
1?.
(?%
inputs?????????00
p
? "-?*
#? 
0?????????00
? ?
6__inference_batch_normalization_33_layer_call_fn_64092?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_33_layer_call_fn_64105?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_33_layer_call_fn_64118i????;?8
1?.
(?%
inputs?????????00
p 
? " ??????????00?
6__inference_batch_normalization_33_layer_call_fn_64131i????;?8
1?.
(?%
inputs?????????00
p
? " ??????????00?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64302?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64320?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64338v????;?8
1?.
(?%
inputs?????????00
p 
? "-?*
#? 
0?????????00
? ?
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_64356v????;?8
1?.
(?%
inputs?????????00
p
? "-?*
#? 
0?????????00
? ?
6__inference_batch_normalization_34_layer_call_fn_64245?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_34_layer_call_fn_64258?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_34_layer_call_fn_64271i????;?8
1?.
(?%
inputs?????????00
p 
? " ??????????00?
6__inference_batch_normalization_34_layer_call_fn_64284i????;?8
1?.
(?%
inputs?????????00
p
? " ??????????00?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64455?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64473?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64491v????;?8
1?.
(?%
inputs?????????00
p 
? "-?*
#? 
0?????????00
? ?
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_64509v????;?8
1?.
(?%
inputs?????????00
p
? "-?*
#? 
0?????????00
? ?
6__inference_batch_normalization_35_layer_call_fn_64398?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_35_layer_call_fn_64411?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_35_layer_call_fn_64424i????;?8
1?.
(?%
inputs?????????00
p 
? " ??????????00?
6__inference_batch_normalization_35_layer_call_fn_64437i????;?8
1?.
(?%
inputs?????????00
p
? " ??????????00?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_63043lZ[7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_23_layer_call_fn_63033_Z[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_63196l`a7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_24_layer_call_fn_63186_`a7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_63561llm7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_25_layer_call_fn_63551_lm7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_conv2d_26_layer_call_and_return_conditional_losses_63714lrs7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_26_layer_call_fn_63704_rs7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_conv2d_27_layer_call_and_return_conditional_losses_64079l~7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
)__inference_conv2d_27_layer_call_fn_64069_~7?4
-?*
(?%
inputs?????????00
? " ??????????00?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_64232n??7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
)__inference_conv2d_28_layer_call_fn_64222a??7?4
-?*
(?%
inputs?????????00
? " ??????????00?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_64385n??7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
)__inference_conv2d_29_layer_call_fn_64375a??7?4
-?*
(?%
inputs?????????00
? " ??????????00?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_62867?TUI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_62890lTU7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
2__inference_conv2d_transpose_3_layer_call_fn_62821?TUI?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
2__inference_conv2d_transpose_3_layer_call_fn_62830_TU7?4
-?*
(?%
inputs?????????
? " ?????????? ?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_63385?fgI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_63408lfg7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
2__inference_conv2d_transpose_4_layer_call_fn_63339?fgI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
2__inference_conv2d_transpose_4_layer_call_fn_63348_fg7?4
-?*
(?%
inputs????????? 
? " ???????????
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_63903?xyI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_63926lxy7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????00
? ?
2__inference_conv2d_transpose_5_layer_call_fn_63857?xyI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
2__inference_conv2d_transpose_5_layer_call_fn_63866_xy7?4
-?*
(?%
inputs?????????
? " ??????????00?
B__inference_dense_1_layer_call_and_return_conditional_losses_61929^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_1_layer_call_fn_61919Q0?-
&?#
!?
inputs??????????
? "????????????
I__inference_leaky_re_lu_25_layer_call_and_return_conditional_losses_63024h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_leaky_re_lu_25_layer_call_fn_63019[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
I__inference_leaky_re_lu_26_layer_call_and_return_conditional_losses_63177h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_leaky_re_lu_26_layer_call_fn_63172[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_63330h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_leaky_re_lu_27_layer_call_fn_63325[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_63542h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
.__inference_leaky_re_lu_28_layer_call_fn_63537[7?4
-?*
(?%
inputs?????????
? " ???????????
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_63695h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
.__inference_leaky_re_lu_29_layer_call_fn_63690[7?4
-?*
(?%
inputs?????????
? " ???????????
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_63848h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
.__inference_leaky_re_lu_30_layer_call_fn_63843[7?4
-?*
(?%
inputs?????????
? " ???????????
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_64060h7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
.__inference_leaky_re_lu_31_layer_call_fn_64055[7?4
-?*
(?%
inputs?????????00
? " ??????????00?
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_64213h7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
.__inference_leaky_re_lu_32_layer_call_fn_64208[7?4
-?*
(?%
inputs?????????00
? " ??????????00?
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_64366h7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
.__inference_leaky_re_lu_33_layer_call_fn_64361[7?4
-?*
(?%
inputs?????????00
? " ??????????00?
B__inference_model_4_layer_call_and_return_conditional_losses_61641?NTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~????????????????8?5
.?+
!?
inputs??????????
p 

 
? "-?*
#? 
0?????????00
? ?
B__inference_model_4_layer_call_and_return_conditional_losses_61910?NTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~????????????????8?5
.?+
!?
inputs??????????
p

 
? "-?*
#? 
0?????????00
? ?
'__inference_model_4_layer_call_fn_61243?NTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~????????????????8?5
.?+
!?
inputs??????????
p 

 
? " ??????????00?
'__inference_model_4_layer_call_fn_61372?NTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~????????????????8?5
.?+
!?
inputs??????????
p

 
? " ??????????00?
D__inference_reshape_1_layer_call_and_return_conditional_losses_61948a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_1_layer_call_fn_61934T0?-
&?#
!?
inputs??????????
? " ???????????
H__inference_sequential_15_layer_call_and_return_conditional_losses_57639?TUVWXYQ?N
G?D
:?7
conv2d_transpose_3_input?????????
p 

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_15_layer_call_and_return_conditional_losses_57658?TUVWXYQ?N
G?D
:?7
conv2d_transpose_3_input?????????
p

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_15_layer_call_and_return_conditional_losses_62020xTUVWXY??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_15_layer_call_and_return_conditional_losses_62058xTUVWXY??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0????????? 
? ?
-__inference_sequential_15_layer_call_fn_57491}TUVWXYQ?N
G?D
:?7
conv2d_transpose_3_input?????????
p 

 
? " ?????????? ?
-__inference_sequential_15_layer_call_fn_57620}TUVWXYQ?N
G?D
:?7
conv2d_transpose_3_input?????????
p

 
? " ?????????? ?
-__inference_sequential_15_layer_call_fn_61965kTUVWXY??<
5?2
(?%
inputs?????????
p 

 
? " ?????????? ?
-__inference_sequential_15_layer_call_fn_61982kTUVWXY??<
5?2
(?%
inputs?????????
p

 
? " ?????????? ?
H__inference_sequential_16_layer_call_and_return_conditional_losses_58196?Z[\]^_`abcdeH?E
>?;
1?.
conv2d_23_input????????? 
p 

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_16_layer_call_and_return_conditional_losses_58230?Z[\]^_`abcdeH?E
>?;
1?.
conv2d_23_input????????? 
p

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_16_layer_call_and_return_conditional_losses_62162~Z[\]^_`abcde??<
5?2
(?%
inputs????????? 
p 

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_16_layer_call_and_return_conditional_losses_62208~Z[\]^_`abcde??<
5?2
(?%
inputs????????? 
p

 
? "-?*
#? 
0????????? 
? ?
-__inference_sequential_16_layer_call_fn_57921zZ[\]^_`abcdeH?E
>?;
1?.
conv2d_23_input????????? 
p 

 
? " ?????????? ?
-__inference_sequential_16_layer_call_fn_58162zZ[\]^_`abcdeH?E
>?;
1?.
conv2d_23_input????????? 
p

 
? " ?????????? ?
-__inference_sequential_16_layer_call_fn_62087qZ[\]^_`abcde??<
5?2
(?%
inputs????????? 
p 

 
? " ?????????? ?
-__inference_sequential_16_layer_call_fn_62116qZ[\]^_`abcde??<
5?2
(?%
inputs????????? 
p

 
? " ?????????? ?
H__inference_sequential_17_layer_call_and_return_conditional_losses_58575?fghijkQ?N
G?D
:?7
conv2d_transpose_4_input????????? 
p 

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_17_layer_call_and_return_conditional_losses_58594?fghijkQ?N
G?D
:?7
conv2d_transpose_4_input????????? 
p

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_17_layer_call_and_return_conditional_losses_62280xfghijk??<
5?2
(?%
inputs????????? 
p 

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_17_layer_call_and_return_conditional_losses_62318xfghijk??<
5?2
(?%
inputs????????? 
p

 
? "-?*
#? 
0?????????
? ?
-__inference_sequential_17_layer_call_fn_58427}fghijkQ?N
G?D
:?7
conv2d_transpose_4_input????????? 
p 

 
? " ???????????
-__inference_sequential_17_layer_call_fn_58556}fghijkQ?N
G?D
:?7
conv2d_transpose_4_input????????? 
p

 
? " ???????????
-__inference_sequential_17_layer_call_fn_62225kfghijk??<
5?2
(?%
inputs????????? 
p 

 
? " ???????????
-__inference_sequential_17_layer_call_fn_62242kfghijk??<
5?2
(?%
inputs????????? 
p

 
? " ???????????
H__inference_sequential_18_layer_call_and_return_conditional_losses_59132?lmnopqrstuvwH?E
>?;
1?.
conv2d_25_input?????????
p 

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_18_layer_call_and_return_conditional_losses_59166?lmnopqrstuvwH?E
>?;
1?.
conv2d_25_input?????????
p

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_18_layer_call_and_return_conditional_losses_62422~lmnopqrstuvw??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_18_layer_call_and_return_conditional_losses_62468~lmnopqrstuvw??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
-__inference_sequential_18_layer_call_fn_58857zlmnopqrstuvwH?E
>?;
1?.
conv2d_25_input?????????
p 

 
? " ???????????
-__inference_sequential_18_layer_call_fn_59098zlmnopqrstuvwH?E
>?;
1?.
conv2d_25_input?????????
p

 
? " ???????????
-__inference_sequential_18_layer_call_fn_62347qlmnopqrstuvw??<
5?2
(?%
inputs?????????
p 

 
? " ???????????
-__inference_sequential_18_layer_call_fn_62376qlmnopqrstuvw??<
5?2
(?%
inputs?????????
p

 
? " ???????????
H__inference_sequential_19_layer_call_and_return_conditional_losses_59511?xyz{|}Q?N
G?D
:?7
conv2d_transpose_5_input?????????
p 

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_19_layer_call_and_return_conditional_losses_59530?xyz{|}Q?N
G?D
:?7
conv2d_transpose_5_input?????????
p

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_19_layer_call_and_return_conditional_losses_62540xxyz{|}??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_19_layer_call_and_return_conditional_losses_62578xxyz{|}??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????00
? ?
-__inference_sequential_19_layer_call_fn_59363}xyz{|}Q?N
G?D
:?7
conv2d_transpose_5_input?????????
p 

 
? " ??????????00?
-__inference_sequential_19_layer_call_fn_59492}xyz{|}Q?N
G?D
:?7
conv2d_transpose_5_input?????????
p

 
? " ??????????00?
-__inference_sequential_19_layer_call_fn_62485kxyz{|}??<
5?2
(?%
inputs?????????
p 

 
? " ??????????00?
-__inference_sequential_19_layer_call_fn_62502kxyz{|}??<
5?2
(?%
inputs?????????
p

 
? " ??????????00?
H__inference_sequential_20_layer_call_and_return_conditional_losses_60068?~??????????H?E
>?;
1?.
conv2d_27_input?????????00
p 

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_20_layer_call_and_return_conditional_losses_60102?~??????????H?E
>?;
1?.
conv2d_27_input?????????00
p

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_20_layer_call_and_return_conditional_losses_62682?~????????????<
5?2
(?%
inputs?????????00
p 

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_20_layer_call_and_return_conditional_losses_62728?~????????????<
5?2
(?%
inputs?????????00
p

 
? "-?*
#? 
0?????????00
? ?
-__inference_sequential_20_layer_call_fn_59793?~??????????H?E
>?;
1?.
conv2d_27_input?????????00
p 

 
? " ??????????00?
-__inference_sequential_20_layer_call_fn_60034?~??????????H?E
>?;
1?.
conv2d_27_input?????????00
p

 
? " ??????????00?
-__inference_sequential_20_layer_call_fn_62607{~????????????<
5?2
(?%
inputs?????????00
p 

 
? " ??????????00?
-__inference_sequential_20_layer_call_fn_62636{~????????????<
5?2
(?%
inputs?????????00
p

 
? " ??????????00?
H__inference_sequential_21_layer_call_and_return_conditional_losses_60388???????H?E
>?;
1?.
conv2d_29_input?????????00
p 

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_21_layer_call_and_return_conditional_losses_60407???????H?E
>?;
1?.
conv2d_29_input?????????00
p

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_21_layer_call_and_return_conditional_losses_62787~????????<
5?2
(?%
inputs?????????00
p 

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_21_layer_call_and_return_conditional_losses_62812~????????<
5?2
(?%
inputs?????????00
p

 
? "-?*
#? 
0?????????00
? ?
-__inference_sequential_21_layer_call_fn_60239z??????H?E
>?;
1?.
conv2d_29_input?????????00
p 

 
? " ??????????00?
-__inference_sequential_21_layer_call_fn_60369z??????H?E
>?;
1?.
conv2d_29_input?????????00
p

 
? " ??????????00?
-__inference_sequential_21_layer_call_fn_62745q????????<
5?2
(?%
inputs?????????00
p 

 
? " ??????????00?
-__inference_sequential_21_layer_call_fn_62762q????????<
5?2
(?%
inputs?????????00
p

 
? " ??????????00?
#__inference_signature_wrapper_61114?NTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~????????????????:?7
? 
0?-
+
args_0!?
args_0??????????"E?B
@
sequential_21/?,
sequential_21?????????00