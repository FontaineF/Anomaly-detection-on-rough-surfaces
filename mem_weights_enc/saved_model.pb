??*
??
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??%
x
latent/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namelatent/kernel
q
!latent/kernel/Read/ReadVariableOpReadVariableOplatent/kernel* 
_output_shapes
:
??*
dtype0
o
latent/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelatent/bias
h
latent/bias/Read/ReadVariableOpReadVariableOplatent/bias*
_output_shapes	
:?*
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:*
dtype0
?
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_18/gamma
?
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_18/beta
?
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_18/moving_mean
?
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_18/moving_variance
?
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:*
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
:*
dtype0
?
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_19/gamma
?
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_19/beta
?
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_19/moving_mean
?
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_19/moving_variance
?
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
:*
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:*
dtype0
?
batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_20/gamma
?
0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_20/beta
?
/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_20/moving_mean
?
6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_20/moving_variance
?
:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
:*
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
:*
dtype0
?
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_21/gamma
?
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_21/beta
?
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_21/moving_mean
?
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_21/moving_variance
?
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
: *
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
: *
dtype0
?
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_22/gamma
?
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_22/beta
?
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_22/moving_mean
?
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_22/moving_variance
?
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
: *
dtype0
?
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_23/gamma
?
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_23/beta
?
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_23/moving_mean
?
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_23/moving_variance
?
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_24/gamma
?
0batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_24/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_24/beta
?
/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_24/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_24/moving_mean
?
6batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_24/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_24/moving_variance
?
:batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_24/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_25/gamma
?
0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_25/beta
?
/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_25/moving_mean
?
6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_25/moving_variance
?
:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes
:@*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ĉ
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api

signatures

_init_input_shape
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
trainable_variables
 regularization_losses
!	keras_api
?
"layer_with_weights-0
"layer-0
#layer_with_weights-1
#layer-1
$layer-2
%layer_with_weights-2
%layer-3
&layer_with_weights-3
&layer-4
'layer-5
(	variables
)trainable_variables
*regularization_losses
+	keras_api
?
,layer_with_weights-0
,layer-0
-layer_with_weights-1
-layer-1
.layer-2
/layer_with_weights-2
/layer-3
0layer_with_weights-3
0layer-4
1layer-5
2	variables
3trainable_variables
4regularization_losses
5	keras_api
R
6	variables
7trainable_variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15
P16
Q17
R18
S19
T20
U21
V22
W23
X24
Y25
Z26
[27
\28
]29
^30
_31
`32
a33
b34
c35
d36
e37
f38
g39
h40
i41
j42
k43
l44
m45
n46
o47
:48
;49
?
@0
A1
B2
C3
F4
G5
H6
I7
L8
M9
N10
O11
R12
S13
T14
U15
X16
Y17
Z18
[19
^20
_21
`22
a23
d24
e25
f26
g27
j28
k29
l30
m31
:32
;33
 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
	trainable_variables

regularization_losses
 
 
h

@kernel
Abias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
?
yaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
z	variables
{trainable_variables
|regularization_losses
}	keras_api
T
~	variables
trainable_variables
?regularization_losses
?	keras_api
l

Fkernel
Gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
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
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
8
@0
A1
B2
C3
F4
G5
H6
I7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
l

Lkernel
Mbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
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

Rkernel
Sbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
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
L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9
V10
W11
8
L0
M1
N2
O3
R4
S5
T6
U7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
l

Xkernel
Ybias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	Zgamma
[beta
\moving_mean
]moving_variance
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

^kernel
_bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	`gamma
abeta
bmoving_mean
cmoving_variance
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
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11
8
X0
Y1
Z2
[3
^4
_5
`6
a7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
(	variables
)trainable_variables
*regularization_losses
l

dkernel
ebias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	fgamma
gbeta
hmoving_mean
imoving_variance
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

jkernel
kbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	lgamma
mbeta
nmoving_mean
omoving_variance
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
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
8
d0
e1
f2
g3
j4
k5
l6
m7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
YW
VARIABLE_VALUElatent/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElatent/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
LJ
VARIABLE_VALUEconv2d_15/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_15/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_18/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_18/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_18/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_18/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_16/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_16/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_19/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_19/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_19/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_19/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_17/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_17/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_20/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_20/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_20/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_20/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_18/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_18/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_21/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_21/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_21/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_21/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_19/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_19/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_22/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_22/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_22/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_22/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_20/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_20/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_23/gamma'variables/32/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_23/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_23/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_23/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_21/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_21/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_24/gamma'variables/38/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_24/beta'variables/39/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_24/moving_mean'variables/40/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_24/moving_variance'variables/41/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_22/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_22/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_25/gamma'variables/44/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_25/beta'variables/45/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_25/moving_mean'variables/46/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_25/moving_variance'variables/47/.ATTRIBUTES/VARIABLE_VALUE
v
D0
E1
J2
K3
P4
Q5
V6
W7
\8
]9
b10
c11
h12
i13
n14
o15
1
0
1
2
3
4
5
6
 
 
 

@0
A1

@0
A1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
 

B0
C1
D2
E3

B0
C1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses

F0
G1

F0
G1
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

H0
I1
J2
K3

H0
I1
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

D0
E1
J2
K3
*
0
1
2
3
4
5
 
 
 

L0
M1

L0
M1
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

N0
O1
P2
Q3

N0
O1
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
R0
S1

R0
S1
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

T0
U1
V2
W3
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

P0
Q1
V2
W3
*
0
1
2
3
4
5
 
 
 

X0
Y1

X0
Y1
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

Z0
[1
\2
]3
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
^0
_1

^0
_1
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

`0
a1
b2
c3
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

\0
]1
b2
c3
*
"0
#1
$2
%3
&4
'5
 
 
 

d0
e1

d0
e1
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

f0
g1
h2
i3
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

j0
k1
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

l0
m1
n2
o3

l0
m1
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

h0
i1
n2
o3
*
,0
-1
.2
/3
04
15
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

D0
E1
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
J0
K1
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
P0
Q1
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
V0
W1
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
\0
]1
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
b0
c1
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
h0
i1
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
n0
o1
 
 
 
 
 
 
 
 
 
?
serving_default_input_3Placeholder*/
_output_shapes
:?????????00*
dtype0*$
shape:?????????00
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_15/kernelconv2d_15/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_16/kernelconv2d_16/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_19/kernelconv2d_19/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_21/kernelconv2d_21/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_22/kernelconv2d_22/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_variancelatent/kernellatent/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_54052
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!latent/kernel/Read/ReadVariableOplatent/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp0batch_normalization_24/gamma/Read/ReadVariableOp/batch_normalization_24/beta/Read/ReadVariableOp6batch_normalization_24/moving_mean/Read/ReadVariableOp:batch_normalization_24/moving_variance/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOpConst*?
Tin8
624*
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
__inference__traced_save_56649
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelatent/kernellatent/biasconv2d_15/kernelconv2d_15/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_16/kernelconv2d_16/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_17/kernelconv2d_17/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_19/kernelconv2d_19/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_21/kernelconv2d_21/biasbatch_normalization_24/gammabatch_normalization_24/beta"batch_normalization_24/moving_mean&batch_normalization_24/moving_varianceconv2d_22/kernelconv2d_22/biasbatch_normalization_25/gammabatch_normalization_25/beta"batch_normalization_25/moving_mean&batch_normalization_25/moving_variance*>
Tin7
523*
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
!__inference__traced_restore_56809??#
?"
?
H__inference_sequential_12_layer_call_and_return_conditional_losses_51871
conv2d_17_input)
conv2d_17_51840:
conv2d_17_51842:*
batch_normalization_20_51845:*
batch_normalization_20_51847:*
batch_normalization_20_51849:*
batch_normalization_20_51851:)
conv2d_18_51855:
conv2d_18_51857:*
batch_normalization_21_51860:*
batch_normalization_21_51862:*
batch_normalization_21_51864:*
batch_normalization_21_51866:
identity??.batch_normalization_20/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputconv2d_17_51840conv2d_17_51842*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51478?
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_20_51845batch_normalization_20_51847batch_normalization_20_51849batch_normalization_20_51851*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51501?
leaky_re_lu_19/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_51516?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_19/PartitionedCall:output:0conv2d_18_51855conv2d_18_51857*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_51528?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_21_51860batch_normalization_21_51862batch_normalization_21_51864batch_normalization_21_51866*
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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51551?
leaky_re_lu_20/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_51566~
IdentityIdentity'leaky_re_lu_20/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_17_input
?	
?
6__inference_batch_normalization_18_layer_call_fn_55284

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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_50783?
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
?
?
6__inference_batch_normalization_23_layer_call_fn_56075

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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52123w
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51781

inputs)
conv2d_17_51750:
conv2d_17_51752:*
batch_normalization_20_51755:*
batch_normalization_20_51757:*
batch_normalization_20_51759:*
batch_normalization_20_51761:)
conv2d_18_51765:
conv2d_18_51767:*
batch_normalization_21_51770:*
batch_normalization_21_51772:*
batch_normalization_21_51774:*
batch_normalization_21_51776:
identity??.batch_normalization_20/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_17_51750conv2d_17_51752*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51478?
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_20_51755batch_normalization_20_51757batch_normalization_20_51759batch_normalization_20_51761*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51695?
leaky_re_lu_19/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_51516?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_19/PartitionedCall:output:0conv2d_18_51765conv2d_18_51767*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_51528?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_21_51770batch_normalization_21_51772batch_normalization_21_51774batch_normalization_21_51776*
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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51635?
leaky_re_lu_20/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_51566~
IdentityIdentity'leaky_re_lu_20/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_24_layer_call_fn_56228

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52645w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?8
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_55176

inputsB
(conv2d_21_conv2d_readvariableop_resource: @7
)conv2d_21_biasadd_readvariableop_resource:@<
.batch_normalization_24_readvariableop_resource:@>
0batch_normalization_24_readvariableop_1_resource:@M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_22_conv2d_readvariableop_resource:@@7
)conv2d_22_biasadd_readvariableop_resource:@<
.batch_normalization_25_readvariableop_resource:@>
0batch_normalization_25_readvariableop_1_resource:@M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource:@
identity??6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_24/ReadVariableOp?'batch_normalization_24/ReadVariableOp_1?6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_25/ReadVariableOp?'batch_normalization_25/ReadVariableOp_1? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_21/Conv2DConv2Dinputs'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_21/BiasAdd:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
leaky_re_lu_23/LeakyRelu	LeakyRelu+batch_normalization_24/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_22/Conv2DConv2D&leaky_re_lu_23/LeakyRelu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_22/BiasAdd:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
leaky_re_lu_24/LeakyRelu	LeakyRelu+batch_normalization_25/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>}
IdentityIdentity&leaky_re_lu_24/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp7^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_conv2d_20_layer_call_fn_56026

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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_52100w
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
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_23_layer_call_fn_56088

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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52207w
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

?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_55271

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51419

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
?
?
)__inference_conv2d_21_layer_call_fn_56179

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_52622w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
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
?
?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56313

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
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
:?????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56412

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56106

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
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56142

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
?!
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_50997

inputs)
conv2d_15_50907:
conv2d_15_50909:*
batch_normalization_18_50930:*
batch_normalization_18_50932:*
batch_normalization_18_50934:*
batch_normalization_18_50936:)
conv2d_16_50957:
conv2d_16_50959:*
batch_normalization_19_50980:*
batch_normalization_19_50982:*
batch_normalization_19_50984:*
batch_normalization_19_50986:
identity??.batch_normalization_18/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_50907conv2d_15_50909*
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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_50906?
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_18_50930batch_normalization_18_50932batch_normalization_18_50934batch_normalization_18_50936*
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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_50929?
leaky_re_lu_17/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_50944?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0conv2d_16_50957conv2d_16_50959*
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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_50956?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_19_50980batch_normalization_19_50982batch_normalization_19_50984batch_normalization_19_50986*
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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_50979?
leaky_re_lu_18/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_50994~
IdentityIdentity'leaky_re_lu_18/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_54052
input_3!
unknown:
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

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: @

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@$

unknown_41:@@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@

unknown_47:
??

unknown_48:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_50761p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????00
!
_user_specified_name	input_3
?	
?
6__inference_batch_normalization_22_layer_call_fn_55909

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
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_51958?
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
?
?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_51927

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
?
e
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_52138

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
?
e
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_50994

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
?
?
'__inference_model_3_layer_call_fn_53725
input_3!
unknown:
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

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: @

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@$

unknown_41:@@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@

unknown_47:
??

unknown_48:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_53517p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????00
!
_user_specified_name	input_3
?
e
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_56017

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_25_layer_call_fn_56394

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52779w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_13_layer_call_fn_54980

inputs!
unknown: 
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_52353w
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
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_22_layer_call_fn_55922

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
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_52073w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51478

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
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
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00w
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
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56295

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_17_layer_call_fn_55567

inputs!
unknown:
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
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51478w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
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
?!
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_52713

inputs)
conv2d_21_52623: @
conv2d_21_52625:@*
batch_normalization_24_52646:@*
batch_normalization_24_52648:@*
batch_normalization_24_52650:@*
batch_normalization_24_52652:@)
conv2d_22_52673:@@
conv2d_22_52675:@*
batch_normalization_25_52696:@*
batch_normalization_25_52698:@*
batch_normalization_25_52700:@*
batch_normalization_25_52702:@
identity??.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_21_52623conv2d_21_52625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_52622?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0batch_normalization_24_52646batch_normalization_24_52648batch_normalization_24_52650batch_normalization_24_52652*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52645?
leaky_re_lu_23/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_52660?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0conv2d_22_52673conv2d_22_52675*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_52672?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_25_52696batch_normalization_25_52698batch_normalization_25_52700batch_normalization_25_52702*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52695?
leaky_re_lu_24/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_52710~
IdentityIdentity'leaky_re_lu_24/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_51123

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
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_53162

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_20_layer_call_fn_55603

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
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51386?
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
?8
?
H__inference_sequential_12_layer_call_and_return_conditional_losses_54876

inputsB
(conv2d_17_conv2d_readvariableop_resource:7
)conv2d_17_biasadd_readvariableop_resource:<
.batch_normalization_20_readvariableop_resource:>
0batch_normalization_20_readvariableop_1_resource:M
?batch_normalization_20_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_18_conv2d_readvariableop_resource:7
)conv2d_18_biasadd_readvariableop_resource:<
.batch_normalization_21_readvariableop_resource:>
0batch_normalization_21_readvariableop_1_resource:M
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:
identity??6batch_normalization_20/FusedBatchNormV3/ReadVariableOp?8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_20/ReadVariableOp?'batch_normalization_20/ReadVariableOp_1?6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_21/ReadVariableOp?'batch_normalization_21/ReadVariableOp_1? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_17/BiasAdd:output:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_19/LeakyRelu	LeakyRelu+batch_normalization_20/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_18/Conv2DConv2D&leaky_re_lu_19/LeakyRelu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_18/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_20/LeakyRelu	LeakyRelu+batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>}
IdentityIdentity&leaky_re_lu_20/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp7^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55548

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
?
?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55647

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
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51695

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
7:?????????00:::::*
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
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55359

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
?
?
6__inference_batch_normalization_22_layer_call_fn_55935

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
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_52267w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_21_layer_call_fn_56012

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
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_52088h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_11_layer_call_fn_51024
conv2d_15_input!
unknown:
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_50997w
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
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_15_input
?"
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_52477
conv2d_19_input)
conv2d_19_52446: 
conv2d_19_52448: *
batch_normalization_22_52451: *
batch_normalization_22_52453: *
batch_normalization_22_52455: *
batch_normalization_22_52457: )
conv2d_20_52461:  
conv2d_20_52463: *
batch_normalization_23_52466: *
batch_normalization_23_52468: *
batch_normalization_23_52470: *
batch_normalization_23_52472: 
identity??.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallconv2d_19_inputconv2d_19_52446conv2d_19_52448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_52050?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0batch_normalization_22_52451batch_normalization_22_52453batch_normalization_22_52455batch_normalization_22_52457*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_52267?
leaky_re_lu_21/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_52088?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0conv2d_20_52461conv2d_20_52463*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_52100?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_23_52466batch_normalization_23_52468batch_normalization_23_52470batch_normalization_23_52472*
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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52207?
leaky_re_lu_22/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_52138~
IdentityIdentity'leaky_re_lu_22/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_19_input
?

?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_52100

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
r
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
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?8
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_55026

inputsB
(conv2d_19_conv2d_readvariableop_resource: 7
)conv2d_19_biasadd_readvariableop_resource: <
.batch_normalization_22_readvariableop_resource: >
0batch_normalization_22_readvariableop_1_resource: M
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_20_conv2d_readvariableop_resource:  7
)conv2d_20_biasadd_readvariableop_resource: <
.batch_normalization_23_readvariableop_resource: >
0batch_normalization_23_readvariableop_1_resource: M
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: 
identity??6batch_normalization_22/FusedBatchNormV3/ReadVariableOp?8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_22/ReadVariableOp?'batch_normalization_22/ReadVariableOp_1?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_19/BiasAdd:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
leaky_re_lu_21/LeakyRelu	LeakyRelu+batch_normalization_22/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_20/Conv2DConv2D&leaky_re_lu_21/LeakyRelu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_20/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
leaky_re_lu_22/LeakyRelu	LeakyRelu+batch_normalization_23/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>}
IdentityIdentity&leaky_re_lu_22/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp7^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_55227

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_53162a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_50979

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
?3
?
B__inference_model_3_layer_call_and_return_conditional_losses_53517

inputs-
sequential_11_53410:!
sequential_11_53412:!
sequential_11_53414:!
sequential_11_53416:!
sequential_11_53418:!
sequential_11_53420:-
sequential_11_53422:!
sequential_11_53424:!
sequential_11_53426:!
sequential_11_53428:!
sequential_11_53430:!
sequential_11_53432:-
sequential_12_53435:!
sequential_12_53437:!
sequential_12_53439:!
sequential_12_53441:!
sequential_12_53443:!
sequential_12_53445:-
sequential_12_53447:!
sequential_12_53449:!
sequential_12_53451:!
sequential_12_53453:!
sequential_12_53455:!
sequential_12_53457:-
sequential_13_53460: !
sequential_13_53462: !
sequential_13_53464: !
sequential_13_53466: !
sequential_13_53468: !
sequential_13_53470: -
sequential_13_53472:  !
sequential_13_53474: !
sequential_13_53476: !
sequential_13_53478: !
sequential_13_53480: !
sequential_13_53482: -
sequential_14_53485: @!
sequential_14_53487:@!
sequential_14_53489:@!
sequential_14_53491:@!
sequential_14_53493:@!
sequential_14_53495:@-
sequential_14_53497:@@!
sequential_14_53499:@!
sequential_14_53501:@!
sequential_14_53503:@!
sequential_14_53505:@!
sequential_14_53507:@ 
latent_53511:
??
latent_53513:	?
identity??latent/StatefulPartitionedCall?%sequential_11/StatefulPartitionedCall?%sequential_12/StatefulPartitionedCall?%sequential_13/StatefulPartitionedCall?%sequential_14/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinputssequential_11_53410sequential_11_53412sequential_11_53414sequential_11_53416sequential_11_53418sequential_11_53420sequential_11_53422sequential_11_53424sequential_11_53426sequential_11_53428sequential_11_53430sequential_11_53432*
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_51209?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCall.sequential_11/StatefulPartitionedCall:output:0sequential_12_53435sequential_12_53437sequential_12_53439sequential_12_53441sequential_12_53443sequential_12_53445sequential_12_53447sequential_12_53449sequential_12_53451sequential_12_53453sequential_12_53455sequential_12_53457*
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51781?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCall.sequential_12/StatefulPartitionedCall:output:0sequential_13_53460sequential_13_53462sequential_13_53464sequential_13_53466sequential_13_53468sequential_13_53470sequential_13_53472sequential_13_53474sequential_13_53476sequential_13_53478sequential_13_53480sequential_13_53482*
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_52353?
%sequential_14/StatefulPartitionedCallStatefulPartitionedCall.sequential_13/StatefulPartitionedCall:output:0sequential_14_53485sequential_14_53487sequential_14_53489sequential_14_53491sequential_14_53493sequential_14_53495sequential_14_53497sequential_14_53499sequential_14_53501sequential_14_53503sequential_14_53505sequential_14_53507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_52925?
flatten_1/PartitionedCallPartitionedCall.sequential_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_53162?
latent/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0latent_53511latent_53513*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_53174w
IdentityIdentity'latent/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^latent/StatefulPartitionedCall&^sequential_11/StatefulPartitionedCall&^sequential_12/StatefulPartitionedCall&^sequential_13/StatefulPartitionedCall&^sequential_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall2N
%sequential_14/StatefulPartitionedCall%sequential_14/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_19_layer_call_fn_55450

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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_50878?
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
?
A__inference_latent_layer_call_and_return_conditional_losses_53174

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_14_layer_call_fn_52981
conv2d_21_input!
unknown: @
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_52925w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
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
_user_specified_nameconv2d_21_input
?
?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55989

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
7:????????? : : : : :*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52645

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52695

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_19_layer_call_fn_55706

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
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_51516h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?!
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_52353

inputs)
conv2d_19_52322: 
conv2d_19_52324: *
batch_normalization_22_52327: *
batch_normalization_22_52329: *
batch_normalization_22_52331: *
batch_normalization_22_52333: )
conv2d_20_52337:  
conv2d_20_52339: *
batch_normalization_23_52342: *
batch_normalization_23_52344: *
batch_normalization_23_52346: *
batch_normalization_23_52348: 
identity??.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_19_52322conv2d_19_52324*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_52050?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0batch_normalization_22_52327batch_normalization_22_52329batch_normalization_22_52331batch_normalization_22_52333*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_52267?
leaky_re_lu_21/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_52088?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0conv2d_20_52337conv2d_20_52339*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_52100?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_23_52342batch_normalization_23_52344batch_normalization_23_52346batch_normalization_23_52348*
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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52207?
leaky_re_lu_22/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_52138~
IdentityIdentity'leaky_re_lu_22/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_19_layer_call_fn_55873

inputs!
unknown: 
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
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_52050w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
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
?
?
-__inference_sequential_12_layer_call_fn_54801

inputs!
unknown:
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51569w
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
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
A__inference_latent_layer_call_and_return_conditional_losses_55252

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_52925

inputs)
conv2d_21_52894: @
conv2d_21_52896:@*
batch_normalization_24_52899:@*
batch_normalization_24_52901:@*
batch_normalization_24_52903:@*
batch_normalization_24_52905:@)
conv2d_22_52909:@@
conv2d_22_52911:@*
batch_normalization_25_52914:@*
batch_normalization_25_52916:@*
batch_normalization_25_52918:@*
batch_normalization_25_52920:@
identity??.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_21_52894conv2d_21_52896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_52622?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0batch_normalization_24_52899batch_normalization_24_52901batch_normalization_24_52903batch_normalization_24_52905*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52839?
leaky_re_lu_23/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_52660?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0conv2d_22_52909conv2d_22_52911*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_52672?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_25_52914batch_normalization_25_52916batch_normalization_25_52918batch_normalization_25_52920*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52779?
leaky_re_lu_24/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_52710~
IdentityIdentity'leaky_re_lu_24/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56448

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_50956

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
6__inference_batch_normalization_25_layer_call_fn_56381

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52695w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55377

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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_56036

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
r
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
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_14_layer_call_fn_55130

inputs!
unknown: @
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@
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
:?????????@**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_52925w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
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
?

?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_55883

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
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
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_55730

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
r
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
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
)__inference_conv2d_15_layer_call_fn_55261

inputs!
unknown:
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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_50906w
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
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55836

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
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_51991

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
?
J
.__inference_leaky_re_lu_18_layer_call_fn_55553

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
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_50994h
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
??
?!
!__inference__traced_restore_56809
file_prefix2
assignvariableop_latent_kernel:
??-
assignvariableop_1_latent_bias:	?=
#assignvariableop_2_conv2d_15_kernel:/
!assignvariableop_3_conv2d_15_bias:=
/assignvariableop_4_batch_normalization_18_gamma:<
.assignvariableop_5_batch_normalization_18_beta:C
5assignvariableop_6_batch_normalization_18_moving_mean:G
9assignvariableop_7_batch_normalization_18_moving_variance:=
#assignvariableop_8_conv2d_16_kernel:/
!assignvariableop_9_conv2d_16_bias:>
0assignvariableop_10_batch_normalization_19_gamma:=
/assignvariableop_11_batch_normalization_19_beta:D
6assignvariableop_12_batch_normalization_19_moving_mean:H
:assignvariableop_13_batch_normalization_19_moving_variance:>
$assignvariableop_14_conv2d_17_kernel:0
"assignvariableop_15_conv2d_17_bias:>
0assignvariableop_16_batch_normalization_20_gamma:=
/assignvariableop_17_batch_normalization_20_beta:D
6assignvariableop_18_batch_normalization_20_moving_mean:H
:assignvariableop_19_batch_normalization_20_moving_variance:>
$assignvariableop_20_conv2d_18_kernel:0
"assignvariableop_21_conv2d_18_bias:>
0assignvariableop_22_batch_normalization_21_gamma:=
/assignvariableop_23_batch_normalization_21_beta:D
6assignvariableop_24_batch_normalization_21_moving_mean:H
:assignvariableop_25_batch_normalization_21_moving_variance:>
$assignvariableop_26_conv2d_19_kernel: 0
"assignvariableop_27_conv2d_19_bias: >
0assignvariableop_28_batch_normalization_22_gamma: =
/assignvariableop_29_batch_normalization_22_beta: D
6assignvariableop_30_batch_normalization_22_moving_mean: H
:assignvariableop_31_batch_normalization_22_moving_variance: >
$assignvariableop_32_conv2d_20_kernel:  0
"assignvariableop_33_conv2d_20_bias: >
0assignvariableop_34_batch_normalization_23_gamma: =
/assignvariableop_35_batch_normalization_23_beta: D
6assignvariableop_36_batch_normalization_23_moving_mean: H
:assignvariableop_37_batch_normalization_23_moving_variance: >
$assignvariableop_38_conv2d_21_kernel: @0
"assignvariableop_39_conv2d_21_bias:@>
0assignvariableop_40_batch_normalization_24_gamma:@=
/assignvariableop_41_batch_normalization_24_beta:@D
6assignvariableop_42_batch_normalization_24_moving_mean:@H
:assignvariableop_43_batch_normalization_24_moving_variance:@>
$assignvariableop_44_conv2d_22_kernel:@@0
"assignvariableop_45_conv2d_22_bias:@>
0assignvariableop_46_batch_normalization_25_gamma:@=
/assignvariableop_47_batch_normalization_25_beta:@D
6assignvariableop_48_batch_normalization_25_moving_mean:@H
:assignvariableop_49_batch_normalization_25_moving_variance:@
identity_51??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*?
value?B?3B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_latent_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_latent_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_15_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_15_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_18_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_18_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_18_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_18_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_16_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_16_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_19_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_19_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_19_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_19_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_17_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_17_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_20_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_20_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_20_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_20_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_18_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_18_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_21_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_21_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_21_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_21_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_19_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_19_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_22_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_22_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_22_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_22_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_20_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_20_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp0assignvariableop_34_batch_normalization_23_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batch_normalization_23_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp6assignvariableop_36_batch_normalization_23_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp:assignvariableop_37_batch_normalization_23_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv2d_21_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv2d_21_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp0assignvariableop_40_batch_normalization_24_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp/assignvariableop_41_batch_normalization_24_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_batch_normalization_24_moving_meanIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp:assignvariableop_43_batch_normalization_24_moving_varianceIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp$assignvariableop_44_conv2d_22_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp"assignvariableop_45_conv2d_22_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp0assignvariableop_46_batch_normalization_25_gammaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp/assignvariableop_47_batch_normalization_25_betaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp6assignvariableop_48_batch_normalization_25_moving_meanIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp:assignvariableop_49_batch_normalization_25_moving_varianceIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?	
Identity_50Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_51IdentityIdentity_50:output:0^NoOp_1*
T0*
_output_shapes
: ?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_51Identity_51:output:0*y
_input_shapesh
f: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51501

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
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55395

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
?
?
6__inference_batch_normalization_19_layer_call_fn_55476

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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_51063w
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
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52594

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
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
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_14_layer_call_fn_55101

inputs!
unknown: @
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@
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
:?????????@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_52713w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_50847

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
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51386

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
?
J
.__inference_leaky_re_lu_22_layer_call_fn_56165

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
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_52138h
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
?
?
6__inference_batch_normalization_24_layer_call_fn_56241

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52839w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_50783

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
?
-__inference_sequential_13_layer_call_fn_52409
conv2d_19_input!
unknown: 
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_52353w
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
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_19_input
?3
?
B__inference_model_3_layer_call_and_return_conditional_losses_53945
input_3-
sequential_11_53838:!
sequential_11_53840:!
sequential_11_53842:!
sequential_11_53844:!
sequential_11_53846:!
sequential_11_53848:-
sequential_11_53850:!
sequential_11_53852:!
sequential_11_53854:!
sequential_11_53856:!
sequential_11_53858:!
sequential_11_53860:-
sequential_12_53863:!
sequential_12_53865:!
sequential_12_53867:!
sequential_12_53869:!
sequential_12_53871:!
sequential_12_53873:-
sequential_12_53875:!
sequential_12_53877:!
sequential_12_53879:!
sequential_12_53881:!
sequential_12_53883:!
sequential_12_53885:-
sequential_13_53888: !
sequential_13_53890: !
sequential_13_53892: !
sequential_13_53894: !
sequential_13_53896: !
sequential_13_53898: -
sequential_13_53900:  !
sequential_13_53902: !
sequential_13_53904: !
sequential_13_53906: !
sequential_13_53908: !
sequential_13_53910: -
sequential_14_53913: @!
sequential_14_53915:@!
sequential_14_53917:@!
sequential_14_53919:@!
sequential_14_53921:@!
sequential_14_53923:@-
sequential_14_53925:@@!
sequential_14_53927:@!
sequential_14_53929:@!
sequential_14_53931:@!
sequential_14_53933:@!
sequential_14_53935:@ 
latent_53939:
??
latent_53941:	?
identity??latent/StatefulPartitionedCall?%sequential_11/StatefulPartitionedCall?%sequential_12/StatefulPartitionedCall?%sequential_13/StatefulPartitionedCall?%sequential_14/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinput_3sequential_11_53838sequential_11_53840sequential_11_53842sequential_11_53844sequential_11_53846sequential_11_53848sequential_11_53850sequential_11_53852sequential_11_53854sequential_11_53856sequential_11_53858sequential_11_53860*
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_51209?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCall.sequential_11/StatefulPartitionedCall:output:0sequential_12_53863sequential_12_53865sequential_12_53867sequential_12_53869sequential_12_53871sequential_12_53873sequential_12_53875sequential_12_53877sequential_12_53879sequential_12_53881sequential_12_53883sequential_12_53885*
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51781?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCall.sequential_12/StatefulPartitionedCall:output:0sequential_13_53888sequential_13_53890sequential_13_53892sequential_13_53894sequential_13_53896sequential_13_53898sequential_13_53900sequential_13_53902sequential_13_53904sequential_13_53906sequential_13_53908sequential_13_53910*
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_52353?
%sequential_14/StatefulPartitionedCallStatefulPartitionedCall.sequential_13/StatefulPartitionedCall:output:0sequential_14_53913sequential_14_53915sequential_14_53917sequential_14_53919sequential_14_53921sequential_14_53923sequential_14_53925sequential_14_53927sequential_14_53929sequential_14_53931sequential_14_53933sequential_14_53935*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_52925?
flatten_1/PartitionedCallPartitionedCall.sequential_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_53162?
latent/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0latent_53939latent_53941*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_53174w
IdentityIdentity'latent/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^latent/StatefulPartitionedCall&^sequential_11/StatefulPartitionedCall&^sequential_12/StatefulPartitionedCall&^sequential_13/StatefulPartitionedCall&^sequential_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall2N
%sequential_14/StatefulPartitionedCall%sequential_14/StatefulPartitionedCall:X T
/
_output_shapes
:?????????00
!
_user_specified_name	input_3
?
?
6__inference_batch_normalization_19_layer_call_fn_55463

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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_50979w
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
6__inference_batch_normalization_21_layer_call_fn_55743

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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51419?
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
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52779

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
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
:?????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_52710

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_21_layer_call_fn_55782

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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51635w
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
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_51566

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

?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_55577

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
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
:?????????00g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????00w
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
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52022

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
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52207

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
6__inference_batch_normalization_22_layer_call_fn_55896

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
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_51927?
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
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52530

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
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
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?"
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_51299
conv2d_15_input)
conv2d_15_51268:
conv2d_15_51270:*
batch_normalization_18_51273:*
batch_normalization_18_51275:*
batch_normalization_18_51277:*
batch_normalization_18_51279:)
conv2d_16_51283:
conv2d_16_51285:*
batch_normalization_19_51288:*
batch_normalization_19_51290:*
batch_normalization_19_51292:*
batch_normalization_19_51294:
identity??.batch_normalization_18/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputconv2d_15_51268conv2d_15_51270*
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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_50906?
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_18_51273batch_normalization_18_51275batch_normalization_18_51277batch_normalization_18_51279*
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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_50929?
leaky_re_lu_17/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_50944?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0conv2d_16_51283conv2d_16_51285*
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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_50956?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_19_51288batch_normalization_19_51290batch_normalization_19_51292batch_normalization_19_51294*
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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_50979?
leaky_re_lu_18/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_50994~
IdentityIdentity'leaky_re_lu_18/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_15_input
?
?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55665

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
6__inference_batch_normalization_20_layer_call_fn_55616

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
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51501w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51635

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
??
?=
B__inference_model_3_layer_call_and_return_conditional_losses_54622

inputsP
6sequential_11_conv2d_15_conv2d_readvariableop_resource:E
7sequential_11_conv2d_15_biasadd_readvariableop_resource:J
<sequential_11_batch_normalization_18_readvariableop_resource:L
>sequential_11_batch_normalization_18_readvariableop_1_resource:[
Msequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:]
Osequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_11_conv2d_16_conv2d_readvariableop_resource:E
7sequential_11_conv2d_16_biasadd_readvariableop_resource:J
<sequential_11_batch_normalization_19_readvariableop_resource:L
>sequential_11_batch_normalization_19_readvariableop_1_resource:[
Msequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:]
Osequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_12_conv2d_17_conv2d_readvariableop_resource:E
7sequential_12_conv2d_17_biasadd_readvariableop_resource:J
<sequential_12_batch_normalization_20_readvariableop_resource:L
>sequential_12_batch_normalization_20_readvariableop_1_resource:[
Msequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_resource:]
Osequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_12_conv2d_18_conv2d_readvariableop_resource:E
7sequential_12_conv2d_18_biasadd_readvariableop_resource:J
<sequential_12_batch_normalization_21_readvariableop_resource:L
>sequential_12_batch_normalization_21_readvariableop_1_resource:[
Msequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:]
Osequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_13_conv2d_19_conv2d_readvariableop_resource: E
7sequential_13_conv2d_19_biasadd_readvariableop_resource: J
<sequential_13_batch_normalization_22_readvariableop_resource: L
>sequential_13_batch_normalization_22_readvariableop_1_resource: [
Msequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_resource: ]
Osequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource: P
6sequential_13_conv2d_20_conv2d_readvariableop_resource:  E
7sequential_13_conv2d_20_biasadd_readvariableop_resource: J
<sequential_13_batch_normalization_23_readvariableop_resource: L
>sequential_13_batch_normalization_23_readvariableop_1_resource: [
Msequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_resource: ]
Osequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: P
6sequential_14_conv2d_21_conv2d_readvariableop_resource: @E
7sequential_14_conv2d_21_biasadd_readvariableop_resource:@J
<sequential_14_batch_normalization_24_readvariableop_resource:@L
>sequential_14_batch_normalization_24_readvariableop_1_resource:@[
Msequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:@]
Osequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:@P
6sequential_14_conv2d_22_conv2d_readvariableop_resource:@@E
7sequential_14_conv2d_22_biasadd_readvariableop_resource:@J
<sequential_14_batch_normalization_25_readvariableop_resource:@L
>sequential_14_batch_normalization_25_readvariableop_1_resource:@[
Msequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_resource:@]
Osequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource:@9
%latent_matmul_readvariableop_resource:
??5
&latent_biasadd_readvariableop_resource:	?
identity??latent/BiasAdd/ReadVariableOp?latent/MatMul/ReadVariableOp?3sequential_11/batch_normalization_18/AssignNewValue?5sequential_11/batch_normalization_18/AssignNewValue_1?Dsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp?Fsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1?3sequential_11/batch_normalization_18/ReadVariableOp?5sequential_11/batch_normalization_18/ReadVariableOp_1?3sequential_11/batch_normalization_19/AssignNewValue?5sequential_11/batch_normalization_19/AssignNewValue_1?Dsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp?Fsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?3sequential_11/batch_normalization_19/ReadVariableOp?5sequential_11/batch_normalization_19/ReadVariableOp_1?.sequential_11/conv2d_15/BiasAdd/ReadVariableOp?-sequential_11/conv2d_15/Conv2D/ReadVariableOp?.sequential_11/conv2d_16/BiasAdd/ReadVariableOp?-sequential_11/conv2d_16/Conv2D/ReadVariableOp?3sequential_12/batch_normalization_20/AssignNewValue?5sequential_12/batch_normalization_20/AssignNewValue_1?Dsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp?Fsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?3sequential_12/batch_normalization_20/ReadVariableOp?5sequential_12/batch_normalization_20/ReadVariableOp_1?3sequential_12/batch_normalization_21/AssignNewValue?5sequential_12/batch_normalization_21/AssignNewValue_1?Dsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp?Fsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?3sequential_12/batch_normalization_21/ReadVariableOp?5sequential_12/batch_normalization_21/ReadVariableOp_1?.sequential_12/conv2d_17/BiasAdd/ReadVariableOp?-sequential_12/conv2d_17/Conv2D/ReadVariableOp?.sequential_12/conv2d_18/BiasAdd/ReadVariableOp?-sequential_12/conv2d_18/Conv2D/ReadVariableOp?3sequential_13/batch_normalization_22/AssignNewValue?5sequential_13/batch_normalization_22/AssignNewValue_1?Dsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_22/ReadVariableOp?5sequential_13/batch_normalization_22/ReadVariableOp_1?3sequential_13/batch_normalization_23/AssignNewValue?5sequential_13/batch_normalization_23/AssignNewValue_1?Dsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_23/ReadVariableOp?5sequential_13/batch_normalization_23/ReadVariableOp_1?.sequential_13/conv2d_19/BiasAdd/ReadVariableOp?-sequential_13/conv2d_19/Conv2D/ReadVariableOp?.sequential_13/conv2d_20/BiasAdd/ReadVariableOp?-sequential_13/conv2d_20/Conv2D/ReadVariableOp?3sequential_14/batch_normalization_24/AssignNewValue?5sequential_14/batch_normalization_24/AssignNewValue_1?Dsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?Fsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?3sequential_14/batch_normalization_24/ReadVariableOp?5sequential_14/batch_normalization_24/ReadVariableOp_1?3sequential_14/batch_normalization_25/AssignNewValue?5sequential_14/batch_normalization_25/AssignNewValue_1?Dsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?Fsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?3sequential_14/batch_normalization_25/ReadVariableOp?5sequential_14/batch_normalization_25/ReadVariableOp_1?.sequential_14/conv2d_21/BiasAdd/ReadVariableOp?-sequential_14/conv2d_21/Conv2D/ReadVariableOp?.sequential_14/conv2d_22/BiasAdd/ReadVariableOp?-sequential_14/conv2d_22/Conv2D/ReadVariableOp?
-sequential_11/conv2d_15/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_11/conv2d_15/Conv2DConv2Dinputs5sequential_11/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_11/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_11/conv2d_15/BiasAddBiasAdd'sequential_11/conv2d_15/Conv2D:output:06sequential_11/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_11/batch_normalization_18/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_18_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_11/batch_normalization_18/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_18_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_11/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3(sequential_11/conv2d_15/BiasAdd:output:0;sequential_11/batch_normalization_18/ReadVariableOp:value:0=sequential_11/batch_normalization_18/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_11/batch_normalization_18/AssignNewValueAssignVariableOpMsequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_resourceBsequential_11/batch_normalization_18/FusedBatchNormV3:batch_mean:0E^sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_11/batch_normalization_18/AssignNewValue_1AssignVariableOpOsequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resourceFsequential_11/batch_normalization_18/FusedBatchNormV3:batch_variance:0G^sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_11/leaky_re_lu_17/LeakyRelu	LeakyRelu9sequential_11/batch_normalization_18/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_11/conv2d_16/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_11/conv2d_16/Conv2DConv2D4sequential_11/leaky_re_lu_17/LeakyRelu:activations:05sequential_11/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_11/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_11/conv2d_16/BiasAddBiasAdd'sequential_11/conv2d_16/Conv2D:output:06sequential_11/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_11/batch_normalization_19/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_11/batch_normalization_19/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_11/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3(sequential_11/conv2d_16/BiasAdd:output:0;sequential_11/batch_normalization_19/ReadVariableOp:value:0=sequential_11/batch_normalization_19/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_11/batch_normalization_19/AssignNewValueAssignVariableOpMsequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_resourceBsequential_11/batch_normalization_19/FusedBatchNormV3:batch_mean:0E^sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_11/batch_normalization_19/AssignNewValue_1AssignVariableOpOsequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resourceFsequential_11/batch_normalization_19/FusedBatchNormV3:batch_variance:0G^sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_11/leaky_re_lu_18/LeakyRelu	LeakyRelu9sequential_11/batch_normalization_19/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_12/conv2d_17/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_12/conv2d_17/Conv2DConv2D4sequential_11/leaky_re_lu_18/LeakyRelu:activations:05sequential_12/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_12/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_12/conv2d_17/BiasAddBiasAdd'sequential_12/conv2d_17/Conv2D:output:06sequential_12/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_12/batch_normalization_20/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_20_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_12/batch_normalization_20/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_20_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_12/batch_normalization_20/FusedBatchNormV3FusedBatchNormV3(sequential_12/conv2d_17/BiasAdd:output:0;sequential_12/batch_normalization_20/ReadVariableOp:value:0=sequential_12/batch_normalization_20/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_12/batch_normalization_20/AssignNewValueAssignVariableOpMsequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_resourceBsequential_12/batch_normalization_20/FusedBatchNormV3:batch_mean:0E^sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_12/batch_normalization_20/AssignNewValue_1AssignVariableOpOsequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resourceFsequential_12/batch_normalization_20/FusedBatchNormV3:batch_variance:0G^sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_12/leaky_re_lu_19/LeakyRelu	LeakyRelu9sequential_12/batch_normalization_20/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_12/conv2d_18/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_12/conv2d_18/Conv2DConv2D4sequential_12/leaky_re_lu_19/LeakyRelu:activations:05sequential_12/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
.sequential_12/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_12/conv2d_18/BiasAddBiasAdd'sequential_12/conv2d_18/Conv2D:output:06sequential_12/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
3sequential_12/batch_normalization_21/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_21_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_12/batch_normalization_21/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_21_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_12/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3(sequential_12/conv2d_18/BiasAdd:output:0;sequential_12/batch_normalization_21/ReadVariableOp:value:0=sequential_12/batch_normalization_21/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_12/batch_normalization_21/AssignNewValueAssignVariableOpMsequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_resourceBsequential_12/batch_normalization_21/FusedBatchNormV3:batch_mean:0E^sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_12/batch_normalization_21/AssignNewValue_1AssignVariableOpOsequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resourceFsequential_12/batch_normalization_21/FusedBatchNormV3:batch_variance:0G^sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_12/leaky_re_lu_20/LeakyRelu	LeakyRelu9sequential_12/batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
-sequential_13/conv2d_19/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_13/conv2d_19/Conv2DConv2D4sequential_12/leaky_re_lu_20/LeakyRelu:activations:05sequential_13/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_13/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_13/conv2d_19/BiasAddBiasAdd'sequential_13/conv2d_19/Conv2D:output:06sequential_13/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_13/batch_normalization_22/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_22_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_13/batch_normalization_22/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_22_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_13/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3(sequential_13/conv2d_19/BiasAdd:output:0;sequential_13/batch_normalization_22/ReadVariableOp:value:0=sequential_13/batch_normalization_22/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_13/batch_normalization_22/AssignNewValueAssignVariableOpMsequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_resourceBsequential_13/batch_normalization_22/FusedBatchNormV3:batch_mean:0E^sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_13/batch_normalization_22/AssignNewValue_1AssignVariableOpOsequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resourceFsequential_13/batch_normalization_22/FusedBatchNormV3:batch_variance:0G^sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_13/leaky_re_lu_21/LeakyRelu	LeakyRelu9sequential_13/batch_normalization_22/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
-sequential_13/conv2d_20/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_13/conv2d_20/Conv2DConv2D4sequential_13/leaky_re_lu_21/LeakyRelu:activations:05sequential_13/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_13/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_13/conv2d_20/BiasAddBiasAdd'sequential_13/conv2d_20/Conv2D:output:06sequential_13/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_13/batch_normalization_23/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_13/batch_normalization_23/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_13/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3(sequential_13/conv2d_20/BiasAdd:output:0;sequential_13/batch_normalization_23/ReadVariableOp:value:0=sequential_13/batch_normalization_23/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_13/batch_normalization_23/AssignNewValueAssignVariableOpMsequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceBsequential_13/batch_normalization_23/FusedBatchNormV3:batch_mean:0E^sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_13/batch_normalization_23/AssignNewValue_1AssignVariableOpOsequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resourceFsequential_13/batch_normalization_23/FusedBatchNormV3:batch_variance:0G^sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_13/leaky_re_lu_22/LeakyRelu	LeakyRelu9sequential_13/batch_normalization_23/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
-sequential_14/conv2d_21/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_14/conv2d_21/Conv2DConv2D4sequential_13/leaky_re_lu_22/LeakyRelu:activations:05sequential_14/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_14/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_14/conv2d_21/BiasAddBiasAdd'sequential_14/conv2d_21/Conv2D:output:06sequential_14/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
3sequential_14/batch_normalization_24/ReadVariableOpReadVariableOp<sequential_14_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype0?
5sequential_14/batch_normalization_24/ReadVariableOp_1ReadVariableOp>sequential_14_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Dsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Fsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5sequential_14/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3(sequential_14/conv2d_21/BiasAdd:output:0;sequential_14/batch_normalization_24/ReadVariableOp:value:0=sequential_14/batch_normalization_24/ReadVariableOp_1:value:0Lsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_14/batch_normalization_24/AssignNewValueAssignVariableOpMsequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_resourceBsequential_14/batch_normalization_24/FusedBatchNormV3:batch_mean:0E^sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_14/batch_normalization_24/AssignNewValue_1AssignVariableOpOsequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resourceFsequential_14/batch_normalization_24/FusedBatchNormV3:batch_variance:0G^sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_14/leaky_re_lu_23/LeakyRelu	LeakyRelu9sequential_14/batch_normalization_24/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>?
-sequential_14/conv2d_22/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
sequential_14/conv2d_22/Conv2DConv2D4sequential_14/leaky_re_lu_23/LeakyRelu:activations:05sequential_14/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_14/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_14/conv2d_22/BiasAddBiasAdd'sequential_14/conv2d_22/Conv2D:output:06sequential_14/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
3sequential_14/batch_normalization_25/ReadVariableOpReadVariableOp<sequential_14_batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype0?
5sequential_14/batch_normalization_25/ReadVariableOp_1ReadVariableOp>sequential_14_batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Dsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Fsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5sequential_14/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3(sequential_14/conv2d_22/BiasAdd:output:0;sequential_14/batch_normalization_25/ReadVariableOp:value:0=sequential_14/batch_normalization_25/ReadVariableOp_1:value:0Lsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_14/batch_normalization_25/AssignNewValueAssignVariableOpMsequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_resourceBsequential_14/batch_normalization_25/FusedBatchNormV3:batch_mean:0E^sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_14/batch_normalization_25/AssignNewValue_1AssignVariableOpOsequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resourceFsequential_14/batch_normalization_25/FusedBatchNormV3:batch_variance:0G^sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
&sequential_14/leaky_re_lu_24/LeakyRelu	LeakyRelu9sequential_14/batch_normalization_25/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
flatten_1/ReshapeReshape4sequential_14/leaky_re_lu_24/LeakyRelu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
latent/MatMul/ReadVariableOpReadVariableOp%latent_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
latent/MatMulMatMulflatten_1/Reshape:output:0$latent/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
latent/BiasAdd/ReadVariableOpReadVariableOp&latent_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
latent/BiasAddBiasAddlatent/MatMul:product:0%latent/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????g
IdentityIdentitylatent/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^latent/BiasAdd/ReadVariableOp^latent/MatMul/ReadVariableOp4^sequential_11/batch_normalization_18/AssignNewValue6^sequential_11/batch_normalization_18/AssignNewValue_1E^sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_18/ReadVariableOp6^sequential_11/batch_normalization_18/ReadVariableOp_14^sequential_11/batch_normalization_19/AssignNewValue6^sequential_11/batch_normalization_19/AssignNewValue_1E^sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_19/ReadVariableOp6^sequential_11/batch_normalization_19/ReadVariableOp_1/^sequential_11/conv2d_15/BiasAdd/ReadVariableOp.^sequential_11/conv2d_15/Conv2D/ReadVariableOp/^sequential_11/conv2d_16/BiasAdd/ReadVariableOp.^sequential_11/conv2d_16/Conv2D/ReadVariableOp4^sequential_12/batch_normalization_20/AssignNewValue6^sequential_12/batch_normalization_20/AssignNewValue_1E^sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_20/ReadVariableOp6^sequential_12/batch_normalization_20/ReadVariableOp_14^sequential_12/batch_normalization_21/AssignNewValue6^sequential_12/batch_normalization_21/AssignNewValue_1E^sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_21/ReadVariableOp6^sequential_12/batch_normalization_21/ReadVariableOp_1/^sequential_12/conv2d_17/BiasAdd/ReadVariableOp.^sequential_12/conv2d_17/Conv2D/ReadVariableOp/^sequential_12/conv2d_18/BiasAdd/ReadVariableOp.^sequential_12/conv2d_18/Conv2D/ReadVariableOp4^sequential_13/batch_normalization_22/AssignNewValue6^sequential_13/batch_normalization_22/AssignNewValue_1E^sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_22/ReadVariableOp6^sequential_13/batch_normalization_22/ReadVariableOp_14^sequential_13/batch_normalization_23/AssignNewValue6^sequential_13/batch_normalization_23/AssignNewValue_1E^sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_23/ReadVariableOp6^sequential_13/batch_normalization_23/ReadVariableOp_1/^sequential_13/conv2d_19/BiasAdd/ReadVariableOp.^sequential_13/conv2d_19/Conv2D/ReadVariableOp/^sequential_13/conv2d_20/BiasAdd/ReadVariableOp.^sequential_13/conv2d_20/Conv2D/ReadVariableOp4^sequential_14/batch_normalization_24/AssignNewValue6^sequential_14/batch_normalization_24/AssignNewValue_1E^sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpG^sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_14^sequential_14/batch_normalization_24/ReadVariableOp6^sequential_14/batch_normalization_24/ReadVariableOp_14^sequential_14/batch_normalization_25/AssignNewValue6^sequential_14/batch_normalization_25/AssignNewValue_1E^sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpG^sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_14^sequential_14/batch_normalization_25/ReadVariableOp6^sequential_14/batch_normalization_25/ReadVariableOp_1/^sequential_14/conv2d_21/BiasAdd/ReadVariableOp.^sequential_14/conv2d_21/Conv2D/ReadVariableOp/^sequential_14/conv2d_22/BiasAdd/ReadVariableOp.^sequential_14/conv2d_22/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
latent/BiasAdd/ReadVariableOplatent/BiasAdd/ReadVariableOp2<
latent/MatMul/ReadVariableOplatent/MatMul/ReadVariableOp2j
3sequential_11/batch_normalization_18/AssignNewValue3sequential_11/batch_normalization_18/AssignNewValue2n
5sequential_11/batch_normalization_18/AssignNewValue_15sequential_11/batch_normalization_18/AssignNewValue_12?
Dsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2?
Fsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_18/ReadVariableOp3sequential_11/batch_normalization_18/ReadVariableOp2n
5sequential_11/batch_normalization_18/ReadVariableOp_15sequential_11/batch_normalization_18/ReadVariableOp_12j
3sequential_11/batch_normalization_19/AssignNewValue3sequential_11/batch_normalization_19/AssignNewValue2n
5sequential_11/batch_normalization_19/AssignNewValue_15sequential_11/batch_normalization_19/AssignNewValue_12?
Dsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2?
Fsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_19/ReadVariableOp3sequential_11/batch_normalization_19/ReadVariableOp2n
5sequential_11/batch_normalization_19/ReadVariableOp_15sequential_11/batch_normalization_19/ReadVariableOp_12`
.sequential_11/conv2d_15/BiasAdd/ReadVariableOp.sequential_11/conv2d_15/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_15/Conv2D/ReadVariableOp-sequential_11/conv2d_15/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_16/BiasAdd/ReadVariableOp.sequential_11/conv2d_16/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_16/Conv2D/ReadVariableOp-sequential_11/conv2d_16/Conv2D/ReadVariableOp2j
3sequential_12/batch_normalization_20/AssignNewValue3sequential_12/batch_normalization_20/AssignNewValue2n
5sequential_12/batch_normalization_20/AssignNewValue_15sequential_12/batch_normalization_20/AssignNewValue_12?
Dsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp2?
Fsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_20/ReadVariableOp3sequential_12/batch_normalization_20/ReadVariableOp2n
5sequential_12/batch_normalization_20/ReadVariableOp_15sequential_12/batch_normalization_20/ReadVariableOp_12j
3sequential_12/batch_normalization_21/AssignNewValue3sequential_12/batch_normalization_21/AssignNewValue2n
5sequential_12/batch_normalization_21/AssignNewValue_15sequential_12/batch_normalization_21/AssignNewValue_12?
Dsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2?
Fsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_21/ReadVariableOp3sequential_12/batch_normalization_21/ReadVariableOp2n
5sequential_12/batch_normalization_21/ReadVariableOp_15sequential_12/batch_normalization_21/ReadVariableOp_12`
.sequential_12/conv2d_17/BiasAdd/ReadVariableOp.sequential_12/conv2d_17/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_17/Conv2D/ReadVariableOp-sequential_12/conv2d_17/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_18/BiasAdd/ReadVariableOp.sequential_12/conv2d_18/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_18/Conv2D/ReadVariableOp-sequential_12/conv2d_18/Conv2D/ReadVariableOp2j
3sequential_13/batch_normalization_22/AssignNewValue3sequential_13/batch_normalization_22/AssignNewValue2n
5sequential_13/batch_normalization_22/AssignNewValue_15sequential_13/batch_normalization_22/AssignNewValue_12?
Dsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_22/ReadVariableOp3sequential_13/batch_normalization_22/ReadVariableOp2n
5sequential_13/batch_normalization_22/ReadVariableOp_15sequential_13/batch_normalization_22/ReadVariableOp_12j
3sequential_13/batch_normalization_23/AssignNewValue3sequential_13/batch_normalization_23/AssignNewValue2n
5sequential_13/batch_normalization_23/AssignNewValue_15sequential_13/batch_normalization_23/AssignNewValue_12?
Dsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_23/ReadVariableOp3sequential_13/batch_normalization_23/ReadVariableOp2n
5sequential_13/batch_normalization_23/ReadVariableOp_15sequential_13/batch_normalization_23/ReadVariableOp_12`
.sequential_13/conv2d_19/BiasAdd/ReadVariableOp.sequential_13/conv2d_19/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_19/Conv2D/ReadVariableOp-sequential_13/conv2d_19/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_20/BiasAdd/ReadVariableOp.sequential_13/conv2d_20/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_20/Conv2D/ReadVariableOp-sequential_13/conv2d_20/Conv2D/ReadVariableOp2j
3sequential_14/batch_normalization_24/AssignNewValue3sequential_14/batch_normalization_24/AssignNewValue2n
5sequential_14/batch_normalization_24/AssignNewValue_15sequential_14/batch_normalization_24/AssignNewValue_12?
Dsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpDsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
Fsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Fsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12j
3sequential_14/batch_normalization_24/ReadVariableOp3sequential_14/batch_normalization_24/ReadVariableOp2n
5sequential_14/batch_normalization_24/ReadVariableOp_15sequential_14/batch_normalization_24/ReadVariableOp_12j
3sequential_14/batch_normalization_25/AssignNewValue3sequential_14/batch_normalization_25/AssignNewValue2n
5sequential_14/batch_normalization_25/AssignNewValue_15sequential_14/batch_normalization_25/AssignNewValue_12?
Dsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpDsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
Fsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Fsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12j
3sequential_14/batch_normalization_25/ReadVariableOp3sequential_14/batch_normalization_25/ReadVariableOp2n
5sequential_14/batch_normalization_25/ReadVariableOp_15sequential_14/batch_normalization_25/ReadVariableOp_12`
.sequential_14/conv2d_21/BiasAdd/ReadVariableOp.sequential_14/conv2d_21/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_21/Conv2D/ReadVariableOp-sequential_14/conv2d_21/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_22/BiasAdd/ReadVariableOp.sequential_14/conv2d_22/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_22/Conv2D/ReadVariableOp-sequential_14/conv2d_22/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
Ò
?7
B__inference_model_3_layer_call_and_return_conditional_losses_54442

inputsP
6sequential_11_conv2d_15_conv2d_readvariableop_resource:E
7sequential_11_conv2d_15_biasadd_readvariableop_resource:J
<sequential_11_batch_normalization_18_readvariableop_resource:L
>sequential_11_batch_normalization_18_readvariableop_1_resource:[
Msequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:]
Osequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_11_conv2d_16_conv2d_readvariableop_resource:E
7sequential_11_conv2d_16_biasadd_readvariableop_resource:J
<sequential_11_batch_normalization_19_readvariableop_resource:L
>sequential_11_batch_normalization_19_readvariableop_1_resource:[
Msequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:]
Osequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_12_conv2d_17_conv2d_readvariableop_resource:E
7sequential_12_conv2d_17_biasadd_readvariableop_resource:J
<sequential_12_batch_normalization_20_readvariableop_resource:L
>sequential_12_batch_normalization_20_readvariableop_1_resource:[
Msequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_resource:]
Osequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_12_conv2d_18_conv2d_readvariableop_resource:E
7sequential_12_conv2d_18_biasadd_readvariableop_resource:J
<sequential_12_batch_normalization_21_readvariableop_resource:L
>sequential_12_batch_normalization_21_readvariableop_1_resource:[
Msequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:]
Osequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_13_conv2d_19_conv2d_readvariableop_resource: E
7sequential_13_conv2d_19_biasadd_readvariableop_resource: J
<sequential_13_batch_normalization_22_readvariableop_resource: L
>sequential_13_batch_normalization_22_readvariableop_1_resource: [
Msequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_resource: ]
Osequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource: P
6sequential_13_conv2d_20_conv2d_readvariableop_resource:  E
7sequential_13_conv2d_20_biasadd_readvariableop_resource: J
<sequential_13_batch_normalization_23_readvariableop_resource: L
>sequential_13_batch_normalization_23_readvariableop_1_resource: [
Msequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_resource: ]
Osequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: P
6sequential_14_conv2d_21_conv2d_readvariableop_resource: @E
7sequential_14_conv2d_21_biasadd_readvariableop_resource:@J
<sequential_14_batch_normalization_24_readvariableop_resource:@L
>sequential_14_batch_normalization_24_readvariableop_1_resource:@[
Msequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:@]
Osequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:@P
6sequential_14_conv2d_22_conv2d_readvariableop_resource:@@E
7sequential_14_conv2d_22_biasadd_readvariableop_resource:@J
<sequential_14_batch_normalization_25_readvariableop_resource:@L
>sequential_14_batch_normalization_25_readvariableop_1_resource:@[
Msequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_resource:@]
Osequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource:@9
%latent_matmul_readvariableop_resource:
??5
&latent_biasadd_readvariableop_resource:	?
identity??latent/BiasAdd/ReadVariableOp?latent/MatMul/ReadVariableOp?Dsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp?Fsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1?3sequential_11/batch_normalization_18/ReadVariableOp?5sequential_11/batch_normalization_18/ReadVariableOp_1?Dsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp?Fsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?3sequential_11/batch_normalization_19/ReadVariableOp?5sequential_11/batch_normalization_19/ReadVariableOp_1?.sequential_11/conv2d_15/BiasAdd/ReadVariableOp?-sequential_11/conv2d_15/Conv2D/ReadVariableOp?.sequential_11/conv2d_16/BiasAdd/ReadVariableOp?-sequential_11/conv2d_16/Conv2D/ReadVariableOp?Dsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp?Fsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?3sequential_12/batch_normalization_20/ReadVariableOp?5sequential_12/batch_normalization_20/ReadVariableOp_1?Dsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp?Fsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?3sequential_12/batch_normalization_21/ReadVariableOp?5sequential_12/batch_normalization_21/ReadVariableOp_1?.sequential_12/conv2d_17/BiasAdd/ReadVariableOp?-sequential_12/conv2d_17/Conv2D/ReadVariableOp?.sequential_12/conv2d_18/BiasAdd/ReadVariableOp?-sequential_12/conv2d_18/Conv2D/ReadVariableOp?Dsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_22/ReadVariableOp?5sequential_13/batch_normalization_22/ReadVariableOp_1?Dsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_23/ReadVariableOp?5sequential_13/batch_normalization_23/ReadVariableOp_1?.sequential_13/conv2d_19/BiasAdd/ReadVariableOp?-sequential_13/conv2d_19/Conv2D/ReadVariableOp?.sequential_13/conv2d_20/BiasAdd/ReadVariableOp?-sequential_13/conv2d_20/Conv2D/ReadVariableOp?Dsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?Fsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?3sequential_14/batch_normalization_24/ReadVariableOp?5sequential_14/batch_normalization_24/ReadVariableOp_1?Dsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?Fsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?3sequential_14/batch_normalization_25/ReadVariableOp?5sequential_14/batch_normalization_25/ReadVariableOp_1?.sequential_14/conv2d_21/BiasAdd/ReadVariableOp?-sequential_14/conv2d_21/Conv2D/ReadVariableOp?.sequential_14/conv2d_22/BiasAdd/ReadVariableOp?-sequential_14/conv2d_22/Conv2D/ReadVariableOp?
-sequential_11/conv2d_15/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_11/conv2d_15/Conv2DConv2Dinputs5sequential_11/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_11/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_11/conv2d_15/BiasAddBiasAdd'sequential_11/conv2d_15/Conv2D:output:06sequential_11/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_11/batch_normalization_18/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_18_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_11/batch_normalization_18/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_18_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_11/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3(sequential_11/conv2d_15/BiasAdd:output:0;sequential_11/batch_normalization_18/ReadVariableOp:value:0=sequential_11/batch_normalization_18/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
&sequential_11/leaky_re_lu_17/LeakyRelu	LeakyRelu9sequential_11/batch_normalization_18/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_11/conv2d_16/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_11/conv2d_16/Conv2DConv2D4sequential_11/leaky_re_lu_17/LeakyRelu:activations:05sequential_11/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_11/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_11/conv2d_16/BiasAddBiasAdd'sequential_11/conv2d_16/Conv2D:output:06sequential_11/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_11/batch_normalization_19/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_11/batch_normalization_19/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_11/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3(sequential_11/conv2d_16/BiasAdd:output:0;sequential_11/batch_normalization_19/ReadVariableOp:value:0=sequential_11/batch_normalization_19/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
&sequential_11/leaky_re_lu_18/LeakyRelu	LeakyRelu9sequential_11/batch_normalization_19/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_12/conv2d_17/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_12/conv2d_17/Conv2DConv2D4sequential_11/leaky_re_lu_18/LeakyRelu:activations:05sequential_12/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
.sequential_12/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_12/conv2d_17/BiasAddBiasAdd'sequential_12/conv2d_17/Conv2D:output:06sequential_12/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
3sequential_12/batch_normalization_20/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_20_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_12/batch_normalization_20/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_20_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_12/batch_normalization_20/FusedBatchNormV3FusedBatchNormV3(sequential_12/conv2d_17/BiasAdd:output:0;sequential_12/batch_normalization_20/ReadVariableOp:value:0=sequential_12/batch_normalization_20/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
&sequential_12/leaky_re_lu_19/LeakyRelu	LeakyRelu9sequential_12/batch_normalization_20/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
-sequential_12/conv2d_18/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_12/conv2d_18/Conv2DConv2D4sequential_12/leaky_re_lu_19/LeakyRelu:activations:05sequential_12/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
.sequential_12/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_12/conv2d_18/BiasAddBiasAdd'sequential_12/conv2d_18/Conv2D:output:06sequential_12/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
3sequential_12/batch_normalization_21/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_21_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_12/batch_normalization_21/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_21_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_12/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3(sequential_12/conv2d_18/BiasAdd:output:0;sequential_12/batch_normalization_21/ReadVariableOp:value:0=sequential_12/batch_normalization_21/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
&sequential_12/leaky_re_lu_20/LeakyRelu	LeakyRelu9sequential_12/batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
-sequential_13/conv2d_19/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_13/conv2d_19/Conv2DConv2D4sequential_12/leaky_re_lu_20/LeakyRelu:activations:05sequential_13/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_13/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_13/conv2d_19/BiasAddBiasAdd'sequential_13/conv2d_19/Conv2D:output:06sequential_13/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_13/batch_normalization_22/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_22_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_13/batch_normalization_22/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_22_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_13/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3(sequential_13/conv2d_19/BiasAdd:output:0;sequential_13/batch_normalization_22/ReadVariableOp:value:0=sequential_13/batch_normalization_22/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
&sequential_13/leaky_re_lu_21/LeakyRelu	LeakyRelu9sequential_13/batch_normalization_22/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
-sequential_13/conv2d_20/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_13/conv2d_20/Conv2DConv2D4sequential_13/leaky_re_lu_21/LeakyRelu:activations:05sequential_13/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_13/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_13/conv2d_20/BiasAddBiasAdd'sequential_13/conv2d_20/Conv2D:output:06sequential_13/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
3sequential_13/batch_normalization_23/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_13/batch_normalization_23/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_13/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3(sequential_13/conv2d_20/BiasAdd:output:0;sequential_13/batch_normalization_23/ReadVariableOp:value:0=sequential_13/batch_normalization_23/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
&sequential_13/leaky_re_lu_22/LeakyRelu	LeakyRelu9sequential_13/batch_normalization_23/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
-sequential_14/conv2d_21/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_14/conv2d_21/Conv2DConv2D4sequential_13/leaky_re_lu_22/LeakyRelu:activations:05sequential_14/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_14/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_14/conv2d_21/BiasAddBiasAdd'sequential_14/conv2d_21/Conv2D:output:06sequential_14/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
3sequential_14/batch_normalization_24/ReadVariableOpReadVariableOp<sequential_14_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype0?
5sequential_14/batch_normalization_24/ReadVariableOp_1ReadVariableOp>sequential_14_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Dsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Fsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5sequential_14/batch_normalization_24/FusedBatchNormV3FusedBatchNormV3(sequential_14/conv2d_21/BiasAdd:output:0;sequential_14/batch_normalization_24/ReadVariableOp:value:0=sequential_14/batch_normalization_24/ReadVariableOp_1:value:0Lsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
&sequential_14/leaky_re_lu_23/LeakyRelu	LeakyRelu9sequential_14/batch_normalization_24/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>?
-sequential_14/conv2d_22/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
sequential_14/conv2d_22/Conv2DConv2D4sequential_14/leaky_re_lu_23/LeakyRelu:activations:05sequential_14/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_14/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_14/conv2d_22/BiasAddBiasAdd'sequential_14/conv2d_22/Conv2D:output:06sequential_14/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
3sequential_14/batch_normalization_25/ReadVariableOpReadVariableOp<sequential_14_batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype0?
5sequential_14/batch_normalization_25/ReadVariableOp_1ReadVariableOp>sequential_14_batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Dsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Fsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5sequential_14/batch_normalization_25/FusedBatchNormV3FusedBatchNormV3(sequential_14/conv2d_22/BiasAdd:output:0;sequential_14/batch_normalization_25/ReadVariableOp:value:0=sequential_14/batch_normalization_25/ReadVariableOp_1:value:0Lsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
&sequential_14/leaky_re_lu_24/LeakyRelu	LeakyRelu9sequential_14/batch_normalization_25/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
flatten_1/ReshapeReshape4sequential_14/leaky_re_lu_24/LeakyRelu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
latent/MatMul/ReadVariableOpReadVariableOp%latent_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
latent/MatMulMatMulflatten_1/Reshape:output:0$latent/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
latent/BiasAdd/ReadVariableOpReadVariableOp&latent_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
latent/BiasAddBiasAddlatent/MatMul:product:0%latent/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????g
IdentityIdentitylatent/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^latent/BiasAdd/ReadVariableOp^latent/MatMul/ReadVariableOpE^sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_18/ReadVariableOp6^sequential_11/batch_normalization_18/ReadVariableOp_1E^sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_19/ReadVariableOp6^sequential_11/batch_normalization_19/ReadVariableOp_1/^sequential_11/conv2d_15/BiasAdd/ReadVariableOp.^sequential_11/conv2d_15/Conv2D/ReadVariableOp/^sequential_11/conv2d_16/BiasAdd/ReadVariableOp.^sequential_11/conv2d_16/Conv2D/ReadVariableOpE^sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_20/ReadVariableOp6^sequential_12/batch_normalization_20/ReadVariableOp_1E^sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_21/ReadVariableOp6^sequential_12/batch_normalization_21/ReadVariableOp_1/^sequential_12/conv2d_17/BiasAdd/ReadVariableOp.^sequential_12/conv2d_17/Conv2D/ReadVariableOp/^sequential_12/conv2d_18/BiasAdd/ReadVariableOp.^sequential_12/conv2d_18/Conv2D/ReadVariableOpE^sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_22/ReadVariableOp6^sequential_13/batch_normalization_22/ReadVariableOp_1E^sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_23/ReadVariableOp6^sequential_13/batch_normalization_23/ReadVariableOp_1/^sequential_13/conv2d_19/BiasAdd/ReadVariableOp.^sequential_13/conv2d_19/Conv2D/ReadVariableOp/^sequential_13/conv2d_20/BiasAdd/ReadVariableOp.^sequential_13/conv2d_20/Conv2D/ReadVariableOpE^sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpG^sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_14^sequential_14/batch_normalization_24/ReadVariableOp6^sequential_14/batch_normalization_24/ReadVariableOp_1E^sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpG^sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_14^sequential_14/batch_normalization_25/ReadVariableOp6^sequential_14/batch_normalization_25/ReadVariableOp_1/^sequential_14/conv2d_21/BiasAdd/ReadVariableOp.^sequential_14/conv2d_21/Conv2D/ReadVariableOp/^sequential_14/conv2d_22/BiasAdd/ReadVariableOp.^sequential_14/conv2d_22/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
latent/BiasAdd/ReadVariableOplatent/BiasAdd/ReadVariableOp2<
latent/MatMul/ReadVariableOplatent/MatMul/ReadVariableOp2?
Dsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2?
Fsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_18/ReadVariableOp3sequential_11/batch_normalization_18/ReadVariableOp2n
5sequential_11/batch_normalization_18/ReadVariableOp_15sequential_11/batch_normalization_18/ReadVariableOp_12?
Dsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2?
Fsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_19/ReadVariableOp3sequential_11/batch_normalization_19/ReadVariableOp2n
5sequential_11/batch_normalization_19/ReadVariableOp_15sequential_11/batch_normalization_19/ReadVariableOp_12`
.sequential_11/conv2d_15/BiasAdd/ReadVariableOp.sequential_11/conv2d_15/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_15/Conv2D/ReadVariableOp-sequential_11/conv2d_15/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_16/BiasAdd/ReadVariableOp.sequential_11/conv2d_16/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_16/Conv2D/ReadVariableOp-sequential_11/conv2d_16/Conv2D/ReadVariableOp2?
Dsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp2?
Fsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_20/ReadVariableOp3sequential_12/batch_normalization_20/ReadVariableOp2n
5sequential_12/batch_normalization_20/ReadVariableOp_15sequential_12/batch_normalization_20/ReadVariableOp_12?
Dsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2?
Fsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_21/ReadVariableOp3sequential_12/batch_normalization_21/ReadVariableOp2n
5sequential_12/batch_normalization_21/ReadVariableOp_15sequential_12/batch_normalization_21/ReadVariableOp_12`
.sequential_12/conv2d_17/BiasAdd/ReadVariableOp.sequential_12/conv2d_17/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_17/Conv2D/ReadVariableOp-sequential_12/conv2d_17/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_18/BiasAdd/ReadVariableOp.sequential_12/conv2d_18/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_18/Conv2D/ReadVariableOp-sequential_12/conv2d_18/Conv2D/ReadVariableOp2?
Dsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_22/ReadVariableOp3sequential_13/batch_normalization_22/ReadVariableOp2n
5sequential_13/batch_normalization_22/ReadVariableOp_15sequential_13/batch_normalization_22/ReadVariableOp_12?
Dsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_23/ReadVariableOp3sequential_13/batch_normalization_23/ReadVariableOp2n
5sequential_13/batch_normalization_23/ReadVariableOp_15sequential_13/batch_normalization_23/ReadVariableOp_12`
.sequential_13/conv2d_19/BiasAdd/ReadVariableOp.sequential_13/conv2d_19/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_19/Conv2D/ReadVariableOp-sequential_13/conv2d_19/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_20/BiasAdd/ReadVariableOp.sequential_13/conv2d_20/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_20/Conv2D/ReadVariableOp-sequential_13/conv2d_20/Conv2D/ReadVariableOp2?
Dsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpDsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
Fsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Fsequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12j
3sequential_14/batch_normalization_24/ReadVariableOp3sequential_14/batch_normalization_24/ReadVariableOp2n
5sequential_14/batch_normalization_24/ReadVariableOp_15sequential_14/batch_normalization_24/ReadVariableOp_12?
Dsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpDsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
Fsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Fsequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12j
3sequential_14/batch_normalization_25/ReadVariableOp3sequential_14/batch_normalization_25/ReadVariableOp2n
5sequential_14/batch_normalization_25/ReadVariableOp_15sequential_14/batch_normalization_25/ReadVariableOp_12`
.sequential_14/conv2d_21/BiasAdd/ReadVariableOp.sequential_14/conv2d_21/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_21/Conv2D/ReadVariableOp-sequential_14/conv2d_21/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_22/BiasAdd/ReadVariableOp.sequential_14/conv2d_22/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_22/Conv2D/ReadVariableOp-sequential_14/conv2d_22/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
-__inference_sequential_12_layer_call_fn_51837
conv2d_17_input!
unknown:
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51781w
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
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_17_input
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56124

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
?
?
-__inference_sequential_13_layer_call_fn_54951

inputs!
unknown: 
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_52141w
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
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55800

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
?
?
)__inference_conv2d_22_layer_call_fn_56332

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_52672w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_18_layer_call_fn_55310

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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_50929w
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
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_51516

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????00*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_52672

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?!
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_51209

inputs)
conv2d_15_51178:
conv2d_15_51180:*
batch_normalization_18_51183:*
batch_normalization_18_51185:*
batch_normalization_18_51187:*
batch_normalization_18_51189:)
conv2d_16_51193:
conv2d_16_51195:*
batch_normalization_19_51198:*
batch_normalization_19_51200:*
batch_normalization_19_51202:*
batch_normalization_19_51204:
identity??.batch_normalization_18/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_51178conv2d_15_51180*
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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_50906?
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_18_51183batch_normalization_18_51185batch_normalization_18_51187batch_normalization_18_51189*
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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_51123?
leaky_re_lu_17/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_50944?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0conv2d_16_51193conv2d_16_51195*
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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_50956?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_19_51198batch_normalization_19_51200batch_normalization_19_51202batch_normalization_19_51204*
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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_51063?
leaky_re_lu_18/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_50994~
IdentityIdentity'leaky_re_lu_18/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
-__inference_sequential_11_layer_call_fn_54680

inputs!
unknown:
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_51209w
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
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
-__inference_sequential_12_layer_call_fn_51596
conv2d_17_input!
unknown:
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51569w
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
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_17_input
?
?
)__inference_conv2d_18_layer_call_fn_55720

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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_51528w
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
:?????????00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_56323

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55971

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
?
?
'__inference_model_3_layer_call_fn_53284
input_3!
unknown:
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

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: @

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@$

unknown_41:@@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@

unknown_47:
??

unknown_48:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_53181p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????00
!
_user_specified_name	input_3
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56160

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
?"
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_51333
conv2d_15_input)
conv2d_15_51302:
conv2d_15_51304:*
batch_normalization_18_51307:*
batch_normalization_18_51309:*
batch_normalization_18_51311:*
batch_normalization_18_51313:)
conv2d_16_51317:
conv2d_16_51319:*
batch_normalization_19_51322:*
batch_normalization_19_51324:*
batch_normalization_19_51326:*
batch_normalization_19_51328:
identity??.batch_normalization_18/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputconv2d_15_51302conv2d_15_51304*
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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_50906?
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_18_51307batch_normalization_18_51309batch_normalization_18_51311batch_normalization_18_51313*
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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_51123?
leaky_re_lu_17/PartitionedCallPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_50944?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0conv2d_16_51317conv2d_16_51319*
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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_50956?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_19_51322batch_normalization_19_51324batch_normalization_19_51326batch_normalization_19_51328*
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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_51063?
leaky_re_lu_18/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_50994~
IdentityIdentity'leaky_re_lu_18/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_15_input
?	
?
6__inference_batch_normalization_20_layer_call_fn_55590

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
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51355?
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
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_52088

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52563

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56259

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56430

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
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
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?"
?
H__inference_sequential_12_layer_call_and_return_conditional_losses_51905
conv2d_17_input)
conv2d_17_51874:
conv2d_17_51876:*
batch_normalization_20_51879:*
batch_normalization_20_51881:*
batch_normalization_20_51883:*
batch_normalization_20_51885:)
conv2d_18_51889:
conv2d_18_51891:*
batch_normalization_21_51894:*
batch_normalization_21_51896:*
batch_normalization_21_51898:*
batch_normalization_21_51900:
identity??.batch_normalization_20/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputconv2d_17_51874conv2d_17_51876*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51478?
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_20_51879batch_normalization_20_51881batch_normalization_20_51883batch_normalization_20_51885*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51695?
leaky_re_lu_19/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_51516?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_19/PartitionedCall:output:0conv2d_18_51889conv2d_18_51891*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_51528?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_21_51894batch_normalization_21_51896batch_normalization_21_51898batch_normalization_21_51900*
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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51635?
leaky_re_lu_20/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_51566~
IdentityIdentity'leaky_re_lu_20/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_17_input
?
J
.__inference_leaky_re_lu_23_layer_call_fn_56318

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
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_52660h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_55233

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55683

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
7:?????????00:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_52073

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
7:????????? : : : : :*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52839

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
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
:?????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?"
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_53049
conv2d_21_input)
conv2d_21_53018: @
conv2d_21_53020:@*
batch_normalization_24_53023:@*
batch_normalization_24_53025:@*
batch_normalization_24_53027:@*
batch_normalization_24_53029:@)
conv2d_22_53033:@@
conv2d_22_53035:@*
batch_normalization_25_53038:@*
batch_normalization_25_53040:@*
batch_normalization_25_53042:@*
batch_normalization_25_53044:@
identity??.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallconv2d_21_inputconv2d_21_53018conv2d_21_53020*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_52622?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0batch_normalization_24_53023batch_normalization_24_53025batch_normalization_24_53027batch_normalization_24_53029*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52839?
leaky_re_lu_23/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_52660?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0conv2d_22_53033conv2d_22_53035*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_52672?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_25_53038batch_normalization_25_53040batch_normalization_25_53042batch_normalization_25_53044*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52779?
leaky_re_lu_24/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_52710~
IdentityIdentity'leaky_re_lu_24/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall:` \
/
_output_shapes
:????????? 
)
_user_specified_nameconv2d_21_input
?	
?
6__inference_batch_normalization_24_layer_call_fn_56202

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52499?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_52050

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
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
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
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
?
?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56277

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
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
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_16_layer_call_fn_55414

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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_50956w
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
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55953

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
?
?
6__inference_batch_normalization_21_layer_call_fn_55769

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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51551w
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
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_50929

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
?F
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_55222

inputsB
(conv2d_21_conv2d_readvariableop_resource: @7
)conv2d_21_biasadd_readvariableop_resource:@<
.batch_normalization_24_readvariableop_resource:@>
0batch_normalization_24_readvariableop_1_resource:@M
?batch_normalization_24_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_22_conv2d_readvariableop_resource:@@7
)conv2d_22_biasadd_readvariableop_resource:@<
.batch_normalization_25_readvariableop_resource:@>
0batch_normalization_25_readvariableop_1_resource:@M
?batch_normalization_25_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource:@
identity??%batch_normalization_24/AssignNewValue?'batch_normalization_24/AssignNewValue_1?6batch_normalization_24/FusedBatchNormV3/ReadVariableOp?8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_24/ReadVariableOp?'batch_normalization_24/ReadVariableOp_1?%batch_normalization_25/AssignNewValue?'batch_normalization_25/AssignNewValue_1?6batch_normalization_25/FusedBatchNormV3/ReadVariableOp?8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_25/ReadVariableOp?'batch_normalization_25/ReadVariableOp_1? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_21/Conv2DConv2Dinputs'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
%batch_normalization_24/ReadVariableOpReadVariableOp.batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_24/ReadVariableOp_1ReadVariableOp0batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_24/FusedBatchNormV3FusedBatchNormV3conv2d_21/BiasAdd:output:0-batch_normalization_24/ReadVariableOp:value:0/batch_normalization_24/ReadVariableOp_1:value:0>batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_24/AssignNewValueAssignVariableOp?batch_normalization_24_fusedbatchnormv3_readvariableop_resource4batch_normalization_24/FusedBatchNormV3:batch_mean:07^batch_normalization_24/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_24/AssignNewValue_1AssignVariableOpAbatch_normalization_24_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_24/FusedBatchNormV3:batch_variance:09^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_23/LeakyRelu	LeakyRelu+batch_normalization_24/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_22/Conv2DConv2D&leaky_re_lu_23/LeakyRelu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
%batch_normalization_25/ReadVariableOpReadVariableOp.batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_25/ReadVariableOp_1ReadVariableOp0batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_25/FusedBatchNormV3FusedBatchNormV3conv2d_22/BiasAdd:output:0-batch_normalization_25/ReadVariableOp:value:0/batch_normalization_25/ReadVariableOp_1:value:0>batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_25/AssignNewValueAssignVariableOp?batch_normalization_25_fusedbatchnormv3_readvariableop_resource4batch_normalization_25/FusedBatchNormV3:batch_mean:07^batch_normalization_25/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_25/AssignNewValue_1AssignVariableOpAbatch_normalization_25_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_25/FusedBatchNormV3:batch_variance:09^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_24/LeakyRelu	LeakyRelu+batch_normalization_25/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>}
IdentityIdentity&leaky_re_lu_24/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp&^batch_normalization_24/AssignNewValue(^batch_normalization_24/AssignNewValue_17^batch_normalization_24/FusedBatchNormV3/ReadVariableOp9^batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_24/ReadVariableOp(^batch_normalization_24/ReadVariableOp_1&^batch_normalization_25/AssignNewValue(^batch_normalization_25/AssignNewValue_17^batch_normalization_25/FusedBatchNormV3/ReadVariableOp9^batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_25/ReadVariableOp(^batch_normalization_25/ReadVariableOp_1!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2N
%batch_normalization_24/AssignNewValue%batch_normalization_24/AssignNewValue2R
'batch_normalization_24/AssignNewValue_1'batch_normalization_24/AssignNewValue_12p
6batch_normalization_24/FusedBatchNormV3/ReadVariableOp6batch_normalization_24/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_24/FusedBatchNormV3/ReadVariableOp_18batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_24/ReadVariableOp%batch_normalization_24/ReadVariableOp2R
'batch_normalization_24/ReadVariableOp_1'batch_normalization_24/ReadVariableOp_12N
%batch_normalization_25/AssignNewValue%batch_normalization_25/AssignNewValue2R
'batch_normalization_25/AssignNewValue_1'batch_normalization_25/AssignNewValue_12p
6batch_normalization_25/FusedBatchNormV3/ReadVariableOp6batch_normalization_25/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_25/FusedBatchNormV3/ReadVariableOp_18batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_25/ReadVariableOp%batch_normalization_25/ReadVariableOp2R
'batch_normalization_25/ReadVariableOp_1'batch_normalization_25/ReadVariableOp_12D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_18_layer_call_fn_55297

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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_50814?
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

?
D__inference_conv2d_21_layer_call_and_return_conditional_losses_56189

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51569

inputs)
conv2d_17_51479:
conv2d_17_51481:*
batch_normalization_20_51502:*
batch_normalization_20_51504:*
batch_normalization_20_51506:*
batch_normalization_20_51508:)
conv2d_18_51529:
conv2d_18_51531:*
batch_normalization_21_51552:*
batch_normalization_21_51554:*
batch_normalization_21_51556:*
batch_normalization_21_51558:
identity??.batch_normalization_20/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_17_51479conv2d_17_51481*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_51478?
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_20_51502batch_normalization_20_51504batch_normalization_20_51506batch_normalization_20_51508*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51501?
leaky_re_lu_19/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_51516?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_19/PartitionedCall:output:0conv2d_18_51529conv2d_18_51531*
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_51528?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0batch_normalization_21_51552batch_normalization_21_51554batch_normalization_21_51556batch_normalization_21_51558*
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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51551?
leaky_re_lu_20/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_51566~
IdentityIdentity'leaky_re_lu_20/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?

?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_55424

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
?

?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_50906

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_54262

inputs!
unknown:
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

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: @

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@$

unknown_41:@@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@

unknown_47:
??

unknown_48:	?
identity??StatefulPartitionedCall?
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_53517p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?!
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_52141

inputs)
conv2d_19_52051: 
conv2d_19_52053: *
batch_normalization_22_52074: *
batch_normalization_22_52076: *
batch_normalization_22_52078: *
batch_normalization_22_52080: )
conv2d_20_52101:  
conv2d_20_52103: *
batch_normalization_23_52124: *
batch_normalization_23_52126: *
batch_normalization_23_52128: *
batch_normalization_23_52130: 
identity??.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_19_52051conv2d_19_52053*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_52050?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0batch_normalization_22_52074batch_normalization_22_52076batch_normalization_22_52078batch_normalization_22_52080*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_52073?
leaky_re_lu_21/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_52088?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0conv2d_20_52101conv2d_20_52103*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_52100?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_23_52124batch_normalization_23_52126batch_normalization_23_52128batch_normalization_23_52130*
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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52123?
leaky_re_lu_22/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_52138~
IdentityIdentity'leaky_re_lu_22/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_11_layer_call_fn_54651

inputs!
unknown:
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_50997w
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
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?"
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_52443
conv2d_19_input)
conv2d_19_52412: 
conv2d_19_52414: *
batch_normalization_22_52417: *
batch_normalization_22_52419: *
batch_normalization_22_52421: *
batch_normalization_22_52423: )
conv2d_20_52427:  
conv2d_20_52429: *
batch_normalization_23_52432: *
batch_normalization_23_52434: *
batch_normalization_23_52436: *
batch_normalization_23_52438: 
identity??.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCallconv2d_19_inputconv2d_19_52412conv2d_19_52414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_52050?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0batch_normalization_22_52417batch_normalization_22_52419batch_normalization_22_52421batch_normalization_22_52423*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_52073?
leaky_re_lu_21/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_52088?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0conv2d_20_52427conv2d_20_52429*
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_52100?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_23_52432batch_normalization_23_52434batch_normalization_23_52436batch_normalization_23_52438*
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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52123?
leaky_re_lu_22/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
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
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_52138~
IdentityIdentity'leaky_re_lu_22/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_19_input
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_50814

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
?
e
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_56170

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
?
e
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_55405

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
?3
?
B__inference_model_3_layer_call_and_return_conditional_losses_53181

inputs-
sequential_11_53056:!
sequential_11_53058:!
sequential_11_53060:!
sequential_11_53062:!
sequential_11_53064:!
sequential_11_53066:-
sequential_11_53068:!
sequential_11_53070:!
sequential_11_53072:!
sequential_11_53074:!
sequential_11_53076:!
sequential_11_53078:-
sequential_12_53081:!
sequential_12_53083:!
sequential_12_53085:!
sequential_12_53087:!
sequential_12_53089:!
sequential_12_53091:-
sequential_12_53093:!
sequential_12_53095:!
sequential_12_53097:!
sequential_12_53099:!
sequential_12_53101:!
sequential_12_53103:-
sequential_13_53106: !
sequential_13_53108: !
sequential_13_53110: !
sequential_13_53112: !
sequential_13_53114: !
sequential_13_53116: -
sequential_13_53118:  !
sequential_13_53120: !
sequential_13_53122: !
sequential_13_53124: !
sequential_13_53126: !
sequential_13_53128: -
sequential_14_53131: @!
sequential_14_53133:@!
sequential_14_53135:@!
sequential_14_53137:@!
sequential_14_53139:@!
sequential_14_53141:@-
sequential_14_53143:@@!
sequential_14_53145:@!
sequential_14_53147:@!
sequential_14_53149:@!
sequential_14_53151:@!
sequential_14_53153:@ 
latent_53175:
??
latent_53177:	?
identity??latent/StatefulPartitionedCall?%sequential_11/StatefulPartitionedCall?%sequential_12/StatefulPartitionedCall?%sequential_13/StatefulPartitionedCall?%sequential_14/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinputssequential_11_53056sequential_11_53058sequential_11_53060sequential_11_53062sequential_11_53064sequential_11_53066sequential_11_53068sequential_11_53070sequential_11_53072sequential_11_53074sequential_11_53076sequential_11_53078*
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_50997?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCall.sequential_11/StatefulPartitionedCall:output:0sequential_12_53081sequential_12_53083sequential_12_53085sequential_12_53087sequential_12_53089sequential_12_53091sequential_12_53093sequential_12_53095sequential_12_53097sequential_12_53099sequential_12_53101sequential_12_53103*
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51569?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCall.sequential_12/StatefulPartitionedCall:output:0sequential_13_53106sequential_13_53108sequential_13_53110sequential_13_53112sequential_13_53114sequential_13_53116sequential_13_53118sequential_13_53120sequential_13_53122sequential_13_53124sequential_13_53126sequential_13_53128*
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_52141?
%sequential_14/StatefulPartitionedCallStatefulPartitionedCall.sequential_13/StatefulPartitionedCall:output:0sequential_14_53131sequential_14_53133sequential_14_53135sequential_14_53137sequential_14_53139sequential_14_53141sequential_14_53143sequential_14_53145sequential_14_53147sequential_14_53149sequential_14_53151sequential_14_53153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_52713?
flatten_1/PartitionedCallPartitionedCall.sequential_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_53162?
latent/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0latent_53175latent_53177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_53174w
IdentityIdentity'latent/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^latent/StatefulPartitionedCall&^sequential_11/StatefulPartitionedCall&^sequential_12/StatefulPartitionedCall&^sequential_13/StatefulPartitionedCall&^sequential_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall2N
%sequential_14/StatefulPartitionedCall%sequential_14/StatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55512

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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_56342

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_24_layer_call_fn_56471

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
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_52710h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_11_layer_call_fn_51265
conv2d_15_input!
unknown:
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_51209w
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
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????00
)
_user_specified_nameconv2d_15_input
?
?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56466

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
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
:?????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_25_layer_call_fn_56368

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52594?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_17_layer_call_fn_55400

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
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_50944h
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
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_55558

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
?
e
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_52660

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_54157

inputs!
unknown:
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

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: $

unknown_35: @

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@$

unknown_41:@@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@

unknown_47:
??

unknown_48:	?
identity??StatefulPartitionedCall?
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
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_53181p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?8
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_54726

inputsB
(conv2d_15_conv2d_readvariableop_resource:7
)conv2d_15_biasadd_readvariableop_resource:<
.batch_normalization_18_readvariableop_resource:>
0batch_normalization_18_readvariableop_1_resource:M
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_16_conv2d_readvariableop_resource:7
)conv2d_16_biasadd_readvariableop_resource:<
.batch_normalization_19_readvariableop_resource:>
0batch_normalization_19_readvariableop_1_resource:M
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:
identity??6batch_normalization_18/FusedBatchNormV3/ReadVariableOp?8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_18/ReadVariableOp?'batch_normalization_18/ReadVariableOp_1?6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_19/ReadVariableOp?'batch_normalization_19/ReadVariableOp_1? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3conv2d_15/BiasAdd:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_17/LeakyRelu	LeakyRelu+batch_normalization_18/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_16/Conv2DConv2D&leaky_re_lu_17/LeakyRelu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_16/BiasAdd:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
leaky_re_lu_18/LeakyRelu	LeakyRelu+batch_normalization_19/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>}
IdentityIdentity&leaky_re_lu_18/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp7^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_50944

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
6__inference_batch_normalization_19_layer_call_fn_55437

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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_50847?
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
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51355

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
??
?=
 __inference__wrapped_model_50761
input_3X
>model_3_sequential_11_conv2d_15_conv2d_readvariableop_resource:M
?model_3_sequential_11_conv2d_15_biasadd_readvariableop_resource:R
Dmodel_3_sequential_11_batch_normalization_18_readvariableop_resource:T
Fmodel_3_sequential_11_batch_normalization_18_readvariableop_1_resource:c
Umodel_3_sequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:e
Wmodel_3_sequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:X
>model_3_sequential_11_conv2d_16_conv2d_readvariableop_resource:M
?model_3_sequential_11_conv2d_16_biasadd_readvariableop_resource:R
Dmodel_3_sequential_11_batch_normalization_19_readvariableop_resource:T
Fmodel_3_sequential_11_batch_normalization_19_readvariableop_1_resource:c
Umodel_3_sequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:e
Wmodel_3_sequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:X
>model_3_sequential_12_conv2d_17_conv2d_readvariableop_resource:M
?model_3_sequential_12_conv2d_17_biasadd_readvariableop_resource:R
Dmodel_3_sequential_12_batch_normalization_20_readvariableop_resource:T
Fmodel_3_sequential_12_batch_normalization_20_readvariableop_1_resource:c
Umodel_3_sequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_resource:e
Wmodel_3_sequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:X
>model_3_sequential_12_conv2d_18_conv2d_readvariableop_resource:M
?model_3_sequential_12_conv2d_18_biasadd_readvariableop_resource:R
Dmodel_3_sequential_12_batch_normalization_21_readvariableop_resource:T
Fmodel_3_sequential_12_batch_normalization_21_readvariableop_1_resource:c
Umodel_3_sequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:e
Wmodel_3_sequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:X
>model_3_sequential_13_conv2d_19_conv2d_readvariableop_resource: M
?model_3_sequential_13_conv2d_19_biasadd_readvariableop_resource: R
Dmodel_3_sequential_13_batch_normalization_22_readvariableop_resource: T
Fmodel_3_sequential_13_batch_normalization_22_readvariableop_1_resource: c
Umodel_3_sequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_resource: e
Wmodel_3_sequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource: X
>model_3_sequential_13_conv2d_20_conv2d_readvariableop_resource:  M
?model_3_sequential_13_conv2d_20_biasadd_readvariableop_resource: R
Dmodel_3_sequential_13_batch_normalization_23_readvariableop_resource: T
Fmodel_3_sequential_13_batch_normalization_23_readvariableop_1_resource: c
Umodel_3_sequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_resource: e
Wmodel_3_sequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: X
>model_3_sequential_14_conv2d_21_conv2d_readvariableop_resource: @M
?model_3_sequential_14_conv2d_21_biasadd_readvariableop_resource:@R
Dmodel_3_sequential_14_batch_normalization_24_readvariableop_resource:@T
Fmodel_3_sequential_14_batch_normalization_24_readvariableop_1_resource:@c
Umodel_3_sequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_resource:@e
Wmodel_3_sequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource:@X
>model_3_sequential_14_conv2d_22_conv2d_readvariableop_resource:@@M
?model_3_sequential_14_conv2d_22_biasadd_readvariableop_resource:@R
Dmodel_3_sequential_14_batch_normalization_25_readvariableop_resource:@T
Fmodel_3_sequential_14_batch_normalization_25_readvariableop_1_resource:@c
Umodel_3_sequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_resource:@e
Wmodel_3_sequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource:@A
-model_3_latent_matmul_readvariableop_resource:
??=
.model_3_latent_biasadd_readvariableop_resource:	?
identity??%model_3/latent/BiasAdd/ReadVariableOp?$model_3/latent/MatMul/ReadVariableOp?Lmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp?Nmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1?;model_3/sequential_11/batch_normalization_18/ReadVariableOp?=model_3/sequential_11/batch_normalization_18/ReadVariableOp_1?Lmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp?Nmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?;model_3/sequential_11/batch_normalization_19/ReadVariableOp?=model_3/sequential_11/batch_normalization_19/ReadVariableOp_1?6model_3/sequential_11/conv2d_15/BiasAdd/ReadVariableOp?5model_3/sequential_11/conv2d_15/Conv2D/ReadVariableOp?6model_3/sequential_11/conv2d_16/BiasAdd/ReadVariableOp?5model_3/sequential_11/conv2d_16/Conv2D/ReadVariableOp?Lmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp?Nmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?;model_3/sequential_12/batch_normalization_20/ReadVariableOp?=model_3/sequential_12/batch_normalization_20/ReadVariableOp_1?Lmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp?Nmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?;model_3/sequential_12/batch_normalization_21/ReadVariableOp?=model_3/sequential_12/batch_normalization_21/ReadVariableOp_1?6model_3/sequential_12/conv2d_17/BiasAdd/ReadVariableOp?5model_3/sequential_12/conv2d_17/Conv2D/ReadVariableOp?6model_3/sequential_12/conv2d_18/BiasAdd/ReadVariableOp?5model_3/sequential_12/conv2d_18/Conv2D/ReadVariableOp?Lmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp?Nmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1?;model_3/sequential_13/batch_normalization_22/ReadVariableOp?=model_3/sequential_13/batch_normalization_22/ReadVariableOp_1?Lmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Nmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?;model_3/sequential_13/batch_normalization_23/ReadVariableOp?=model_3/sequential_13/batch_normalization_23/ReadVariableOp_1?6model_3/sequential_13/conv2d_19/BiasAdd/ReadVariableOp?5model_3/sequential_13/conv2d_19/Conv2D/ReadVariableOp?6model_3/sequential_13/conv2d_20/BiasAdd/ReadVariableOp?5model_3/sequential_13/conv2d_20/Conv2D/ReadVariableOp?Lmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp?Nmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1?;model_3/sequential_14/batch_normalization_24/ReadVariableOp?=model_3/sequential_14/batch_normalization_24/ReadVariableOp_1?Lmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp?Nmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1?;model_3/sequential_14/batch_normalization_25/ReadVariableOp?=model_3/sequential_14/batch_normalization_25/ReadVariableOp_1?6model_3/sequential_14/conv2d_21/BiasAdd/ReadVariableOp?5model_3/sequential_14/conv2d_21/Conv2D/ReadVariableOp?6model_3/sequential_14/conv2d_22/BiasAdd/ReadVariableOp?5model_3/sequential_14/conv2d_22/Conv2D/ReadVariableOp?
5model_3/sequential_11/conv2d_15/Conv2D/ReadVariableOpReadVariableOp>model_3_sequential_11_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_3/sequential_11/conv2d_15/Conv2DConv2Dinput_3=model_3/sequential_11/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
6model_3/sequential_11/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp?model_3_sequential_11_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_3/sequential_11/conv2d_15/BiasAddBiasAdd/model_3/sequential_11/conv2d_15/Conv2D:output:0>model_3/sequential_11/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
;model_3/sequential_11/batch_normalization_18/ReadVariableOpReadVariableOpDmodel_3_sequential_11_batch_normalization_18_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_3/sequential_11/batch_normalization_18/ReadVariableOp_1ReadVariableOpFmodel_3_sequential_11_batch_normalization_18_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_sequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_sequential_11_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_3/sequential_11/batch_normalization_18/FusedBatchNormV3FusedBatchNormV30model_3/sequential_11/conv2d_15/BiasAdd:output:0Cmodel_3/sequential_11/batch_normalization_18/ReadVariableOp:value:0Emodel_3/sequential_11/batch_normalization_18/ReadVariableOp_1:value:0Tmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
.model_3/sequential_11/leaky_re_lu_17/LeakyRelu	LeakyReluAmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
5model_3/sequential_11/conv2d_16/Conv2D/ReadVariableOpReadVariableOp>model_3_sequential_11_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_3/sequential_11/conv2d_16/Conv2DConv2D<model_3/sequential_11/leaky_re_lu_17/LeakyRelu:activations:0=model_3/sequential_11/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
6model_3/sequential_11/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp?model_3_sequential_11_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_3/sequential_11/conv2d_16/BiasAddBiasAdd/model_3/sequential_11/conv2d_16/Conv2D:output:0>model_3/sequential_11/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
;model_3/sequential_11/batch_normalization_19/ReadVariableOpReadVariableOpDmodel_3_sequential_11_batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_3/sequential_11/batch_normalization_19/ReadVariableOp_1ReadVariableOpFmodel_3_sequential_11_batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_sequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_sequential_11_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_3/sequential_11/batch_normalization_19/FusedBatchNormV3FusedBatchNormV30model_3/sequential_11/conv2d_16/BiasAdd:output:0Cmodel_3/sequential_11/batch_normalization_19/ReadVariableOp:value:0Emodel_3/sequential_11/batch_normalization_19/ReadVariableOp_1:value:0Tmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
.model_3/sequential_11/leaky_re_lu_18/LeakyRelu	LeakyReluAmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
5model_3/sequential_12/conv2d_17/Conv2D/ReadVariableOpReadVariableOp>model_3_sequential_12_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_3/sequential_12/conv2d_17/Conv2DConv2D<model_3/sequential_11/leaky_re_lu_18/LeakyRelu:activations:0=model_3/sequential_12/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
6model_3/sequential_12/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp?model_3_sequential_12_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_3/sequential_12/conv2d_17/BiasAddBiasAdd/model_3/sequential_12/conv2d_17/Conv2D:output:0>model_3/sequential_12/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
;model_3/sequential_12/batch_normalization_20/ReadVariableOpReadVariableOpDmodel_3_sequential_12_batch_normalization_20_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_3/sequential_12/batch_normalization_20/ReadVariableOp_1ReadVariableOpFmodel_3_sequential_12_batch_normalization_20_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_sequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_sequential_12_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_3/sequential_12/batch_normalization_20/FusedBatchNormV3FusedBatchNormV30model_3/sequential_12/conv2d_17/BiasAdd:output:0Cmodel_3/sequential_12/batch_normalization_20/ReadVariableOp:value:0Emodel_3/sequential_12/batch_normalization_20/ReadVariableOp_1:value:0Tmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
is_training( ?
.model_3/sequential_12/leaky_re_lu_19/LeakyRelu	LeakyReluAmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
5model_3/sequential_12/conv2d_18/Conv2D/ReadVariableOpReadVariableOp>model_3_sequential_12_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&model_3/sequential_12/conv2d_18/Conv2DConv2D<model_3/sequential_12/leaky_re_lu_19/LeakyRelu:activations:0=model_3/sequential_12/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
6model_3/sequential_12/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp?model_3_sequential_12_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'model_3/sequential_12/conv2d_18/BiasAddBiasAdd/model_3/sequential_12/conv2d_18/Conv2D:output:0>model_3/sequential_12/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
;model_3/sequential_12/batch_normalization_21/ReadVariableOpReadVariableOpDmodel_3_sequential_12_batch_normalization_21_readvariableop_resource*
_output_shapes
:*
dtype0?
=model_3/sequential_12/batch_normalization_21/ReadVariableOp_1ReadVariableOpFmodel_3_sequential_12_batch_normalization_21_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Lmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_sequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Nmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_sequential_12_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
=model_3/sequential_12/batch_normalization_21/FusedBatchNormV3FusedBatchNormV30model_3/sequential_12/conv2d_18/BiasAdd:output:0Cmodel_3/sequential_12/batch_normalization_21/ReadVariableOp:value:0Emodel_3/sequential_12/batch_normalization_21/ReadVariableOp_1:value:0Tmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
.model_3/sequential_12/leaky_re_lu_20/LeakyRelu	LeakyReluAmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
5model_3/sequential_13/conv2d_19/Conv2D/ReadVariableOpReadVariableOp>model_3_sequential_13_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
&model_3/sequential_13/conv2d_19/Conv2DConv2D<model_3/sequential_12/leaky_re_lu_20/LeakyRelu:activations:0=model_3/sequential_13/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
6model_3/sequential_13/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp?model_3_sequential_13_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
'model_3/sequential_13/conv2d_19/BiasAddBiasAdd/model_3/sequential_13/conv2d_19/Conv2D:output:0>model_3/sequential_13/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
;model_3/sequential_13/batch_normalization_22/ReadVariableOpReadVariableOpDmodel_3_sequential_13_batch_normalization_22_readvariableop_resource*
_output_shapes
: *
dtype0?
=model_3/sequential_13/batch_normalization_22/ReadVariableOp_1ReadVariableOpFmodel_3_sequential_13_batch_normalization_22_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Lmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_sequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Nmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_sequential_13_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
=model_3/sequential_13/batch_normalization_22/FusedBatchNormV3FusedBatchNormV30model_3/sequential_13/conv2d_19/BiasAdd:output:0Cmodel_3/sequential_13/batch_normalization_22/ReadVariableOp:value:0Emodel_3/sequential_13/batch_normalization_22/ReadVariableOp_1:value:0Tmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
.model_3/sequential_13/leaky_re_lu_21/LeakyRelu	LeakyReluAmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
5model_3/sequential_13/conv2d_20/Conv2D/ReadVariableOpReadVariableOp>model_3_sequential_13_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
&model_3/sequential_13/conv2d_20/Conv2DConv2D<model_3/sequential_13/leaky_re_lu_21/LeakyRelu:activations:0=model_3/sequential_13/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
6model_3/sequential_13/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp?model_3_sequential_13_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
'model_3/sequential_13/conv2d_20/BiasAddBiasAdd/model_3/sequential_13/conv2d_20/Conv2D:output:0>model_3/sequential_13/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
;model_3/sequential_13/batch_normalization_23/ReadVariableOpReadVariableOpDmodel_3_sequential_13_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
=model_3/sequential_13/batch_normalization_23/ReadVariableOp_1ReadVariableOpFmodel_3_sequential_13_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Lmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_sequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Nmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_sequential_13_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
=model_3/sequential_13/batch_normalization_23/FusedBatchNormV3FusedBatchNormV30model_3/sequential_13/conv2d_20/BiasAdd:output:0Cmodel_3/sequential_13/batch_normalization_23/ReadVariableOp:value:0Emodel_3/sequential_13/batch_normalization_23/ReadVariableOp_1:value:0Tmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
.model_3/sequential_13/leaky_re_lu_22/LeakyRelu	LeakyReluAmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
5model_3/sequential_14/conv2d_21/Conv2D/ReadVariableOpReadVariableOp>model_3_sequential_14_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
&model_3/sequential_14/conv2d_21/Conv2DConv2D<model_3/sequential_13/leaky_re_lu_22/LeakyRelu:activations:0=model_3/sequential_14/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
6model_3/sequential_14/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp?model_3_sequential_14_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
'model_3/sequential_14/conv2d_21/BiasAddBiasAdd/model_3/sequential_14/conv2d_21/Conv2D:output:0>model_3/sequential_14/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
;model_3/sequential_14/batch_normalization_24/ReadVariableOpReadVariableOpDmodel_3_sequential_14_batch_normalization_24_readvariableop_resource*
_output_shapes
:@*
dtype0?
=model_3/sequential_14/batch_normalization_24/ReadVariableOp_1ReadVariableOpFmodel_3_sequential_14_batch_normalization_24_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Lmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_sequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Nmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_sequential_14_batch_normalization_24_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
=model_3/sequential_14/batch_normalization_24/FusedBatchNormV3FusedBatchNormV30model_3/sequential_14/conv2d_21/BiasAdd:output:0Cmodel_3/sequential_14/batch_normalization_24/ReadVariableOp:value:0Emodel_3/sequential_14/batch_normalization_24/ReadVariableOp_1:value:0Tmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
.model_3/sequential_14/leaky_re_lu_23/LeakyRelu	LeakyReluAmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>?
5model_3/sequential_14/conv2d_22/Conv2D/ReadVariableOpReadVariableOp>model_3_sequential_14_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
&model_3/sequential_14/conv2d_22/Conv2DConv2D<model_3/sequential_14/leaky_re_lu_23/LeakyRelu:activations:0=model_3/sequential_14/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
6model_3/sequential_14/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp?model_3_sequential_14_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
'model_3/sequential_14/conv2d_22/BiasAddBiasAdd/model_3/sequential_14/conv2d_22/Conv2D:output:0>model_3/sequential_14/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
;model_3/sequential_14/batch_normalization_25/ReadVariableOpReadVariableOpDmodel_3_sequential_14_batch_normalization_25_readvariableop_resource*
_output_shapes
:@*
dtype0?
=model_3/sequential_14/batch_normalization_25/ReadVariableOp_1ReadVariableOpFmodel_3_sequential_14_batch_normalization_25_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Lmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_sequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Nmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_sequential_14_batch_normalization_25_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
=model_3/sequential_14/batch_normalization_25/FusedBatchNormV3FusedBatchNormV30model_3/sequential_14/conv2d_22/BiasAdd:output:0Cmodel_3/sequential_14/batch_normalization_25/ReadVariableOp:value:0Emodel_3/sequential_14/batch_normalization_25/ReadVariableOp_1:value:0Tmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
.model_3/sequential_14/leaky_re_lu_24/LeakyRelu	LeakyReluAmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>h
model_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
model_3/flatten_1/ReshapeReshape<model_3/sequential_14/leaky_re_lu_24/LeakyRelu:activations:0 model_3/flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
$model_3/latent/MatMul/ReadVariableOpReadVariableOp-model_3_latent_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_3/latent/MatMulMatMul"model_3/flatten_1/Reshape:output:0,model_3/latent/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%model_3/latent/BiasAdd/ReadVariableOpReadVariableOp.model_3_latent_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_3/latent/BiasAddBiasAddmodel_3/latent/MatMul:product:0-model_3/latent/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
IdentityIdentitymodel_3/latent/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp&^model_3/latent/BiasAdd/ReadVariableOp%^model_3/latent/MatMul/ReadVariableOpM^model_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpO^model_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1<^model_3/sequential_11/batch_normalization_18/ReadVariableOp>^model_3/sequential_11/batch_normalization_18/ReadVariableOp_1M^model_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpO^model_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1<^model_3/sequential_11/batch_normalization_19/ReadVariableOp>^model_3/sequential_11/batch_normalization_19/ReadVariableOp_17^model_3/sequential_11/conv2d_15/BiasAdd/ReadVariableOp6^model_3/sequential_11/conv2d_15/Conv2D/ReadVariableOp7^model_3/sequential_11/conv2d_16/BiasAdd/ReadVariableOp6^model_3/sequential_11/conv2d_16/Conv2D/ReadVariableOpM^model_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpO^model_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1<^model_3/sequential_12/batch_normalization_20/ReadVariableOp>^model_3/sequential_12/batch_normalization_20/ReadVariableOp_1M^model_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpO^model_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1<^model_3/sequential_12/batch_normalization_21/ReadVariableOp>^model_3/sequential_12/batch_normalization_21/ReadVariableOp_17^model_3/sequential_12/conv2d_17/BiasAdd/ReadVariableOp6^model_3/sequential_12/conv2d_17/Conv2D/ReadVariableOp7^model_3/sequential_12/conv2d_18/BiasAdd/ReadVariableOp6^model_3/sequential_12/conv2d_18/Conv2D/ReadVariableOpM^model_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpO^model_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1<^model_3/sequential_13/batch_normalization_22/ReadVariableOp>^model_3/sequential_13/batch_normalization_22/ReadVariableOp_1M^model_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpO^model_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1<^model_3/sequential_13/batch_normalization_23/ReadVariableOp>^model_3/sequential_13/batch_normalization_23/ReadVariableOp_17^model_3/sequential_13/conv2d_19/BiasAdd/ReadVariableOp6^model_3/sequential_13/conv2d_19/Conv2D/ReadVariableOp7^model_3/sequential_13/conv2d_20/BiasAdd/ReadVariableOp6^model_3/sequential_13/conv2d_20/Conv2D/ReadVariableOpM^model_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpO^model_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1<^model_3/sequential_14/batch_normalization_24/ReadVariableOp>^model_3/sequential_14/batch_normalization_24/ReadVariableOp_1M^model_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpO^model_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1<^model_3/sequential_14/batch_normalization_25/ReadVariableOp>^model_3/sequential_14/batch_normalization_25/ReadVariableOp_17^model_3/sequential_14/conv2d_21/BiasAdd/ReadVariableOp6^model_3/sequential_14/conv2d_21/Conv2D/ReadVariableOp7^model_3/sequential_14/conv2d_22/BiasAdd/ReadVariableOp6^model_3/sequential_14/conv2d_22/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%model_3/latent/BiasAdd/ReadVariableOp%model_3/latent/BiasAdd/ReadVariableOp2L
$model_3/latent/MatMul/ReadVariableOp$model_3/latent/MatMul/ReadVariableOp2?
Lmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOpLmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Nmodel_3/sequential_11/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12z
;model_3/sequential_11/batch_normalization_18/ReadVariableOp;model_3/sequential_11/batch_normalization_18/ReadVariableOp2~
=model_3/sequential_11/batch_normalization_18/ReadVariableOp_1=model_3/sequential_11/batch_normalization_18/ReadVariableOp_12?
Lmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOpLmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Nmodel_3/sequential_11/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12z
;model_3/sequential_11/batch_normalization_19/ReadVariableOp;model_3/sequential_11/batch_normalization_19/ReadVariableOp2~
=model_3/sequential_11/batch_normalization_19/ReadVariableOp_1=model_3/sequential_11/batch_normalization_19/ReadVariableOp_12p
6model_3/sequential_11/conv2d_15/BiasAdd/ReadVariableOp6model_3/sequential_11/conv2d_15/BiasAdd/ReadVariableOp2n
5model_3/sequential_11/conv2d_15/Conv2D/ReadVariableOp5model_3/sequential_11/conv2d_15/Conv2D/ReadVariableOp2p
6model_3/sequential_11/conv2d_16/BiasAdd/ReadVariableOp6model_3/sequential_11/conv2d_16/BiasAdd/ReadVariableOp2n
5model_3/sequential_11/conv2d_16/Conv2D/ReadVariableOp5model_3/sequential_11/conv2d_16/Conv2D/ReadVariableOp2?
Lmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOpLmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1Nmodel_3/sequential_12/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12z
;model_3/sequential_12/batch_normalization_20/ReadVariableOp;model_3/sequential_12/batch_normalization_20/ReadVariableOp2~
=model_3/sequential_12/batch_normalization_20/ReadVariableOp_1=model_3/sequential_12/batch_normalization_20/ReadVariableOp_12?
Lmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOpLmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Nmodel_3/sequential_12/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12z
;model_3/sequential_12/batch_normalization_21/ReadVariableOp;model_3/sequential_12/batch_normalization_21/ReadVariableOp2~
=model_3/sequential_12/batch_normalization_21/ReadVariableOp_1=model_3/sequential_12/batch_normalization_21/ReadVariableOp_12p
6model_3/sequential_12/conv2d_17/BiasAdd/ReadVariableOp6model_3/sequential_12/conv2d_17/BiasAdd/ReadVariableOp2n
5model_3/sequential_12/conv2d_17/Conv2D/ReadVariableOp5model_3/sequential_12/conv2d_17/Conv2D/ReadVariableOp2p
6model_3/sequential_12/conv2d_18/BiasAdd/ReadVariableOp6model_3/sequential_12/conv2d_18/BiasAdd/ReadVariableOp2n
5model_3/sequential_12/conv2d_18/Conv2D/ReadVariableOp5model_3/sequential_12/conv2d_18/Conv2D/ReadVariableOp2?
Lmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOpLmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Nmodel_3/sequential_13/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12z
;model_3/sequential_13/batch_normalization_22/ReadVariableOp;model_3/sequential_13/batch_normalization_22/ReadVariableOp2~
=model_3/sequential_13/batch_normalization_22/ReadVariableOp_1=model_3/sequential_13/batch_normalization_22/ReadVariableOp_12?
Lmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOpLmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Nmodel_3/sequential_13/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12z
;model_3/sequential_13/batch_normalization_23/ReadVariableOp;model_3/sequential_13/batch_normalization_23/ReadVariableOp2~
=model_3/sequential_13/batch_normalization_23/ReadVariableOp_1=model_3/sequential_13/batch_normalization_23/ReadVariableOp_12p
6model_3/sequential_13/conv2d_19/BiasAdd/ReadVariableOp6model_3/sequential_13/conv2d_19/BiasAdd/ReadVariableOp2n
5model_3/sequential_13/conv2d_19/Conv2D/ReadVariableOp5model_3/sequential_13/conv2d_19/Conv2D/ReadVariableOp2p
6model_3/sequential_13/conv2d_20/BiasAdd/ReadVariableOp6model_3/sequential_13/conv2d_20/BiasAdd/ReadVariableOp2n
5model_3/sequential_13/conv2d_20/Conv2D/ReadVariableOp5model_3/sequential_13/conv2d_20/Conv2D/ReadVariableOp2?
Lmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOpLmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_1Nmodel_3/sequential_14/batch_normalization_24/FusedBatchNormV3/ReadVariableOp_12z
;model_3/sequential_14/batch_normalization_24/ReadVariableOp;model_3/sequential_14/batch_normalization_24/ReadVariableOp2~
=model_3/sequential_14/batch_normalization_24/ReadVariableOp_1=model_3/sequential_14/batch_normalization_24/ReadVariableOp_12?
Lmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOpLmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_1Nmodel_3/sequential_14/batch_normalization_25/FusedBatchNormV3/ReadVariableOp_12z
;model_3/sequential_14/batch_normalization_25/ReadVariableOp;model_3/sequential_14/batch_normalization_25/ReadVariableOp2~
=model_3/sequential_14/batch_normalization_25/ReadVariableOp_1=model_3/sequential_14/batch_normalization_25/ReadVariableOp_12p
6model_3/sequential_14/conv2d_21/BiasAdd/ReadVariableOp6model_3/sequential_14/conv2d_21/BiasAdd/ReadVariableOp2n
5model_3/sequential_14/conv2d_21/Conv2D/ReadVariableOp5model_3/sequential_14/conv2d_21/Conv2D/ReadVariableOp2p
6model_3/sequential_14/conv2d_22/BiasAdd/ReadVariableOp6model_3/sequential_14/conv2d_22/BiasAdd/ReadVariableOp2n
5model_3/sequential_14/conv2d_22/Conv2D/ReadVariableOp5model_3/sequential_14/conv2d_22/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????00
!
_user_specified_name	input_3
?
?
6__inference_batch_normalization_18_layer_call_fn_55323

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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_51123w
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
6__inference_batch_normalization_23_layer_call_fn_56049

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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_51991?
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
?
-__inference_sequential_14_layer_call_fn_52740
conv2d_21_input!
unknown: @
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_52713w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
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
_user_specified_nameconv2d_21_input
?
e
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_56476

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55701

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
7:?????????00:::::*
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
:?????????00?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_55864

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
?
6__inference_batch_normalization_23_layer_call_fn_56062

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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52022?
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_52622

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51450

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
?F
?
H__inference_sequential_11_layer_call_and_return_conditional_losses_54772

inputsB
(conv2d_15_conv2d_readvariableop_resource:7
)conv2d_15_biasadd_readvariableop_resource:<
.batch_normalization_18_readvariableop_resource:>
0batch_normalization_18_readvariableop_1_resource:M
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_16_conv2d_readvariableop_resource:7
)conv2d_16_biasadd_readvariableop_resource:<
.batch_normalization_19_readvariableop_resource:>
0batch_normalization_19_readvariableop_1_resource:M
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:
identity??%batch_normalization_18/AssignNewValue?'batch_normalization_18/AssignNewValue_1?6batch_normalization_18/FusedBatchNormV3/ReadVariableOp?8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_18/ReadVariableOp?'batch_normalization_18/ReadVariableOp_1?%batch_normalization_19/AssignNewValue?'batch_normalization_19/AssignNewValue_1?6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_19/ReadVariableOp?'batch_normalization_19/ReadVariableOp_1? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3conv2d_15/BiasAdd:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_18/AssignNewValueAssignVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource4batch_normalization_18/FusedBatchNormV3:batch_mean:07^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_18/AssignNewValue_1AssignVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_18/FusedBatchNormV3:batch_variance:09^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_17/LeakyRelu	LeakyRelu+batch_normalization_18/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_16/Conv2DConv2D&leaky_re_lu_17/LeakyRelu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_16/BiasAdd:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_18/LeakyRelu	LeakyRelu+batch_normalization_19/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>}
IdentityIdentity&leaky_re_lu_18/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????00?
NoOpNoOp&^batch_normalization_18/AssignNewValue(^batch_normalization_18/AssignNewValue_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2N
%batch_normalization_18/AssignNewValue%batch_normalization_18/AssignNewValue2R
'batch_normalization_18/AssignNewValue_1'batch_normalization_18/AssignNewValue_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52499

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_12_layer_call_fn_54830

inputs!
unknown:
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51781w
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
3:?????????00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55854

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
?_
?
__inference__traced_save_56649
file_prefix,
(savev2_latent_kernel_read_readvariableop*
&savev2_latent_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop;
7savev2_batch_normalization_24_gamma_read_readvariableop:
6savev2_batch_normalization_24_beta_read_readvariableopA
=savev2_batch_normalization_24_moving_mean_read_readvariableopE
Asavev2_batch_normalization_24_moving_variance_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop;
7savev2_batch_normalization_25_gamma_read_readvariableop:
6savev2_batch_normalization_25_beta_read_readvariableopA
=savev2_batch_normalization_25_moving_mean_read_readvariableopE
Asavev2_batch_normalization_25_moving_variance_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*?
value?B?3B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_latent_kernel_read_readvariableop&savev2_latent_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop7savev2_batch_normalization_24_gamma_read_readvariableop6savev2_batch_normalization_24_beta_read_readvariableop=savev2_batch_normalization_24_moving_mean_read_readvariableopAsavev2_batch_normalization_24_moving_variance_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop7savev2_batch_normalization_25_gamma_read_readvariableop6savev2_batch_normalization_25_beta_read_readvariableop=savev2_batch_normalization_25_moving_mean_read_readvariableopAsavev2_batch_normalization_25_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *A
dtypes7
523?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?::::::::::::::::::::::::: : : : : : :  : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 
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
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :  

_output_shapes
: :,!(
&
_output_shapes
:  : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: :,'(
&
_output_shapes
: @: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@: +

_output_shapes
:@: ,

_output_shapes
:@:,-(
&
_output_shapes
:@@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@: 1

_output_shapes
:@: 2

_output_shapes
:@:3

_output_shapes
: 
?
?
&__inference_latent_layer_call_fn_55242

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_53174p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?F
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_55072

inputsB
(conv2d_19_conv2d_readvariableop_resource: 7
)conv2d_19_biasadd_readvariableop_resource: <
.batch_normalization_22_readvariableop_resource: >
0batch_normalization_22_readvariableop_1_resource: M
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_20_conv2d_readvariableop_resource:  7
)conv2d_20_biasadd_readvariableop_resource: <
.batch_normalization_23_readvariableop_resource: >
0batch_normalization_23_readvariableop_1_resource: M
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: 
identity??%batch_normalization_22/AssignNewValue?'batch_normalization_22/AssignNewValue_1?6batch_normalization_22/FusedBatchNormV3/ReadVariableOp?8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_22/ReadVariableOp?'batch_normalization_22/ReadVariableOp_1?%batch_normalization_23/AssignNewValue?'batch_normalization_23/AssignNewValue_1?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_19/BiasAdd:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_22/AssignNewValueAssignVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource4batch_normalization_22/FusedBatchNormV3:batch_mean:07^batch_normalization_22/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_22/AssignNewValue_1AssignVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_22/FusedBatchNormV3:batch_variance:09^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_21/LeakyRelu	LeakyRelu+batch_normalization_22/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_20/Conv2DConv2D&leaky_re_lu_21/LeakyRelu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_20/BiasAdd:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_22/LeakyRelu	LeakyRelu+batch_normalization_23/FusedBatchNormV3:y:0*/
_output_shapes
:????????? *
alpha%???>}
IdentityIdentity&leaky_re_lu_22/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp&^batch_normalization_22/AssignNewValue(^batch_normalization_22/AssignNewValue_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2N
%batch_normalization_22/AssignNewValue%batch_normalization_22/AssignNewValue2R
'batch_normalization_22/AssignNewValue_1'batch_normalization_22/AssignNewValue_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55530

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
?
?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51551

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
?"
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_53015
conv2d_21_input)
conv2d_21_52984: @
conv2d_21_52986:@*
batch_normalization_24_52989:@*
batch_normalization_24_52991:@*
batch_normalization_24_52993:@*
batch_normalization_24_52995:@)
conv2d_22_52999:@@
conv2d_22_53001:@*
batch_normalization_25_53004:@*
batch_normalization_25_53006:@*
batch_normalization_25_53008:@*
batch_normalization_25_53010:@
identity??.batch_normalization_24/StatefulPartitionedCall?.batch_normalization_25/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallconv2d_21_inputconv2d_21_52984conv2d_21_52986*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_52622?
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0batch_normalization_24_52989batch_normalization_24_52991batch_normalization_24_52993batch_normalization_24_52995*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52645?
leaky_re_lu_23/PartitionedCallPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_52660?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0conv2d_22_52999conv2d_22_53001*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_52672?
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_25_53004batch_normalization_25_53006batch_normalization_25_53008batch_normalization_25_53010*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52695?
leaky_re_lu_24/PartitionedCallPartitionedCall7batch_normalization_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_52710~
IdentityIdentity'leaky_re_lu_24/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:????????? : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall:` \
/
_output_shapes
:????????? 
)
_user_specified_nameconv2d_21_input
?
e
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_55711

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????00*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????00"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
-__inference_sequential_13_layer_call_fn_52168
conv2d_19_input!
unknown: 
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_52141w
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
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_19_input
?

?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_51528

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
r
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
:?????????00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_50878

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
6__inference_batch_normalization_25_layer_call_fn_56355

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_52563?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_52267

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
7:????????? : : : : :*
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
:????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_21_layer_call_fn_55756

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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_51450?
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
6__inference_batch_normalization_20_layer_call_fn_55629

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
:?????????00*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_51695w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????00`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?F
?
H__inference_sequential_12_layer_call_and_return_conditional_losses_54922

inputsB
(conv2d_17_conv2d_readvariableop_resource:7
)conv2d_17_biasadd_readvariableop_resource:<
.batch_normalization_20_readvariableop_resource:>
0batch_normalization_20_readvariableop_1_resource:M
?batch_normalization_20_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_18_conv2d_readvariableop_resource:7
)conv2d_18_biasadd_readvariableop_resource:<
.batch_normalization_21_readvariableop_resource:>
0batch_normalization_21_readvariableop_1_resource:M
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:
identity??%batch_normalization_20/AssignNewValue?'batch_normalization_20/AssignNewValue_1?6batch_normalization_20/FusedBatchNormV3/ReadVariableOp?8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_20/ReadVariableOp?'batch_normalization_20/ReadVariableOp_1?%batch_normalization_21/AssignNewValue?'batch_normalization_21/AssignNewValue_1?6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_21/ReadVariableOp?'batch_normalization_21/ReadVariableOp_1? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00*
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00?
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_17/BiasAdd:output:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????00:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_20/AssignNewValueAssignVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource4batch_normalization_20/FusedBatchNormV3:batch_mean:07^batch_normalization_20/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_20/AssignNewValue_1AssignVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_20/FusedBatchNormV3:batch_variance:09^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_19/LeakyRelu	LeakyRelu+batch_normalization_20/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00*
alpha%???>?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_18/Conv2DConv2D&leaky_re_lu_19/LeakyRelu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_18/BiasAdd:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_20/LeakyRelu	LeakyRelu+batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>}
IdentityIdentity&leaky_re_lu_20/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_20/AssignNewValue(^batch_normalization_20/AssignNewValue_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_1&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????00: : : : : : : : : : : : 2N
%batch_normalization_20/AssignNewValue%batch_normalization_20/AssignNewValue2R
'batch_normalization_20/AssignNewValue_1'batch_normalization_20/AssignNewValue_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_24_layer_call_fn_56215

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_52530?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55341

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
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_51958

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
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_51063

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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55818

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
?
?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_52123

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
?
J
.__inference_leaky_re_lu_20_layer_call_fn_55859

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
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_51566h
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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55494

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
?3
?
B__inference_model_3_layer_call_and_return_conditional_losses_53835
input_3-
sequential_11_53728:!
sequential_11_53730:!
sequential_11_53732:!
sequential_11_53734:!
sequential_11_53736:!
sequential_11_53738:-
sequential_11_53740:!
sequential_11_53742:!
sequential_11_53744:!
sequential_11_53746:!
sequential_11_53748:!
sequential_11_53750:-
sequential_12_53753:!
sequential_12_53755:!
sequential_12_53757:!
sequential_12_53759:!
sequential_12_53761:!
sequential_12_53763:-
sequential_12_53765:!
sequential_12_53767:!
sequential_12_53769:!
sequential_12_53771:!
sequential_12_53773:!
sequential_12_53775:-
sequential_13_53778: !
sequential_13_53780: !
sequential_13_53782: !
sequential_13_53784: !
sequential_13_53786: !
sequential_13_53788: -
sequential_13_53790:  !
sequential_13_53792: !
sequential_13_53794: !
sequential_13_53796: !
sequential_13_53798: !
sequential_13_53800: -
sequential_14_53803: @!
sequential_14_53805:@!
sequential_14_53807:@!
sequential_14_53809:@!
sequential_14_53811:@!
sequential_14_53813:@-
sequential_14_53815:@@!
sequential_14_53817:@!
sequential_14_53819:@!
sequential_14_53821:@!
sequential_14_53823:@!
sequential_14_53825:@ 
latent_53829:
??
latent_53831:	?
identity??latent/StatefulPartitionedCall?%sequential_11/StatefulPartitionedCall?%sequential_12/StatefulPartitionedCall?%sequential_13/StatefulPartitionedCall?%sequential_14/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinput_3sequential_11_53728sequential_11_53730sequential_11_53732sequential_11_53734sequential_11_53736sequential_11_53738sequential_11_53740sequential_11_53742sequential_11_53744sequential_11_53746sequential_11_53748sequential_11_53750*
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_50997?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCall.sequential_11/StatefulPartitionedCall:output:0sequential_12_53753sequential_12_53755sequential_12_53757sequential_12_53759sequential_12_53761sequential_12_53763sequential_12_53765sequential_12_53767sequential_12_53769sequential_12_53771sequential_12_53773sequential_12_53775*
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_51569?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCall.sequential_12/StatefulPartitionedCall:output:0sequential_13_53778sequential_13_53780sequential_13_53782sequential_13_53784sequential_13_53786sequential_13_53788sequential_13_53790sequential_13_53792sequential_13_53794sequential_13_53796sequential_13_53798sequential_13_53800*
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_52141?
%sequential_14/StatefulPartitionedCallStatefulPartitionedCall.sequential_13/StatefulPartitionedCall:output:0sequential_14_53803sequential_14_53805sequential_14_53807sequential_14_53809sequential_14_53811sequential_14_53813sequential_14_53815sequential_14_53817sequential_14_53819sequential_14_53821sequential_14_53823sequential_14_53825*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_52713?
flatten_1/PartitionedCallPartitionedCall.sequential_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_53162?
latent/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0latent_53829latent_53831*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_53174w
IdentityIdentity'latent/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^latent/StatefulPartitionedCall&^sequential_11/StatefulPartitionedCall&^sequential_12/StatefulPartitionedCall&^sequential_13/StatefulPartitionedCall&^sequential_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
:?????????00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall2N
%sequential_14/StatefulPartitionedCall%sequential_14/StatefulPartitionedCall:X T
/
_output_shapes
:?????????00
!
_user_specified_name	input_3
?
?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_56007

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
7:????????? : : : : :*
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
:????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_38
serving_default_input_3:0?????????00;
latent1
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
"layer_with_weights-0
"layer-0
#layer_with_weights-1
#layer-1
$layer-2
%layer_with_weights-2
%layer-3
&layer_with_weights-3
&layer-4
'layer-5
(	variables
)trainable_variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
,layer_with_weights-0
,layer-0
-layer_with_weights-1
-layer-1
.layer-2
/layer_with_weights-2
/layer-3
0layer_with_weights-3
0layer-4
1layer-5
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15
P16
Q17
R18
S19
T20
U21
V22
W23
X24
Y25
Z26
[27
\28
]29
^30
_31
`32
a33
b34
c35
d36
e37
f38
g39
h40
i41
j42
k43
l44
m45
n46
o47
:48
;49"
trackable_list_wrapper
?
@0
A1
B2
C3
F4
G5
H6
I7
L8
M9
N10
O11
R12
S13
T14
U15
X16
Y17
Z18
[19
^20
_21
`22
a23
d24
e25
f26
g27
j28
k29
l30
m31
:32
;33"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
	trainable_variables

regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
?

@kernel
Abias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
yaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
z	variables
{trainable_variables
|regularization_losses
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
~	variables
trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Fkernel
Gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
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
v
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11"
trackable_list_wrapper
X
@0
A1
B2
C3
F4
G5
H6
I7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Lkernel
Mbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
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

Rkernel
Sbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
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
v
L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9
V10
W11"
trackable_list_wrapper
X
L0
M1
N2
O3
R4
S5
T6
U7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Xkernel
Ybias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Zgamma
[beta
\moving_mean
]moving_variance
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

^kernel
_bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	`gamma
abeta
bmoving_mean
cmoving_variance
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
v
X0
Y1
Z2
[3
\4
]5
^6
_7
`8
a9
b10
c11"
trackable_list_wrapper
X
X0
Y1
Z2
[3
^4
_5
`6
a7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
(	variables
)trainable_variables
*regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

dkernel
ebias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	fgamma
gbeta
hmoving_mean
imoving_variance
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

jkernel
kbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	lgamma
mbeta
nmoving_mean
omoving_variance
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
v
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11"
trackable_list_wrapper
X
d0
e1
f2
g3
j4
k5
l6
m7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
6	variables
7trainable_variables
8regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
??2latent/kernel
:?2latent/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_15/kernel
:2conv2d_15/bias
*:(2batch_normalization_18/gamma
):'2batch_normalization_18/beta
2:0 (2"batch_normalization_18/moving_mean
6:4 (2&batch_normalization_18/moving_variance
*:(2conv2d_16/kernel
:2conv2d_16/bias
*:(2batch_normalization_19/gamma
):'2batch_normalization_19/beta
2:0 (2"batch_normalization_19/moving_mean
6:4 (2&batch_normalization_19/moving_variance
*:(2conv2d_17/kernel
:2conv2d_17/bias
*:(2batch_normalization_20/gamma
):'2batch_normalization_20/beta
2:0 (2"batch_normalization_20/moving_mean
6:4 (2&batch_normalization_20/moving_variance
*:(2conv2d_18/kernel
:2conv2d_18/bias
*:(2batch_normalization_21/gamma
):'2batch_normalization_21/beta
2:0 (2"batch_normalization_21/moving_mean
6:4 (2&batch_normalization_21/moving_variance
*:( 2conv2d_19/kernel
: 2conv2d_19/bias
*:( 2batch_normalization_22/gamma
):' 2batch_normalization_22/beta
2:0  (2"batch_normalization_22/moving_mean
6:4  (2&batch_normalization_22/moving_variance
*:(  2conv2d_20/kernel
: 2conv2d_20/bias
*:( 2batch_normalization_23/gamma
):' 2batch_normalization_23/beta
2:0  (2"batch_normalization_23/moving_mean
6:4  (2&batch_normalization_23/moving_variance
*:( @2conv2d_21/kernel
:@2conv2d_21/bias
*:(@2batch_normalization_24/gamma
):'@2batch_normalization_24/beta
2:0@ (2"batch_normalization_24/moving_mean
6:4@ (2&batch_normalization_24/moving_variance
*:(@@2conv2d_22/kernel
:@2conv2d_22/bias
*:(@2batch_normalization_25/gamma
):'@2batch_normalization_25/beta
2:0@ (2"batch_normalization_25/moving_mean
6:4@ (2&batch_normalization_25/moving_variance
?
D0
E1
J2
K3
P4
Q5
V6
W7
\8
]9
b10
c11
h12
i13
n14
o15"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
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
~	variables
trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
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
H0
I1
J2
K3"
trackable_list_wrapper
.
H0
I1"
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
<
D0
E1
J2
K3"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
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
N0
O1
P2
Q3"
trackable_list_wrapper
.
N0
O1"
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
R0
S1"
trackable_list_wrapper
.
R0
S1"
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
T0
U1
V2
W3"
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
<
P0
Q1
V2
W3"
trackable_list_wrapper
J
0
1
2
3
4
5"
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
.
X0
Y1"
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
Z0
[1
\2
]3"
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
^0
_1"
trackable_list_wrapper
.
^0
_1"
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
`0
a1
b2
c3"
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
<
\0
]1
b2
c3"
trackable_list_wrapper
J
"0
#1
$2
%3
&4
'5"
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
.
d0
e1"
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
f0
g1
h2
i3"
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
j0
k1"
trackable_list_wrapper
.
j0
k1"
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
l0
m1
n2
o3"
trackable_list_wrapper
.
l0
m1"
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
<
h0
i1
n2
o3"
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
D0
E1"
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
J0
K1"
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
P0
Q1"
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
V0
W1"
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
\0
]1"
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
b0
c1"
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
h0
i1"
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
n0
o1"
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
'__inference_model_3_layer_call_fn_53284
'__inference_model_3_layer_call_fn_54157
'__inference_model_3_layer_call_fn_54262
'__inference_model_3_layer_call_fn_53725?
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
B__inference_model_3_layer_call_and_return_conditional_losses_54442
B__inference_model_3_layer_call_and_return_conditional_losses_54622
B__inference_model_3_layer_call_and_return_conditional_losses_53835
B__inference_model_3_layer_call_and_return_conditional_losses_53945?
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
 __inference__wrapped_model_50761input_3"?
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
?2?
-__inference_sequential_11_layer_call_fn_51024
-__inference_sequential_11_layer_call_fn_54651
-__inference_sequential_11_layer_call_fn_54680
-__inference_sequential_11_layer_call_fn_51265?
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
H__inference_sequential_11_layer_call_and_return_conditional_losses_54726
H__inference_sequential_11_layer_call_and_return_conditional_losses_54772
H__inference_sequential_11_layer_call_and_return_conditional_losses_51299
H__inference_sequential_11_layer_call_and_return_conditional_losses_51333?
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
-__inference_sequential_12_layer_call_fn_51596
-__inference_sequential_12_layer_call_fn_54801
-__inference_sequential_12_layer_call_fn_54830
-__inference_sequential_12_layer_call_fn_51837?
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
H__inference_sequential_12_layer_call_and_return_conditional_losses_54876
H__inference_sequential_12_layer_call_and_return_conditional_losses_54922
H__inference_sequential_12_layer_call_and_return_conditional_losses_51871
H__inference_sequential_12_layer_call_and_return_conditional_losses_51905?
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
-__inference_sequential_13_layer_call_fn_52168
-__inference_sequential_13_layer_call_fn_54951
-__inference_sequential_13_layer_call_fn_54980
-__inference_sequential_13_layer_call_fn_52409?
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_55026
H__inference_sequential_13_layer_call_and_return_conditional_losses_55072
H__inference_sequential_13_layer_call_and_return_conditional_losses_52443
H__inference_sequential_13_layer_call_and_return_conditional_losses_52477?
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
-__inference_sequential_14_layer_call_fn_52740
-__inference_sequential_14_layer_call_fn_55101
-__inference_sequential_14_layer_call_fn_55130
-__inference_sequential_14_layer_call_fn_52981?
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
H__inference_sequential_14_layer_call_and_return_conditional_losses_55176
H__inference_sequential_14_layer_call_and_return_conditional_losses_55222
H__inference_sequential_14_layer_call_and_return_conditional_losses_53015
H__inference_sequential_14_layer_call_and_return_conditional_losses_53049?
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
?2?
)__inference_flatten_1_layer_call_fn_55227?
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
D__inference_flatten_1_layer_call_and_return_conditional_losses_55233?
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
&__inference_latent_layer_call_fn_55242?
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
A__inference_latent_layer_call_and_return_conditional_losses_55252?
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
#__inference_signature_wrapper_54052input_3"?
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
)__inference_conv2d_15_layer_call_fn_55261?
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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_55271?
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
6__inference_batch_normalization_18_layer_call_fn_55284
6__inference_batch_normalization_18_layer_call_fn_55297
6__inference_batch_normalization_18_layer_call_fn_55310
6__inference_batch_normalization_18_layer_call_fn_55323?
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
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55341
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55359
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55377
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55395?
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
.__inference_leaky_re_lu_17_layer_call_fn_55400?
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
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_55405?
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
)__inference_conv2d_16_layer_call_fn_55414?
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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_55424?
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
6__inference_batch_normalization_19_layer_call_fn_55437
6__inference_batch_normalization_19_layer_call_fn_55450
6__inference_batch_normalization_19_layer_call_fn_55463
6__inference_batch_normalization_19_layer_call_fn_55476?
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
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55494
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55512
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55530
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55548?
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
.__inference_leaky_re_lu_18_layer_call_fn_55553?
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
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_55558?
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
)__inference_conv2d_17_layer_call_fn_55567?
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
D__inference_conv2d_17_layer_call_and_return_conditional_losses_55577?
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
6__inference_batch_normalization_20_layer_call_fn_55590
6__inference_batch_normalization_20_layer_call_fn_55603
6__inference_batch_normalization_20_layer_call_fn_55616
6__inference_batch_normalization_20_layer_call_fn_55629?
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
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55647
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55665
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55683
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55701?
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
.__inference_leaky_re_lu_19_layer_call_fn_55706?
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
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_55711?
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
)__inference_conv2d_18_layer_call_fn_55720?
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_55730?
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
6__inference_batch_normalization_21_layer_call_fn_55743
6__inference_batch_normalization_21_layer_call_fn_55756
6__inference_batch_normalization_21_layer_call_fn_55769
6__inference_batch_normalization_21_layer_call_fn_55782?
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
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55800
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55818
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55836
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55854?
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
.__inference_leaky_re_lu_20_layer_call_fn_55859?
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
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_55864?
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
)__inference_conv2d_19_layer_call_fn_55873?
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
D__inference_conv2d_19_layer_call_and_return_conditional_losses_55883?
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
6__inference_batch_normalization_22_layer_call_fn_55896
6__inference_batch_normalization_22_layer_call_fn_55909
6__inference_batch_normalization_22_layer_call_fn_55922
6__inference_batch_normalization_22_layer_call_fn_55935?
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
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55953
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55971
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55989
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_56007?
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
.__inference_leaky_re_lu_21_layer_call_fn_56012?
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
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_56017?
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
)__inference_conv2d_20_layer_call_fn_56026?
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
D__inference_conv2d_20_layer_call_and_return_conditional_losses_56036?
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
6__inference_batch_normalization_23_layer_call_fn_56049
6__inference_batch_normalization_23_layer_call_fn_56062
6__inference_batch_normalization_23_layer_call_fn_56075
6__inference_batch_normalization_23_layer_call_fn_56088?
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
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56106
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56124
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56142
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56160?
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
.__inference_leaky_re_lu_22_layer_call_fn_56165?
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
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_56170?
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
)__inference_conv2d_21_layer_call_fn_56179?
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
D__inference_conv2d_21_layer_call_and_return_conditional_losses_56189?
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
6__inference_batch_normalization_24_layer_call_fn_56202
6__inference_batch_normalization_24_layer_call_fn_56215
6__inference_batch_normalization_24_layer_call_fn_56228
6__inference_batch_normalization_24_layer_call_fn_56241?
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
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56259
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56277
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56295
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56313?
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
.__inference_leaky_re_lu_23_layer_call_fn_56318?
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
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_56323?
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
)__inference_conv2d_22_layer_call_fn_56332?
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
D__inference_conv2d_22_layer_call_and_return_conditional_losses_56342?
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
6__inference_batch_normalization_25_layer_call_fn_56355
6__inference_batch_normalization_25_layer_call_fn_56368
6__inference_batch_normalization_25_layer_call_fn_56381
6__inference_batch_normalization_25_layer_call_fn_56394?
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
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56412
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56430
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56448
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56466?
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
.__inference_leaky_re_lu_24_layer_call_fn_56471?
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
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_56476?
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
 __inference__wrapped_model_50761?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;8?5
.?+
)?&
input_3?????????00
? "0?-
+
latent!?
latent???????????
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55341?BCDEM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55359?BCDEM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55377rBCDE;?8
1?.
(?%
inputs?????????00
p 
? "-?*
#? 
0?????????00
? ?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_55395rBCDE;?8
1?.
(?%
inputs?????????00
p
? "-?*
#? 
0?????????00
? ?
6__inference_batch_normalization_18_layer_call_fn_55284?BCDEM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_18_layer_call_fn_55297?BCDEM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_18_layer_call_fn_55310eBCDE;?8
1?.
(?%
inputs?????????00
p 
? " ??????????00?
6__inference_batch_normalization_18_layer_call_fn_55323eBCDE;?8
1?.
(?%
inputs?????????00
p
? " ??????????00?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55494?HIJKM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55512?HIJKM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55530rHIJK;?8
1?.
(?%
inputs?????????00
p 
? "-?*
#? 
0?????????00
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_55548rHIJK;?8
1?.
(?%
inputs?????????00
p
? "-?*
#? 
0?????????00
? ?
6__inference_batch_normalization_19_layer_call_fn_55437?HIJKM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_19_layer_call_fn_55450?HIJKM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_19_layer_call_fn_55463eHIJK;?8
1?.
(?%
inputs?????????00
p 
? " ??????????00?
6__inference_batch_normalization_19_layer_call_fn_55476eHIJK;?8
1?.
(?%
inputs?????????00
p
? " ??????????00?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55647?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55665?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55683rNOPQ;?8
1?.
(?%
inputs?????????00
p 
? "-?*
#? 
0?????????00
? ?
Q__inference_batch_normalization_20_layer_call_and_return_conditional_losses_55701rNOPQ;?8
1?.
(?%
inputs?????????00
p
? "-?*
#? 
0?????????00
? ?
6__inference_batch_normalization_20_layer_call_fn_55590?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_20_layer_call_fn_55603?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_20_layer_call_fn_55616eNOPQ;?8
1?.
(?%
inputs?????????00
p 
? " ??????????00?
6__inference_batch_normalization_20_layer_call_fn_55629eNOPQ;?8
1?.
(?%
inputs?????????00
p
? " ??????????00?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55800?TUVWM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55818?TUVWM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55836rTUVW;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
Q__inference_batch_normalization_21_layer_call_and_return_conditional_losses_55854rTUVW;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
6__inference_batch_normalization_21_layer_call_fn_55743?TUVWM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_21_layer_call_fn_55756?TUVWM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
6__inference_batch_normalization_21_layer_call_fn_55769eTUVW;?8
1?.
(?%
inputs?????????
p 
? " ???????????
6__inference_batch_normalization_21_layer_call_fn_55782eTUVW;?8
1?.
(?%
inputs?????????
p
? " ???????????
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55953?Z[\]M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55971?Z[\]M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_55989rZ[\];?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
Q__inference_batch_normalization_22_layer_call_and_return_conditional_losses_56007rZ[\];?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
6__inference_batch_normalization_22_layer_call_fn_55896?Z[\]M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_22_layer_call_fn_55909?Z[\]M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_22_layer_call_fn_55922eZ[\];?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
6__inference_batch_normalization_22_layer_call_fn_55935eZ[\];?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56106?`abcM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56124?`abcM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56142r`abc;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
Q__inference_batch_normalization_23_layer_call_and_return_conditional_losses_56160r`abc;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
6__inference_batch_normalization_23_layer_call_fn_56049?`abcM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_23_layer_call_fn_56062?`abcM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_23_layer_call_fn_56075e`abc;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
6__inference_batch_normalization_23_layer_call_fn_56088e`abc;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56259?fghiM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56277?fghiM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56295rfghi;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
Q__inference_batch_normalization_24_layer_call_and_return_conditional_losses_56313rfghi;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
6__inference_batch_normalization_24_layer_call_fn_56202?fghiM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_24_layer_call_fn_56215?fghiM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_24_layer_call_fn_56228efghi;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
6__inference_batch_normalization_24_layer_call_fn_56241efghi;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56412?lmnoM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56430?lmnoM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56448rlmno;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
Q__inference_batch_normalization_25_layer_call_and_return_conditional_losses_56466rlmno;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
6__inference_batch_normalization_25_layer_call_fn_56355?lmnoM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_25_layer_call_fn_56368?lmnoM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_25_layer_call_fn_56381elmno;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
6__inference_batch_normalization_25_layer_call_fn_56394elmno;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_55271l@A7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
)__inference_conv2d_15_layer_call_fn_55261_@A7?4
-?*
(?%
inputs?????????00
? " ??????????00?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_55424lFG7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
)__inference_conv2d_16_layer_call_fn_55414_FG7?4
-?*
(?%
inputs?????????00
? " ??????????00?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_55577lLM7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
)__inference_conv2d_17_layer_call_fn_55567_LM7?4
-?*
(?%
inputs?????????00
? " ??????????00?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_55730lRS7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_18_layer_call_fn_55720_RS7?4
-?*
(?%
inputs?????????00
? " ???????????
D__inference_conv2d_19_layer_call_and_return_conditional_losses_55883lXY7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_19_layer_call_fn_55873_XY7?4
-?*
(?%
inputs?????????
? " ?????????? ?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_56036l^_7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_20_layer_call_fn_56026_^_7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
D__inference_conv2d_21_layer_call_and_return_conditional_losses_56189lde7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_21_layer_call_fn_56179_de7?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_56342ljk7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_22_layer_call_fn_56332_jk7?4
-?*
(?%
inputs?????????@
? " ??????????@?
D__inference_flatten_1_layer_call_and_return_conditional_losses_55233a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
)__inference_flatten_1_layer_call_fn_55227T7?4
-?*
(?%
inputs?????????@
? "????????????
A__inference_latent_layer_call_and_return_conditional_losses_55252^:;0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_latent_layer_call_fn_55242Q:;0?-
&?#
!?
inputs??????????
? "????????????
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_55405h7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
.__inference_leaky_re_lu_17_layer_call_fn_55400[7?4
-?*
(?%
inputs?????????00
? " ??????????00?
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_55558h7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
.__inference_leaky_re_lu_18_layer_call_fn_55553[7?4
-?*
(?%
inputs?????????00
? " ??????????00?
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_55711h7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00
? ?
.__inference_leaky_re_lu_19_layer_call_fn_55706[7?4
-?*
(?%
inputs?????????00
? " ??????????00?
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_55864h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
.__inference_leaky_re_lu_20_layer_call_fn_55859[7?4
-?*
(?%
inputs?????????
? " ???????????
I__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_56017h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_leaky_re_lu_21_layer_call_fn_56012[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
I__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_56170h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_leaky_re_lu_22_layer_call_fn_56165[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
I__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_56323h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
.__inference_leaky_re_lu_23_layer_call_fn_56318[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
I__inference_leaky_re_lu_24_layer_call_and_return_conditional_losses_56476h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
.__inference_leaky_re_lu_24_layer_call_fn_56471[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
B__inference_model_3_layer_call_and_return_conditional_losses_53835?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;@?=
6?3
)?&
input_3?????????00
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_53945?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;@?=
6?3
)?&
input_3?????????00
p

 
? "&?#
?
0??????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_54442?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;??<
5?2
(?%
inputs?????????00
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_54622?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;??<
5?2
(?%
inputs?????????00
p

 
? "&?#
?
0??????????
? ?
'__inference_model_3_layer_call_fn_53284?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;@?=
6?3
)?&
input_3?????????00
p 

 
? "????????????
'__inference_model_3_layer_call_fn_53725?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;@?=
6?3
)?&
input_3?????????00
p

 
? "????????????
'__inference_model_3_layer_call_fn_54157?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;??<
5?2
(?%
inputs?????????00
p 

 
? "????????????
'__inference_model_3_layer_call_fn_54262?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;??<
5?2
(?%
inputs?????????00
p

 
? "????????????
H__inference_sequential_11_layer_call_and_return_conditional_losses_51299?@ABCDEFGHIJKH?E
>?;
1?.
conv2d_15_input?????????00
p 

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_11_layer_call_and_return_conditional_losses_51333?@ABCDEFGHIJKH?E
>?;
1?.
conv2d_15_input?????????00
p

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_11_layer_call_and_return_conditional_losses_54726~@ABCDEFGHIJK??<
5?2
(?%
inputs?????????00
p 

 
? "-?*
#? 
0?????????00
? ?
H__inference_sequential_11_layer_call_and_return_conditional_losses_54772~@ABCDEFGHIJK??<
5?2
(?%
inputs?????????00
p

 
? "-?*
#? 
0?????????00
? ?
-__inference_sequential_11_layer_call_fn_51024z@ABCDEFGHIJKH?E
>?;
1?.
conv2d_15_input?????????00
p 

 
? " ??????????00?
-__inference_sequential_11_layer_call_fn_51265z@ABCDEFGHIJKH?E
>?;
1?.
conv2d_15_input?????????00
p

 
? " ??????????00?
-__inference_sequential_11_layer_call_fn_54651q@ABCDEFGHIJK??<
5?2
(?%
inputs?????????00
p 

 
? " ??????????00?
-__inference_sequential_11_layer_call_fn_54680q@ABCDEFGHIJK??<
5?2
(?%
inputs?????????00
p

 
? " ??????????00?
H__inference_sequential_12_layer_call_and_return_conditional_losses_51871?LMNOPQRSTUVWH?E
>?;
1?.
conv2d_17_input?????????00
p 

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_12_layer_call_and_return_conditional_losses_51905?LMNOPQRSTUVWH?E
>?;
1?.
conv2d_17_input?????????00
p

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_12_layer_call_and_return_conditional_losses_54876~LMNOPQRSTUVW??<
5?2
(?%
inputs?????????00
p 

 
? "-?*
#? 
0?????????
? ?
H__inference_sequential_12_layer_call_and_return_conditional_losses_54922~LMNOPQRSTUVW??<
5?2
(?%
inputs?????????00
p

 
? "-?*
#? 
0?????????
? ?
-__inference_sequential_12_layer_call_fn_51596zLMNOPQRSTUVWH?E
>?;
1?.
conv2d_17_input?????????00
p 

 
? " ???????????
-__inference_sequential_12_layer_call_fn_51837zLMNOPQRSTUVWH?E
>?;
1?.
conv2d_17_input?????????00
p

 
? " ???????????
-__inference_sequential_12_layer_call_fn_54801qLMNOPQRSTUVW??<
5?2
(?%
inputs?????????00
p 

 
? " ???????????
-__inference_sequential_12_layer_call_fn_54830qLMNOPQRSTUVW??<
5?2
(?%
inputs?????????00
p

 
? " ???????????
H__inference_sequential_13_layer_call_and_return_conditional_losses_52443?XYZ[\]^_`abcH?E
>?;
1?.
conv2d_19_input?????????
p 

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_13_layer_call_and_return_conditional_losses_52477?XYZ[\]^_`abcH?E
>?;
1?.
conv2d_19_input?????????
p

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_13_layer_call_and_return_conditional_losses_55026~XYZ[\]^_`abc??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0????????? 
? ?
H__inference_sequential_13_layer_call_and_return_conditional_losses_55072~XYZ[\]^_`abc??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0????????? 
? ?
-__inference_sequential_13_layer_call_fn_52168zXYZ[\]^_`abcH?E
>?;
1?.
conv2d_19_input?????????
p 

 
? " ?????????? ?
-__inference_sequential_13_layer_call_fn_52409zXYZ[\]^_`abcH?E
>?;
1?.
conv2d_19_input?????????
p

 
? " ?????????? ?
-__inference_sequential_13_layer_call_fn_54951qXYZ[\]^_`abc??<
5?2
(?%
inputs?????????
p 

 
? " ?????????? ?
-__inference_sequential_13_layer_call_fn_54980qXYZ[\]^_`abc??<
5?2
(?%
inputs?????????
p

 
? " ?????????? ?
H__inference_sequential_14_layer_call_and_return_conditional_losses_53015?defghijklmnoH?E
>?;
1?.
conv2d_21_input????????? 
p 

 
? "-?*
#? 
0?????????@
? ?
H__inference_sequential_14_layer_call_and_return_conditional_losses_53049?defghijklmnoH?E
>?;
1?.
conv2d_21_input????????? 
p

 
? "-?*
#? 
0?????????@
? ?
H__inference_sequential_14_layer_call_and_return_conditional_losses_55176~defghijklmno??<
5?2
(?%
inputs????????? 
p 

 
? "-?*
#? 
0?????????@
? ?
H__inference_sequential_14_layer_call_and_return_conditional_losses_55222~defghijklmno??<
5?2
(?%
inputs????????? 
p

 
? "-?*
#? 
0?????????@
? ?
-__inference_sequential_14_layer_call_fn_52740zdefghijklmnoH?E
>?;
1?.
conv2d_21_input????????? 
p 

 
? " ??????????@?
-__inference_sequential_14_layer_call_fn_52981zdefghijklmnoH?E
>?;
1?.
conv2d_21_input????????? 
p

 
? " ??????????@?
-__inference_sequential_14_layer_call_fn_55101qdefghijklmno??<
5?2
(?%
inputs????????? 
p 

 
? " ??????????@?
-__inference_sequential_14_layer_call_fn_55130qdefghijklmno??<
5?2
(?%
inputs????????? 
p

 
? " ??????????@?
#__inference_signature_wrapper_54052?2@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmno:;C?@
? 
9?6
4
input_3)?&
input_3?????????00"0?-
+
latent!?
latent??????????