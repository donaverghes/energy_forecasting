Ó
š
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
ě
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.8.02v1.8.0-0-g93bc2e20728öŽ

global_step/Initializer/zerosConst*
dtype0	*
_output_shapes
: *
value	B	 R *
_class
loc:@global_step
k
global_step
VariableV2*
_class
loc:@global_step*
shape: *
dtype0	*
_output_shapes
: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_class
loc:@global_step*
_output_shapes
: *
T0	
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
shape:˙˙˙˙˙˙˙˙˙1
Y
ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
k

ExpandDims
ExpandDimsPlaceholderExpandDims/dim*+
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
T0
g
SqueezeSqueeze
ExpandDims*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
squeeze_dims

b
Reshape/shapeConst*
dtype0*
_output_shapes
:*!
valueB"˙˙˙˙1      
`
ReshapeReshapeSqueezeReshape/shape*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙1
Ľ
.conv1d/kernel/Initializer/random_uniform/shapeConst*!
valueB"         * 
_class
loc:@conv1d/kernel*
dtype0*
_output_shapes
:

,conv1d/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *SŻž* 
_class
loc:@conv1d/kernel*
dtype0

,conv1d/kernel/Initializer/random_uniform/maxConst*
valueB
 *SŻ>* 
_class
loc:@conv1d/kernel*
dtype0*
_output_shapes
: 
Ó
6conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv1d/kernel/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv1d/kernel*
dtype0*"
_output_shapes
:
Ň
,conv1d/kernel/Initializer/random_uniform/subSub,conv1d/kernel/Initializer/random_uniform/max,conv1d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv1d/kernel*
_output_shapes
: *
T0
č
,conv1d/kernel/Initializer/random_uniform/mulMul6conv1d/kernel/Initializer/random_uniform/RandomUniform,conv1d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:
Ú
(conv1d/kernel/Initializer/random_uniformAdd,conv1d/kernel/Initializer/random_uniform/mul,conv1d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:

conv1d/kernel
VariableV2* 
_class
loc:@conv1d/kernel*
shape:*
dtype0*"
_output_shapes
:
Ś
conv1d/kernel/AssignAssignconv1d/kernel(conv1d/kernel/Initializer/random_uniform*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:
|
conv1d/kernel/readIdentityconv1d/kernel* 
_class
loc:@conv1d/kernel*"
_output_shapes
:*
T0

conv1d/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@conv1d/bias*
dtype0*
_output_shapes
:
s
conv1d/bias
VariableV2*
_output_shapes
:*
_class
loc:@conv1d/bias*
shape:*
dtype0

conv1d/bias/AssignAssignconv1d/biasconv1d/bias/Initializer/zeros*
T0*
_class
loc:@conv1d/bias*
_output_shapes
:
n
conv1d/bias/readIdentityconv1d/bias*
T0*
_class
loc:@conv1d/bias*
_output_shapes
:
^
conv1d/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
^
conv1d/conv1d/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :

conv1d/conv1d/ExpandDims
ExpandDimsReshapeconv1d/conv1d/ExpandDims/dim*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙1
`
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
value	B : *
dtype0

conv1d/conv1d/ExpandDims_1
ExpandDimsconv1d/kernel/readconv1d/conv1d/ExpandDims_1/dim*&
_output_shapes
:*
T0
ś
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/ExpandDimsconv1d/conv1d/ExpandDims_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
T0*
strides
*
paddingSAME

conv1d/conv1d/SqueezeSqueezeconv1d/conv1d/Conv2D*+
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
squeeze_dims
*
T0
x
conv1d/BiasAddBiasAddconv1d/conv1d/Squeezeconv1d/bias/read*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙1
Y
conv1d/ReluReluconv1d/BiasAdd*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙1
^
max_pooling1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 

max_pooling1d/ExpandDims
ExpandDimsconv1d/Relumax_pooling1d/ExpandDims/dim*/
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
T0
§
max_pooling1d/MaxPoolMaxPoolmax_pooling1d/ExpandDims*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
ksize
*
paddingVALID

max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
Š
0conv1d_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*!
valueB"         *"
_class
loc:@conv1d_1/kernel*
dtype0

.conv1d_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *  ž*"
_class
loc:@conv1d_1/kernel*
dtype0

.conv1d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *  >*"
_class
loc:@conv1d_1/kernel*
dtype0*
_output_shapes
: 
Ů
8conv1d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv1d_1/kernel/Initializer/random_uniform/shape*
dtype0*"
_output_shapes
:*
T0*"
_class
loc:@conv1d_1/kernel
Ú
.conv1d_1/kernel/Initializer/random_uniform/subSub.conv1d_1/kernel/Initializer/random_uniform/max.conv1d_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv1d_1/kernel
đ
.conv1d_1/kernel/Initializer/random_uniform/mulMul8conv1d_1/kernel/Initializer/random_uniform/RandomUniform.conv1d_1/kernel/Initializer/random_uniform/sub*"
_output_shapes
:*
T0*"
_class
loc:@conv1d_1/kernel
â
*conv1d_1/kernel/Initializer/random_uniformAdd.conv1d_1/kernel/Initializer/random_uniform/mul.conv1d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:*
T0

conv1d_1/kernel
VariableV2*"
_class
loc:@conv1d_1/kernel*
shape:*
dtype0*"
_output_shapes
:
Ž
conv1d_1/kernel/AssignAssignconv1d_1/kernel*conv1d_1/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:

conv1d_1/kernel/readIdentityconv1d_1/kernel*"
_output_shapes
:*
T0*"
_class
loc:@conv1d_1/kernel

conv1d_1/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@conv1d_1/bias*
dtype0*
_output_shapes
:
w
conv1d_1/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:* 
_class
loc:@conv1d_1/bias

conv1d_1/bias/AssignAssignconv1d_1/biasconv1d_1/bias/Initializer/zeros*
T0* 
_class
loc:@conv1d_1/bias*
_output_shapes
:
t
conv1d_1/bias/readIdentityconv1d_1/bias*
_output_shapes
:*
T0* 
_class
loc:@conv1d_1/bias
`
conv1d_1/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
`
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0

conv1d_1/conv1d/ExpandDims
ExpandDimsmax_pooling1d/Squeezeconv1d_1/conv1d/ExpandDims/dim*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
 conv1d_1/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 

conv1d_1/conv1d/ExpandDims_1
ExpandDimsconv1d_1/kernel/read conv1d_1/conv1d/ExpandDims_1/dim*
T0*&
_output_shapes
:
ź
conv1d_1/conv1d/Conv2DConv2Dconv1d_1/conv1d/ExpandDimsconv1d_1/conv1d/ExpandDims_1*
T0*
strides
*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙

conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
conv1d_1/BiasAddBiasAddconv1d_1/conv1d/Squeezeconv1d_1/bias/read*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
conv1d_1/ReluReluconv1d_1/BiasAdd*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
max_pooling1d_1/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 

max_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relumax_pooling1d_1/ExpandDims/dim*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
max_pooling1d_1/MaxPoolMaxPoolmax_pooling1d_1/ExpandDims*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
ksize
*
paddingVALID

max_pooling1d_1/SqueezeSqueezemax_pooling1d_1/MaxPool*
squeeze_dims
*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"˙˙˙˙Ŕ   
q
	Reshape_1Reshapemax_pooling1d_1/SqueezeReshape_1/shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ*
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"Ŕ      *
_class
loc:@dense/kernel*
dtype0

+dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *3ž*
_class
loc:@dense/kernel*
dtype0

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *3>*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Í
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	Ŕ
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Ŕ
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
:	Ŕ*
T0

dense/kernel
VariableV2*
_output_shapes
:	Ŕ*
_class
loc:@dense/kernel*
shape:	Ŕ*
dtype0

dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_output_shapes
:	Ŕ*
T0*
_class
loc:@dense/kernel
v
dense/kernel/readIdentitydense/kernel*
_output_shapes
:	Ŕ*
T0*
_class
loc:@dense/kernel

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
q

dense/bias
VariableV2*
_output_shapes
:*
_class
loc:@dense/bias*
shape:*
dtype0

dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
_class
loc:@dense/bias*
_output_shapes
:
k
dense/bias/readIdentity
dense/bias*
_output_shapes
:*
T0*
_class
loc:@dense/bias
f
dense/MatMulMatMul	Reshape_1dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
S

dense/ReluReludense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *!
_class
loc:@dense_1/kernel*
dtype0

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄż*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *qÄ?*!
_class
loc:@dense_1/kernel
Ň
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
č
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
Ú
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

dense_1/kernel
VariableV2*
shape
:*
dtype0*
_output_shapes

:*!
_class
loc:@dense_1/kernel
Ś
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

dense_1/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
_class
loc:@dense_1/bias*
dtype0
u
dense_1/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
_class
loc:@dense_1/bias

dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
:*
T0
k
dense_1/MatMulMatMul
dense/Reludense_1/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_cccb2c1c210d4aea90b61fe1d4be645e/part*
dtype0
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ň
save/SaveV2/tensor_namesConst"/device:CPU:0*
valueB	Bconv1d/biasBconv1d/kernelBconv1d_1/biasBconv1d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:	

save/SaveV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv1d/biasconv1d/kernelconv1d_1/biasconv1d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step"/device:CPU:0*
dtypes
2		
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
ő
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueB	Bconv1d/biasBconv1d/kernelBconv1d_1/biasBconv1d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:	

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
Ç
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		
w
save/AssignAssignconv1d/biassave/RestoreV2*
T0*
_class
loc:@conv1d/bias*
_output_shapes
:

save/Assign_1Assignconv1d/kernelsave/RestoreV2:1*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:

save/Assign_2Assignconv1d_1/biassave/RestoreV2:2*
T0* 
_class
loc:@conv1d_1/bias*
_output_shapes
:

save/Assign_3Assignconv1d_1/kernelsave/RestoreV2:3*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:
y
save/Assign_4Assign
dense/biassave/RestoreV2:4*
_output_shapes
:*
T0*
_class
loc:@dense/bias

save/Assign_5Assigndense/kernelsave/RestoreV2:5*
_class
loc:@dense/kernel*
_output_shapes
:	Ŕ*
T0
}
save/Assign_6Assigndense_1/biassave/RestoreV2:6*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

save/Assign_7Assigndense_1/kernelsave/RestoreV2:7*!
_class
loc:@dense_1/kernel*
_output_shapes

:*
T0
w
save/Assign_8Assignglobal_stepsave/RestoreV2:8*
_output_shapes
: *
T0	*
_class
loc:@global_step
¨
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_8662a1f18cb54364a8b454e0cff46ffa/part
j
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
ô
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
valueB	Bconv1d/biasBconv1d/kernelBconv1d_1/biasBconv1d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0

save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	

save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesconv1d/biasconv1d/kernelconv1d_1/biasconv1d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step"/device:CPU:0*
dtypes
2		
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: *
T0
Ś
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
{
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
÷
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
valueB	Bconv1d/biasBconv1d/kernelBconv1d_1/biasBconv1d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:	

!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
Ď
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2		*8
_output_shapes&
$:::::::::
{
save_1/AssignAssignconv1d/biassave_1/RestoreV2*
T0*
_class
loc:@conv1d/bias*
_output_shapes
:

save_1/Assign_1Assignconv1d/kernelsave_1/RestoreV2:1*"
_output_shapes
:*
T0* 
_class
loc:@conv1d/kernel

save_1/Assign_2Assignconv1d_1/biassave_1/RestoreV2:2* 
_class
loc:@conv1d_1/bias*
_output_shapes
:*
T0

save_1/Assign_3Assignconv1d_1/kernelsave_1/RestoreV2:3*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:
}
save_1/Assign_4Assign
dense/biassave_1/RestoreV2:4*
T0*
_class
loc:@dense/bias*
_output_shapes
:

save_1/Assign_5Assigndense/kernelsave_1/RestoreV2:5*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Ŕ

save_1/Assign_6Assigndense_1/biassave_1/RestoreV2:6*
_output_shapes
:*
T0*
_class
loc:@dense_1/bias

save_1/Assign_7Assigndense_1/kernelsave_1/RestoreV2:7*
_output_shapes

:*
T0*!
_class
loc:@dense_1/kernel
{
save_1/Assign_8Assignglobal_stepsave_1/RestoreV2:8*
_class
loc:@global_step*
_output_shapes
: *
T0	
ź
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"ż
trainable_variables§¤
i
conv1d/kernel:0conv1d/kernel/Assignconv1d/kernel/read:02*conv1d/kernel/Initializer/random_uniform:0
X
conv1d/bias:0conv1d/bias/Assignconv1d/bias/read:02conv1d/bias/Initializer/zeros:0
q
conv1d_1/kernel:0conv1d_1/kernel/Assignconv1d_1/kernel/read:02,conv1d_1/kernel/Initializer/random_uniform:0
`
conv1d_1/bias:0conv1d_1/bias/Assignconv1d_1/bias/read:02!conv1d_1/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"
	variablesţ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
i
conv1d/kernel:0conv1d/kernel/Assignconv1d/kernel/read:02*conv1d/kernel/Initializer/random_uniform:0
X
conv1d/bias:0conv1d/bias/Assignconv1d/bias/read:02conv1d/bias/Initializer/zeros:0
q
conv1d_1/kernel:0conv1d_1/kernel/Assignconv1d_1/kernel/read:02,conv1d_1/kernel/Initializer/random_uniform:0
`
conv1d_1/bias:0conv1d_1/bias/Assignconv1d_1/bias/read:02!conv1d_1/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0" 
legacy_init_op


group_deps*
serving_default
-
price$
Placeholder:0˙˙˙˙˙˙˙˙˙15
	predicted(
dense_1/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict*
predictions
-
price$
Placeholder:0˙˙˙˙˙˙˙˙˙15
	predicted(
dense_1/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict