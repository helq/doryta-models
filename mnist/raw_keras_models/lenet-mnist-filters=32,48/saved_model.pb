??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8??
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:0*
dtype0
h

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	x*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?	x*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:x*
dtype0
h

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:xT*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:T*
dtype0
h

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Td*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:Td*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:d*
dtype0
h

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_4
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
p

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*
shared_name
Variable_5
i
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes

:d
*
dtype0
n
Adadelta/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdadelta/iter
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
_output_shapes
: *
dtype0	
p
Adadelta/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/decay
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
_output_shapes
: *
dtype0
?
Adadelta/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdadelta/learning_rate
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
_output_shapes
: *
dtype0
l
Adadelta/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/rho
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
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
!Adadelta/conv2d/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d/kernel/accum_grad
?
5Adadelta/conv2d/kernel/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d/kernel/accum_grad*&
_output_shapes
: *
dtype0
?
Adadelta/conv2d/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adadelta/conv2d/bias/accum_grad
?
3Adadelta/conv2d/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/conv2d/bias/accum_grad*
_output_shapes
: *
dtype0
?
Adadelta/Variable/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdadelta/Variable/accum_grad
?
0Adadelta/Variable/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad*
_output_shapes
: *
dtype0
?
#Adadelta/conv2d_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*4
shared_name%#Adadelta/conv2d_1/kernel/accum_grad
?
7Adadelta/conv2d_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_1/kernel/accum_grad*&
_output_shapes
: 0*
dtype0
?
!Adadelta/conv2d_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adadelta/conv2d_1/bias/accum_grad
?
5Adadelta/conv2d_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_1/bias/accum_grad*
_output_shapes
:0*
dtype0
?
Adadelta/Variable/accum_grad_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_1
?
2Adadelta/Variable/accum_grad_1/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_1*
_output_shapes
: *
dtype0
?
 Adadelta/dense/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	x*1
shared_name" Adadelta/dense/kernel/accum_grad
?
4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense/kernel/accum_grad*
_output_shapes
:	?	x*
dtype0
?
Adadelta/dense/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*/
shared_name Adadelta/dense/bias/accum_grad
?
2Adadelta/dense/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_grad*
_output_shapes
:x*
dtype0
?
Adadelta/Variable/accum_grad_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_2
?
2Adadelta/Variable/accum_grad_2/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_2*
_output_shapes
: *
dtype0
?
"Adadelta/dense_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*3
shared_name$"Adadelta/dense_1/kernel/accum_grad
?
6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_1/kernel/accum_grad*
_output_shapes

:xT*
dtype0
?
 Adadelta/dense_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*1
shared_name" Adadelta/dense_1/bias/accum_grad
?
4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_1/bias/accum_grad*
_output_shapes
:T*
dtype0
?
Adadelta/Variable/accum_grad_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_3
?
2Adadelta/Variable/accum_grad_3/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_3*
_output_shapes
: *
dtype0
?
"Adadelta/dense_2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Td*3
shared_name$"Adadelta/dense_2/kernel/accum_grad
?
6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_2/kernel/accum_grad*
_output_shapes

:Td*
dtype0
?
 Adadelta/dense_2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*1
shared_name" Adadelta/dense_2/bias/accum_grad
?
4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_2/bias/accum_grad*
_output_shapes
:d*
dtype0
?
Adadelta/Variable/accum_grad_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_4
?
2Adadelta/Variable/accum_grad_4/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_4*
_output_shapes
: *
dtype0
?
 Adadelta/conv2d/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adadelta/conv2d/kernel/accum_var
?
4Adadelta/conv2d/kernel/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/conv2d/kernel/accum_var*&
_output_shapes
: *
dtype0
?
Adadelta/conv2d/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/conv2d/bias/accum_var
?
2Adadelta/conv2d/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv2d/bias/accum_var*
_output_shapes
: *
dtype0
?
Adadelta/Variable/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdadelta/Variable/accum_var
?
/Adadelta/Variable/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var*
_output_shapes
: *
dtype0
?
"Adadelta/conv2d_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*3
shared_name$"Adadelta/conv2d_1/kernel/accum_var
?
6Adadelta/conv2d_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_1/kernel/accum_var*&
_output_shapes
: 0*
dtype0
?
 Adadelta/conv2d_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Adadelta/conv2d_1/bias/accum_var
?
4Adadelta/conv2d_1/bias/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/conv2d_1/bias/accum_var*
_output_shapes
:0*
dtype0
?
Adadelta/Variable/accum_var_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_1
?
1Adadelta/Variable/accum_var_1/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_1*
_output_shapes
: *
dtype0
?
Adadelta/dense/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	x*0
shared_name!Adadelta/dense/kernel/accum_var
?
3Adadelta/dense/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/kernel/accum_var*
_output_shapes
:	?	x*
dtype0
?
Adadelta/dense/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*.
shared_nameAdadelta/dense/bias/accum_var
?
1Adadelta/dense/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_var*
_output_shapes
:x*
dtype0
?
Adadelta/Variable/accum_var_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_2
?
1Adadelta/Variable/accum_var_2/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_2*
_output_shapes
: *
dtype0
?
!Adadelta/dense_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*2
shared_name#!Adadelta/dense_1/kernel/accum_var
?
5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_1/kernel/accum_var*
_output_shapes

:xT*
dtype0
?
Adadelta/dense_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*0
shared_name!Adadelta/dense_1/bias/accum_var
?
3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_1/bias/accum_var*
_output_shapes
:T*
dtype0
?
Adadelta/Variable/accum_var_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_3
?
1Adadelta/Variable/accum_var_3/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_3*
_output_shapes
: *
dtype0
?
!Adadelta/dense_2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Td*2
shared_name#!Adadelta/dense_2/kernel/accum_var
?
5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_2/kernel/accum_var*
_output_shapes

:Td*
dtype0
?
Adadelta/dense_2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adadelta/dense_2/bias/accum_var
?
3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_2/bias/accum_var*
_output_shapes
:d*
dtype0
?
Adadelta/Variable/accum_var_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_4
?
1Adadelta/Variable/accum_var_4/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_4*
_output_shapes
: *
dtype0

NoOpNoOp
?d
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?d
value?dB?d B?d
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
a
	sharpness
	variables
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
a
*	sharpness
+	variables
,regularization_losses
-trainable_variables
.	keras_api
R
/	variables
0regularization_losses
1trainable_variables
2	keras_api
R
3	variables
4regularization_losses
5trainable_variables
6	keras_api
h

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
a
=	sharpness
>	variables
?regularization_losses
@trainable_variables
A	keras_api
h

Bkernel
Cbias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
a
H	sharpness
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
h

Mkernel
Nbias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
a
S	sharpness
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
e
X_rescaled_key
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
?
]iter
	^decay
_learning_rate
`rho
accum_grad?
accum_grad?
accum_grad?$
accum_grad?%
accum_grad?*
accum_grad?7
accum_grad?8
accum_grad?=
accum_grad?B
accum_grad?C
accum_grad?H
accum_grad?M
accum_grad?N
accum_grad?S
accum_grad?	accum_var?	accum_var?	accum_var?$	accum_var?%	accum_var?*	accum_var?7	accum_var?8	accum_var?=	accum_var?B	accum_var?C	accum_var?H	accum_var?M	accum_var?N	accum_var?S	accum_var?
v
0
1
2
$3
%4
*5
76
87
=8
B9
C10
H11
M12
N13
S14
X15
 
n
0
1
2
$3
%4
*5
76
87
=8
B9
C10
H11
M12
N13
S14
?

alayers
bmetrics
cnon_trainable_variables
dlayer_metrics
	variables
regularization_losses
trainable_variables
elayer_regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

flayers
gmetrics
hnon_trainable_variables
ilayer_metrics
	variables
regularization_losses
trainable_variables
jlayer_regularization_losses
WU
VARIABLE_VALUEVariable9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?

klayers
lmetrics
mnon_trainable_variables
nlayer_metrics
	variables
regularization_losses
trainable_variables
olayer_regularization_losses
 
 
 
?

players
qmetrics
rnon_trainable_variables
slayer_metrics
 	variables
!regularization_losses
"trainable_variables
tlayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
?

ulayers
vmetrics
wnon_trainable_variables
xlayer_metrics
&	variables
'regularization_losses
(trainable_variables
ylayer_regularization_losses
YW
VARIABLE_VALUE
Variable_19layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUE

*0
 

*0
?

zlayers
{metrics
|non_trainable_variables
}layer_metrics
+	variables
,regularization_losses
-trainable_variables
~layer_regularization_losses
 
 
 
?

layers
?metrics
?non_trainable_variables
?layer_metrics
/	variables
0regularization_losses
1trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
3	variables
4regularization_losses
5trainable_variables
 ?layer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
9	variables
:regularization_losses
;trainable_variables
 ?layer_regularization_losses
YW
VARIABLE_VALUE
Variable_29layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUE

=0
 

=0
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
>	variables
?regularization_losses
@trainable_variables
 ?layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
D	variables
Eregularization_losses
Ftrainable_variables
 ?layer_regularization_losses
YW
VARIABLE_VALUE
Variable_39layer_with_weights-7/sharpness/.ATTRIBUTES/VARIABLE_VALUE

H0
 

H0
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
I	variables
Jregularization_losses
Ktrainable_variables
 ?layer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
 

M0
N1
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
O	variables
Pregularization_losses
Qtrainable_variables
 ?layer_regularization_losses
YW
VARIABLE_VALUE
Variable_49layer_with_weights-9/sharpness/.ATTRIBUTES/VARIABLE_VALUE

S0
 

S0
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
T	variables
Uregularization_losses
Vtrainable_variables
 ?layer_regularization_losses
^\
VARIABLE_VALUE
Variable_5>layer_with_weights-10/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUE

X0
 
 
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
Y	variables
Zregularization_losses
[trainable_variables
 ?layer_regularization_losses
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
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
9
10
11
12
13

?0
?1

X0
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
 
 
 
 
 

X0
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE!Adadelta/conv2d/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv2d/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_grad^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adadelta/conv2d_1/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/conv2d_1/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_grad_1^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/dense/kernel/accum_grad[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense/bias/accum_gradYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_grad_2^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_1/kernel/accum_grad[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/dense_1/bias/accum_gradYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_grad_3^layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_2/kernel/accum_grad[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/dense_2/bias/accum_gradYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_grad_4^layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/conv2d/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv2d/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_var]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/conv2d_1/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/conv2d_1/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_var_1]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense/kernel/accum_varZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense/bias/accum_varXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_var_2]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/dense_1/kernel/accum_varZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense_1/bias/accum_varXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_var_3]layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/dense_2/kernel/accum_varZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense_2/bias/accum_varXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/Variable/accum_var_4]layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasVariableconv2d_1/kernelconv2d_1/bias
Variable_1dense/kernel
dense/bias
Variable_2dense_1/kerneldense_1/bias
Variable_3dense_2/kerneldense_2/bias
Variable_4
Variable_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_45543
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOpVariable/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOpVariable_1/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpVariable_2/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpVariable_3/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_5/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp5Adadelta/conv2d/kernel/accum_grad/Read/ReadVariableOp3Adadelta/conv2d/bias/accum_grad/Read/ReadVariableOp0Adadelta/Variable/accum_grad/Read/ReadVariableOp7Adadelta/conv2d_1/kernel/accum_grad/Read/ReadVariableOp5Adadelta/conv2d_1/bias/accum_grad/Read/ReadVariableOp2Adadelta/Variable/accum_grad_1/Read/ReadVariableOp4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOp2Adadelta/dense/bias/accum_grad/Read/ReadVariableOp2Adadelta/Variable/accum_grad_2/Read/ReadVariableOp6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOp2Adadelta/Variable/accum_grad_3/Read/ReadVariableOp6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOp2Adadelta/Variable/accum_grad_4/Read/ReadVariableOp4Adadelta/conv2d/kernel/accum_var/Read/ReadVariableOp2Adadelta/conv2d/bias/accum_var/Read/ReadVariableOp/Adadelta/Variable/accum_var/Read/ReadVariableOp6Adadelta/conv2d_1/kernel/accum_var/Read/ReadVariableOp4Adadelta/conv2d_1/bias/accum_var/Read/ReadVariableOp1Adadelta/Variable/accum_var_1/Read/ReadVariableOp3Adadelta/dense/kernel/accum_var/Read/ReadVariableOp1Adadelta/dense/bias/accum_var/Read/ReadVariableOp1Adadelta/Variable/accum_var_2/Read/ReadVariableOp5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOp1Adadelta/Variable/accum_var_3/Read/ReadVariableOp5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOp1Adadelta/Variable/accum_var_4/Read/ReadVariableOpConst*C
Tin<
:28	*
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
__inference__traced_save_46649
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasVariableconv2d_1/kernelconv2d_1/bias
Variable_1dense/kernel
dense/bias
Variable_2dense_1/kerneldense_1/bias
Variable_3dense_2/kerneldense_2/bias
Variable_4
Variable_5Adadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcounttotal_1count_1!Adadelta/conv2d/kernel/accum_gradAdadelta/conv2d/bias/accum_gradAdadelta/Variable/accum_grad#Adadelta/conv2d_1/kernel/accum_grad!Adadelta/conv2d_1/bias/accum_gradAdadelta/Variable/accum_grad_1 Adadelta/dense/kernel/accum_gradAdadelta/dense/bias/accum_gradAdadelta/Variable/accum_grad_2"Adadelta/dense_1/kernel/accum_grad Adadelta/dense_1/bias/accum_gradAdadelta/Variable/accum_grad_3"Adadelta/dense_2/kernel/accum_grad Adadelta/dense_2/bias/accum_gradAdadelta/Variable/accum_grad_4 Adadelta/conv2d/kernel/accum_varAdadelta/conv2d/bias/accum_varAdadelta/Variable/accum_var"Adadelta/conv2d_1/kernel/accum_var Adadelta/conv2d_1/bias/accum_varAdadelta/Variable/accum_var_1Adadelta/dense/kernel/accum_varAdadelta/dense/bias/accum_varAdadelta/Variable/accum_var_2!Adadelta/dense_1/kernel/accum_varAdadelta/dense_1/bias/accum_varAdadelta/Variable/accum_var_3!Adadelta/dense_2/kernel/accum_varAdadelta/dense_2/bias/accum_varAdadelta/Variable/accum_var_4*B
Tin;
927*
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
!__inference__traced_restore_46821??
?
?
!spiking_b_relu_1_cond_false_45839%
!spiking_b_relu_1_cond_placeholderA
=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value"
spiking_b_relu_1_cond_identity?
spiking_b_relu_1/cond/IdentityIdentity=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:?????????

02 
spiking_b_relu_1/cond/Identity"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_46321

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????T2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
R
cond_true_46193
cond_identity_cast
cond_placeholder
cond_identityx
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:?????????

02
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?
R
cond_true_45036
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:?????????d2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
?
spiking_b_relu_cond_true_455754
0spiking_b_relu_cond_identity_spiking_b_relu_cast#
spiking_b_relu_cond_placeholder 
spiking_b_relu_cond_identity?
spiking_b_relu/cond/IdentityIdentity0spiking_b_relu_cond_identity_spiking_b_relu_cast*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/cond/Identity"E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?
?
'__inference_dense_2_layer_call_fn_46397

inputs
unknown:Td
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_450042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????T: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_45380
conv2d_input!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	?	x
	unknown_6:x
	unknown_7: 
	unknown_8:xT
	unknown_9:T

unknown_10: 

unknown_11:Td

unknown_12:d

unknown_13: 

unknown_14:d

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_453082
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
spiking_b_relu_cond_false_45797#
spiking_b_relu_cond_placeholder=
9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value 
spiking_b_relu_cond_identity?
spiking_b_relu/cond/IdentityIdentity9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/cond/Identity"E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?
?
!spiking_b_relu_3_cond_false_45703%
!spiking_b_relu_3_cond_placeholderA
=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value"
spiking_b_relu_3_cond_identity?
spiking_b_relu_3/cond/IdentityIdentity=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:?????????T2 
spiking_b_relu_3/cond/Identity"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44801

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_46119

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y?
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
GreaterEqualo
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/yi
sub_1Subinputssub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
sub_1c
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:????????? 2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yl
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:????????? 2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *#
else_branchR
cond_false_46107*.
output_shapes
:????????? *"
then_branchR
cond_true_461062
conds
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:????????? 2
cond/Identityy
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? : 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
|
0__inference_spiking_b_relu_4_layer_call_fn_46445

inputs
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_450492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?"
!__inference__traced_restore_46821
file_prefix8
assignvariableop_conv2d_kernel: ,
assignvariableop_1_conv2d_bias: %
assignvariableop_2_variable: <
"assignvariableop_3_conv2d_1_kernel: 0.
 assignvariableop_4_conv2d_1_bias:0'
assignvariableop_5_variable_1: 2
assignvariableop_6_dense_kernel:	?	x+
assignvariableop_7_dense_bias:x'
assignvariableop_8_variable_2: 3
!assignvariableop_9_dense_1_kernel:xT.
 assignvariableop_10_dense_1_bias:T(
assignvariableop_11_variable_3: 4
"assignvariableop_12_dense_2_kernel:Td.
 assignvariableop_13_dense_2_bias:d(
assignvariableop_14_variable_4: 0
assignvariableop_15_variable_5:d
+
!assignvariableop_16_adadelta_iter:	 ,
"assignvariableop_17_adadelta_decay: 4
*assignvariableop_18_adadelta_learning_rate: *
 assignvariableop_19_adadelta_rho: #
assignvariableop_20_total: #
assignvariableop_21_count: %
assignvariableop_22_total_1: %
assignvariableop_23_count_1: O
5assignvariableop_24_adadelta_conv2d_kernel_accum_grad: A
3assignvariableop_25_adadelta_conv2d_bias_accum_grad: :
0assignvariableop_26_adadelta_variable_accum_grad: Q
7assignvariableop_27_adadelta_conv2d_1_kernel_accum_grad: 0C
5assignvariableop_28_adadelta_conv2d_1_bias_accum_grad:0<
2assignvariableop_29_adadelta_variable_accum_grad_1: G
4assignvariableop_30_adadelta_dense_kernel_accum_grad:	?	x@
2assignvariableop_31_adadelta_dense_bias_accum_grad:x<
2assignvariableop_32_adadelta_variable_accum_grad_2: H
6assignvariableop_33_adadelta_dense_1_kernel_accum_grad:xTB
4assignvariableop_34_adadelta_dense_1_bias_accum_grad:T<
2assignvariableop_35_adadelta_variable_accum_grad_3: H
6assignvariableop_36_adadelta_dense_2_kernel_accum_grad:TdB
4assignvariableop_37_adadelta_dense_2_bias_accum_grad:d<
2assignvariableop_38_adadelta_variable_accum_grad_4: N
4assignvariableop_39_adadelta_conv2d_kernel_accum_var: @
2assignvariableop_40_adadelta_conv2d_bias_accum_var: 9
/assignvariableop_41_adadelta_variable_accum_var: P
6assignvariableop_42_adadelta_conv2d_1_kernel_accum_var: 0B
4assignvariableop_43_adadelta_conv2d_1_bias_accum_var:0;
1assignvariableop_44_adadelta_variable_accum_var_1: F
3assignvariableop_45_adadelta_dense_kernel_accum_var:	?	x?
1assignvariableop_46_adadelta_dense_bias_accum_var:x;
1assignvariableop_47_adadelta_variable_accum_var_2: G
5assignvariableop_48_adadelta_dense_1_kernel_accum_var:xTA
3assignvariableop_49_adadelta_dense_1_bias_accum_var:T;
1assignvariableop_50_adadelta_variable_accum_var_3: G
5assignvariableop_51_adadelta_dense_2_kernel_accum_var:TdA
3assignvariableop_52_adadelta_dense_2_bias_accum_var:d;
1assignvariableop_53_adadelta_variable_accum_var_4: 
identity_55??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*? 
value? B? 7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-7/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-9/sharpness/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_variableIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_3Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_4Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_5Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_adadelta_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_adadelta_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adadelta_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp assignvariableop_19_adadelta_rhoIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adadelta_conv2d_kernel_accum_gradIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adadelta_conv2d_bias_accum_gradIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adadelta_variable_accum_gradIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adadelta_conv2d_1_kernel_accum_gradIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adadelta_conv2d_1_bias_accum_gradIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adadelta_variable_accum_grad_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adadelta_dense_kernel_accum_gradIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adadelta_dense_bias_accum_gradIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adadelta_variable_accum_grad_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adadelta_dense_1_kernel_accum_gradIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adadelta_dense_1_bias_accum_gradIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adadelta_variable_accum_grad_3Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adadelta_dense_2_kernel_accum_gradIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adadelta_dense_2_bias_accum_gradIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adadelta_variable_accum_grad_4Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adadelta_conv2d_kernel_accum_varIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adadelta_conv2d_bias_accum_varIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp/assignvariableop_41_adadelta_variable_accum_varIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adadelta_conv2d_1_kernel_accum_varIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adadelta_conv2d_1_bias_accum_varIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp1assignvariableop_44_adadelta_variable_accum_var_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adadelta_dense_kernel_accum_varIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp1assignvariableop_46_adadelta_dense_bias_accum_varIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp1assignvariableop_47_adadelta_variable_accum_var_2Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adadelta_dense_1_kernel_accum_varIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adadelta_dense_1_bias_accum_varIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp1assignvariableop_50_adadelta_variable_accum_var_3Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adadelta_dense_2_kernel_accum_varIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adadelta_dense_2_bias_accum_varIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp1assignvariableop_53_adadelta_variable_accum_var_4Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54f
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_55?	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_55Identity_55:output:0*?
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_46438

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/ya
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:?????????d2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:?????????d2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yd
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:?????????d2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????d2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????d2
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_46426*&
output_shapes
:?????????d*"
then_branchR
cond_true_464252
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:?????????d2
cond/Identityq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
R
cond_true_44845
cond_identity_cast
cond_placeholder
cond_identityx
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:?????????

02
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?
?
 spiking_b_relu_1_cond_true_456178
4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast%
!spiking_b_relu_1_cond_placeholder"
spiking_b_relu_1_cond_identity?
spiking_b_relu_1/cond/IdentityIdentity4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast*
T0*/
_output_shapes
:?????????

02 
spiking_b_relu_1/cond/Identity"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?
?
!spiking_b_relu_4_cond_false_45744%
!spiking_b_relu_4_cond_placeholderA
=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value"
spiking_b_relu_4_cond_identity?
spiking_b_relu_4/cond/IdentityIdentity=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:?????????d2 
spiking_b_relu_4/cond/Identity"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
\
cond_false_44846
cond_placeholder
cond_identity_clip_by_value
cond_identity?
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:?????????

02
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?
?
'__inference_dense_1_layer_call_fn_46330

inputs
unknown:xT
	unknown_0:T
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_449452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
R
cond_true_44977
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:?????????T2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
\
cond_false_44978
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:?????????T2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?

?
@__inference_dense_layer_call_and_return_conditional_losses_46254

inputs1
matmul_readvariableop_resource:	?	x-
biasadd_readvariableop_resource:x
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????x2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
 spiking_b_relu_4_cond_true_459648
4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast%
!spiking_b_relu_4_cond_placeholder"
spiking_b_relu_4_cond_identity?
spiking_b_relu_4/cond/IdentityIdentity4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast*
T0*'
_output_shapes
:?????????d2 
spiking_b_relu_4/cond/Identity"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
I
-__inference_max_pooling2d_layer_call_fn_46146

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
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_448012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_46141

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_446962
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_45543
conv2d_input!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	?	x
	unknown_6:x
	unknown_7: 
	unknown_8:xT
	unknown_9:T

unknown_10: 

unknown_11:Td

unknown_12:d

unknown_13: 

unknown_14:d

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_446872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
spiking_b_relu_cond_false_45576#
spiking_b_relu_cond_placeholder=
9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value 
spiking_b_relu_cond_identity?
spiking_b_relu/cond/IdentityIdentity9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/cond/Identity"E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?
|
0__inference_spiking_b_relu_2_layer_call_fn_46311

inputs
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_449312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????x2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_45985

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 0
&spiking_b_relu_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: 06
(conv2d_1_biasadd_readvariableop_resource:02
(spiking_b_relu_1_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	?	x3
%dense_biasadd_readvariableop_resource:x2
(spiking_b_relu_2_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:xT5
'dense_1_biasadd_readvariableop_resource:T2
(spiking_b_relu_3_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:Td5
'dense_2_biasadd_readvariableop_resource:d2
(spiking_b_relu_4_readvariableop_resource: @
.softmax__decode_matmul_readvariableop_resource:d

identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?%softmax__decode/MatMul/ReadVariableOp?#spiking_b_relu/Equal/ReadVariableOp?spiking_b_relu/ReadVariableOp?%spiking_b_relu_1/Equal/ReadVariableOp?spiking_b_relu_1/ReadVariableOp?%spiking_b_relu_2/Equal/ReadVariableOp?spiking_b_relu_2/ReadVariableOp?%spiking_b_relu_3/Equal/ReadVariableOp?spiking_b_relu_3/ReadVariableOp?%spiking_b_relu_4/Equal/ReadVariableOp?spiking_b_relu_4/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdd?
spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu/GreaterEqual/y?
spiking_b_relu/GreaterEqualGreaterEqualconv2d/BiasAdd:output:0&spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/GreaterEqual?
spiking_b_relu/CastCastspiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
spiking_b_relu/Cast?
spiking_b_relu/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype02
spiking_b_relu/ReadVariableOpq
spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu/sub/x?
spiking_b_relu/subSubspiking_b_relu/sub/x:output:0%spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu/subq
spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu/add/y?
spiking_b_relu/addAddV2spiking_b_relu/sub:z:0spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu/addy
spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu/truediv/x?
spiking_b_relu/truedivRealDiv!spiking_b_relu/truediv/x:output:0spiking_b_relu/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu/truedivu
spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu/sub_1/y?
spiking_b_relu/sub_1Subconv2d/BiasAdd:output:0spiking_b_relu/sub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/sub_1?
spiking_b_relu/mulMulspiking_b_relu/truediv:z:0spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/mulu
spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu/add_1/y?
spiking_b_relu/add_1AddV2spiking_b_relu/mul:z:0spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/add_1?
&spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&spiking_b_relu/clip_by_value/Minimum/y?
$spiking_b_relu/clip_by_value/MinimumMinimumspiking_b_relu/add_1:z:0/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2&
$spiking_b_relu/clip_by_value/Minimum?
spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
spiking_b_relu/clip_by_value/y?
spiking_b_relu/clip_by_valueMaximum(spiking_b_relu/clip_by_value/Minimum:z:0'spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/clip_by_value?
#spiking_b_relu/Equal/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype02%
#spiking_b_relu/Equal/ReadVariableOpu
spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu/Equal/y?
spiking_b_relu/EqualEqual+spiking_b_relu/Equal/ReadVariableOp:value:0spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu/Equal?
spiking_b_relu/condStatelessIfspiking_b_relu/Equal:z:0spiking_b_relu/Cast:y:0 spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
else_branch#R!
spiking_b_relu_cond_false_45797*.
output_shapes
:????????? *1
then_branch"R 
spiking_b_relu_cond_true_457962
spiking_b_relu/cond?
spiking_b_relu/cond/IdentityIdentityspiking_b_relu/cond:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/cond/Identity?
max_pooling2d/MaxPoolMaxPool%spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

0*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

02
conv2d_1/BiasAdd?
spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spiking_b_relu_1/GreaterEqual/y?
spiking_b_relu_1/GreaterEqualGreaterEqualconv2d_1/BiasAdd:output:0(spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????

02
spiking_b_relu_1/GreaterEqual?
spiking_b_relu_1/CastCast!spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

02
spiking_b_relu_1/Cast?
spiking_b_relu_1/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype02!
spiking_b_relu_1/ReadVariableOpu
spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_1/sub/x?
spiking_b_relu_1/subSubspiking_b_relu_1/sub/x:output:0'spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu_1/subu
spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu_1/add/y?
spiking_b_relu_1/addAddV2spiking_b_relu_1/sub:z:0spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_1/add}
spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_1/truediv/x?
spiking_b_relu_1/truedivRealDiv#spiking_b_relu_1/truediv/x:output:0spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu_1/truedivy
spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_1/sub_1/y?
spiking_b_relu_1/sub_1Subconv2d_1/BiasAdd:output:0!spiking_b_relu_1/sub_1/y:output:0*
T0*/
_output_shapes
:?????????

02
spiking_b_relu_1/sub_1?
spiking_b_relu_1/mulMulspiking_b_relu_1/truediv:z:0spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:?????????

02
spiking_b_relu_1/muly
spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_1/add_1/y?
spiking_b_relu_1/add_1AddV2spiking_b_relu_1/mul:z:0!spiking_b_relu_1/add_1/y:output:0*
T0*/
_output_shapes
:?????????

02
spiking_b_relu_1/add_1?
(spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(spiking_b_relu_1/clip_by_value/Minimum/y?
&spiking_b_relu_1/clip_by_value/MinimumMinimumspiking_b_relu_1/add_1:z:01spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????

02(
&spiking_b_relu_1/clip_by_value/Minimum?
 spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 spiking_b_relu_1/clip_by_value/y?
spiking_b_relu_1/clip_by_valueMaximum*spiking_b_relu_1/clip_by_value/Minimum:z:0)spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????

02 
spiking_b_relu_1/clip_by_value?
%spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype02'
%spiking_b_relu_1/Equal/ReadVariableOpy
spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_1/Equal/y?
spiking_b_relu_1/EqualEqual-spiking_b_relu_1/Equal/ReadVariableOp:value:0!spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_1/Equal?
spiking_b_relu_1/condStatelessIfspiking_b_relu_1/Equal:z:0spiking_b_relu_1/Cast:y:0"spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????

0* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_1_cond_false_45839*.
output_shapes
:?????????

0*3
then_branch$R"
 spiking_b_relu_1_cond_true_458382
spiking_b_relu_1/cond?
spiking_b_relu_1/cond/IdentityIdentityspiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:?????????

02 
spiking_b_relu_1/cond/Identity?
max_pooling2d_1/MaxPoolMaxPool'spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense/BiasAdd?
spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spiking_b_relu_2/GreaterEqual/y?
spiking_b_relu_2/GreaterEqualGreaterEqualdense/BiasAdd:output:0(spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????x2
spiking_b_relu_2/GreaterEqual?
spiking_b_relu_2/CastCast!spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????x2
spiking_b_relu_2/Cast?
spiking_b_relu_2/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype02!
spiking_b_relu_2/ReadVariableOpu
spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_2/sub/x?
spiking_b_relu_2/subSubspiking_b_relu_2/sub/x:output:0'spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu_2/subu
spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu_2/add/y?
spiking_b_relu_2/addAddV2spiking_b_relu_2/sub:z:0spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_2/add}
spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_2/truediv/x?
spiking_b_relu_2/truedivRealDiv#spiking_b_relu_2/truediv/x:output:0spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu_2/truedivy
spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_2/sub_1/y?
spiking_b_relu_2/sub_1Subdense/BiasAdd:output:0!spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:?????????x2
spiking_b_relu_2/sub_1?
spiking_b_relu_2/mulMulspiking_b_relu_2/truediv:z:0spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:?????????x2
spiking_b_relu_2/muly
spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_2/add_1/y?
spiking_b_relu_2/add_1AddV2spiking_b_relu_2/mul:z:0!spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:?????????x2
spiking_b_relu_2/add_1?
(spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(spiking_b_relu_2/clip_by_value/Minimum/y?
&spiking_b_relu_2/clip_by_value/MinimumMinimumspiking_b_relu_2/add_1:z:01spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????x2(
&spiking_b_relu_2/clip_by_value/Minimum?
 spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 spiking_b_relu_2/clip_by_value/y?
spiking_b_relu_2/clip_by_valueMaximum*spiking_b_relu_2/clip_by_value/Minimum:z:0)spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????x2 
spiking_b_relu_2/clip_by_value?
%spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype02'
%spiking_b_relu_2/Equal/ReadVariableOpy
spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_2/Equal/y?
spiking_b_relu_2/EqualEqual-spiking_b_relu_2/Equal/ReadVariableOp:value:0!spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_2/Equal?
spiking_b_relu_2/condStatelessIfspiking_b_relu_2/Equal:z:0spiking_b_relu_2/Cast:y:0"spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_2_cond_false_45883*&
output_shapes
:?????????x*3
then_branch$R"
 spiking_b_relu_2_cond_true_458822
spiking_b_relu_2/cond?
spiking_b_relu_2/cond/IdentityIdentityspiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:?????????x2 
spiking_b_relu_2/cond/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul'spiking_b_relu_2/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_1/BiasAdd?
spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spiking_b_relu_3/GreaterEqual/y?
spiking_b_relu_3/GreaterEqualGreaterEqualdense_1/BiasAdd:output:0(spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T2
spiking_b_relu_3/GreaterEqual?
spiking_b_relu_3/CastCast!spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T2
spiking_b_relu_3/Cast?
spiking_b_relu_3/ReadVariableOpReadVariableOp(spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype02!
spiking_b_relu_3/ReadVariableOpu
spiking_b_relu_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_3/sub/x?
spiking_b_relu_3/subSubspiking_b_relu_3/sub/x:output:0'spiking_b_relu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu_3/subu
spiking_b_relu_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu_3/add/y?
spiking_b_relu_3/addAddV2spiking_b_relu_3/sub:z:0spiking_b_relu_3/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_3/add}
spiking_b_relu_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_3/truediv/x?
spiking_b_relu_3/truedivRealDiv#spiking_b_relu_3/truediv/x:output:0spiking_b_relu_3/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu_3/truedivy
spiking_b_relu_3/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_3/sub_1/y?
spiking_b_relu_3/sub_1Subdense_1/BiasAdd:output:0!spiking_b_relu_3/sub_1/y:output:0*
T0*'
_output_shapes
:?????????T2
spiking_b_relu_3/sub_1?
spiking_b_relu_3/mulMulspiking_b_relu_3/truediv:z:0spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:?????????T2
spiking_b_relu_3/muly
spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_3/add_1/y?
spiking_b_relu_3/add_1AddV2spiking_b_relu_3/mul:z:0!spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:?????????T2
spiking_b_relu_3/add_1?
(spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(spiking_b_relu_3/clip_by_value/Minimum/y?
&spiking_b_relu_3/clip_by_value/MinimumMinimumspiking_b_relu_3/add_1:z:01spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????T2(
&spiking_b_relu_3/clip_by_value/Minimum?
 spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 spiking_b_relu_3/clip_by_value/y?
spiking_b_relu_3/clip_by_valueMaximum*spiking_b_relu_3/clip_by_value/Minimum:z:0)spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????T2 
spiking_b_relu_3/clip_by_value?
%spiking_b_relu_3/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype02'
%spiking_b_relu_3/Equal/ReadVariableOpy
spiking_b_relu_3/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_3/Equal/y?
spiking_b_relu_3/EqualEqual-spiking_b_relu_3/Equal/ReadVariableOp:value:0!spiking_b_relu_3/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_3/Equal?
spiking_b_relu_3/condStatelessIfspiking_b_relu_3/Equal:z:0spiking_b_relu_3/Cast:y:0"spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_3_cond_false_45924*&
output_shapes
:?????????T*3
then_branch$R"
 spiking_b_relu_3_cond_true_459232
spiking_b_relu_3/cond?
spiking_b_relu_3/cond/IdentityIdentityspiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:?????????T2 
spiking_b_relu_3/cond/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul'spiking_b_relu_3/cond/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_2/BiasAdd?
spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spiking_b_relu_4/GreaterEqual/y?
spiking_b_relu_4/GreaterEqualGreaterEqualdense_2/BiasAdd:output:0(spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
spiking_b_relu_4/GreaterEqual?
spiking_b_relu_4/CastCast!spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
spiking_b_relu_4/Cast?
spiking_b_relu_4/ReadVariableOpReadVariableOp(spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype02!
spiking_b_relu_4/ReadVariableOpu
spiking_b_relu_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_4/sub/x?
spiking_b_relu_4/subSubspiking_b_relu_4/sub/x:output:0'spiking_b_relu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu_4/subu
spiking_b_relu_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu_4/add/y?
spiking_b_relu_4/addAddV2spiking_b_relu_4/sub:z:0spiking_b_relu_4/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_4/add}
spiking_b_relu_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_4/truediv/x?
spiking_b_relu_4/truedivRealDiv#spiking_b_relu_4/truediv/x:output:0spiking_b_relu_4/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu_4/truedivy
spiking_b_relu_4/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_4/sub_1/y?
spiking_b_relu_4/sub_1Subdense_2/BiasAdd:output:0!spiking_b_relu_4/sub_1/y:output:0*
T0*'
_output_shapes
:?????????d2
spiking_b_relu_4/sub_1?
spiking_b_relu_4/mulMulspiking_b_relu_4/truediv:z:0spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:?????????d2
spiking_b_relu_4/muly
spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_4/add_1/y?
spiking_b_relu_4/add_1AddV2spiking_b_relu_4/mul:z:0!spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:?????????d2
spiking_b_relu_4/add_1?
(spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(spiking_b_relu_4/clip_by_value/Minimum/y?
&spiking_b_relu_4/clip_by_value/MinimumMinimumspiking_b_relu_4/add_1:z:01spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????d2(
&spiking_b_relu_4/clip_by_value/Minimum?
 spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 spiking_b_relu_4/clip_by_value/y?
spiking_b_relu_4/clip_by_valueMaximum*spiking_b_relu_4/clip_by_value/Minimum:z:0)spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????d2 
spiking_b_relu_4/clip_by_value?
%spiking_b_relu_4/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype02'
%spiking_b_relu_4/Equal/ReadVariableOpy
spiking_b_relu_4/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_4/Equal/y?
spiking_b_relu_4/EqualEqual-spiking_b_relu_4/Equal/ReadVariableOp:value:0!spiking_b_relu_4/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_4/Equal?
spiking_b_relu_4/condStatelessIfspiking_b_relu_4/Equal:z:0spiking_b_relu_4/Cast:y:0"spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_4_cond_false_45965*&
output_shapes
:?????????d*3
then_branch$R"
 spiking_b_relu_4_cond_true_459642
spiking_b_relu_4/cond?
spiking_b_relu_4/cond/IdentityIdentityspiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:?????????d2 
spiking_b_relu_4/cond/Identitys
softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
softmax__decode/mul/x?
softmax__decode/mulMulsoftmax__decode/mul/x:output:0'spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:?????????d2
softmax__decode/muls
softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
softmax__decode/sub/y?
softmax__decode/subSubsoftmax__decode/mul:z:0softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:?????????d2
softmax__decode/sub?
%softmax__decode/MatMul/ReadVariableOpReadVariableOp.softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02'
%softmax__decode/MatMul/ReadVariableOp?
softmax__decode/MatMulMatMulsoftmax__decode/sub:z:0-softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
softmax__decode/MatMul?
softmax__decode/SoftmaxSoftmax softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:?????????
2
softmax__decode/Softmax|
IdentityIdentity!softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^softmax__decode/MatMul/ReadVariableOp$^spiking_b_relu/Equal/ReadVariableOp^spiking_b_relu/ReadVariableOp&^spiking_b_relu_1/Equal/ReadVariableOp ^spiking_b_relu_1/ReadVariableOp&^spiking_b_relu_2/Equal/ReadVariableOp ^spiking_b_relu_2/ReadVariableOp&^spiking_b_relu_3/Equal/ReadVariableOp ^spiking_b_relu_3/ReadVariableOp&^spiking_b_relu_4/Equal/ReadVariableOp ^spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2N
%softmax__decode/MatMul/ReadVariableOp%softmax__decode/MatMul/ReadVariableOp2J
#spiking_b_relu/Equal/ReadVariableOp#spiking_b_relu/Equal/ReadVariableOp2>
spiking_b_relu/ReadVariableOpspiking_b_relu/ReadVariableOp2N
%spiking_b_relu_1/Equal/ReadVariableOp%spiking_b_relu_1/Equal/ReadVariableOp2B
spiking_b_relu_1/ReadVariableOpspiking_b_relu_1/ReadVariableOp2N
%spiking_b_relu_2/Equal/ReadVariableOp%spiking_b_relu_2/Equal/ReadVariableOp2B
spiking_b_relu_2/ReadVariableOpspiking_b_relu_2/ReadVariableOp2N
%spiking_b_relu_3/Equal/ReadVariableOp%spiking_b_relu_3/Equal/ReadVariableOp2B
spiking_b_relu_3/ReadVariableOpspiking_b_relu_3/ReadVariableOp2N
%spiking_b_relu_4/Equal/ReadVariableOp%spiking_b_relu_4/Equal/ReadVariableOp2B
spiking_b_relu_4/ReadVariableOpspiking_b_relu_4/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
cond_false_46107
cond_placeholder
cond_identity_clip_by_value
cond_identity?
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:????????? 2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?
C
'__inference_flatten_layer_call_fn_46244

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
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_448742
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_46304

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????x2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????x2
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/ya
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:?????????x2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:?????????x2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yd
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:?????????x2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????x2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????x2
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_46292*&
output_shapes
:?????????x*"
then_branchR
cond_true_462912
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:?????????x2
cond/Identityq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:?????????x2

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
\
cond_false_46426
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:?????????d2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
?
,sequential_spiking_b_relu_2_cond_false_445850
,sequential_spiking_b_relu_2_cond_placeholderW
Ssequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_clip_by_value-
)sequential_spiking_b_relu_2_cond_identity?
)sequential/spiking_b_relu_2/cond/IdentityIdentitySsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:?????????x2+
)sequential/spiking_b_relu_2/cond/Identity"_
)sequential_spiking_b_relu_2_cond_identity2sequential/spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?
|
0__inference_spiking_b_relu_1_layer_call_fn_46213

inputs
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_448582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????

02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????

0: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????

0
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_44696

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_46165

inputs!
unknown: 0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_448132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????

02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_45105
conv2d_input!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	?	x
	unknown_6:x
	unknown_7: 
	unknown_8:xT
	unknown_9:T

unknown_10: 

unknown_11:Td

unknown_12:d

unknown_13: 

unknown_14:d

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_450702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
!spiking_b_relu_1_cond_false_45618%
!spiking_b_relu_1_cond_placeholderA
=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value"
spiking_b_relu_1_cond_identity?
spiking_b_relu_1/cond/IdentityIdentity=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:?????????

02 
spiking_b_relu_1/cond/Identity"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?E
?
E__inference_sequential_layer_call_and_return_conditional_losses_45070

inputs&
conv2d_44749: 
conv2d_44751: 
spiking_b_relu_44794: (
conv2d_1_44814: 0
conv2d_1_44816:0 
spiking_b_relu_1_44859: 
dense_44887:	?	x
dense_44889:x 
spiking_b_relu_2_44932: 
dense_1_44946:xT
dense_1_44948:T 
spiking_b_relu_3_44991: 
dense_2_45005:Td
dense_2_45007:d 
spiking_b_relu_4_45050: '
softmax__decode_45066:d

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?'softmax__decode/StatefulPartitionedCall?&spiking_b_relu/StatefulPartitionedCall?(spiking_b_relu_1/StatefulPartitionedCall?(spiking_b_relu_2/StatefulPartitionedCall?(spiking_b_relu_3/StatefulPartitionedCall?(spiking_b_relu_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_44749conv2d_44751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_447482 
conv2d/StatefulPartitionedCall?
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0spiking_b_relu_44794*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_447932(
&spiking_b_relu/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_448012
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_44814conv2d_1_44816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_448132"
 conv2d_1/StatefulPartitionedCall?
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0spiking_b_relu_1_44859*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_448582*
(spiking_b_relu_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_448662!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_448742
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_44887dense_44889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_448862
dense/StatefulPartitionedCall?
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_2_44932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_449312*
(spiking_b_relu_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0dense_1_44946dense_1_44948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_449452!
dense_1/StatefulPartitionedCall?
(spiking_b_relu_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_3_44991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_449902*
(spiking_b_relu_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_3/StatefulPartitionedCall:output:0dense_2_45005dense_2_45007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_450042!
dense_2/StatefulPartitionedCall?
(spiking_b_relu_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_4_45050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_450492*
(spiking_b_relu_4/StatefulPartitionedCall?
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_4/StatefulPartitionedCall:output:0softmax__decode_45066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_450652)
'softmax__decode/StatefulPartitionedCall?
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall2T
(spiking_b_relu_3/StatefulPartitionedCall(spiking_b_relu_3/StatefulPartitionedCall2T
(spiking_b_relu_4/StatefulPartitionedCall(spiking_b_relu_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_44886

inputs1
matmul_readvariableop_resource:	?	x-
biasadd_readvariableop_resource:x
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????x2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_44945

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????T2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
,sequential_spiking_b_relu_3_cond_false_446260
,sequential_spiking_b_relu_3_cond_placeholderW
Ssequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_clip_by_value-
)sequential_spiking_b_relu_3_cond_identity?
)sequential/spiking_b_relu_3/cond/IdentityIdentitySsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:?????????T2+
)sequential/spiking_b_relu_3/cond/Identity"_
)sequential_spiking_b_relu_3_cond_identity2sequential/spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
\
cond_false_44919
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:?????????x2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_45004

inputs0
matmul_readvariableop_resource:Td-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Td*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_46233

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
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_448662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

0:W S
/
_output_shapes
:?????????

0
 
_user_specified_nameinputs
?
R
cond_true_46425
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:?????????d2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_44874

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
 spiking_b_relu_3_cond_true_459238
4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast%
!spiking_b_relu_3_cond_placeholder"
spiking_b_relu_3_cond_identity?
spiking_b_relu_3/cond/IdentityIdentity4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast*
T0*'
_output_shapes
:?????????T2 
spiking_b_relu_3/cond/Identity"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
?
 spiking_b_relu_2_cond_true_456618
4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast%
!spiking_b_relu_2_cond_placeholder"
spiking_b_relu_2_cond_identity?
spiking_b_relu_2/cond/IdentityIdentity4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast*
T0*'
_output_shapes
:?????????x2 
spiking_b_relu_2/cond/Identity"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?
\
cond_false_46359
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:?????????T2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
R
cond_true_46358
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:?????????T2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_46156

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????

02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
 spiking_b_relu_2_cond_true_458828
4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast%
!spiking_b_relu_2_cond_placeholder"
spiking_b_relu_2_cond_identity?
spiking_b_relu_2/cond/IdentityIdentity4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast*
T0*'
_output_shapes
:?????????x2 
spiking_b_relu_2/cond/Identity"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?
?
*__inference_sequential_layer_call_fn_46022

inputs!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	?	x
	unknown_6:x
	unknown_7: 
	unknown_8:xT
	unknown_9:T

unknown_10: 

unknown_11:Td

unknown_12:d

unknown_13: 

unknown_14:d

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_450702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_44990

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T2
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/ya
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:?????????T2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:?????????T2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yd
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:?????????T2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????T2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????T2
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_44978*&
output_shapes
:?????????T*"
then_branchR
cond_true_449772
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:?????????T2
cond/Identityq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:?????????T2

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????T: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
 spiking_b_relu_3_cond_true_457028
4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast%
!spiking_b_relu_3_cond_placeholder"
spiking_b_relu_3_cond_identity?
spiking_b_relu_3/cond/IdentityIdentity4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast*
T0*'
_output_shapes
:?????????T2 
spiking_b_relu_3/cond/Identity"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
?
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_45049

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/ya
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:?????????d2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:?????????d2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yd
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:?????????d2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????d2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????d2
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_45037*&
output_shapes
:?????????d*"
then_branchR
cond_true_450362
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:?????????d2
cond/Identityq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_46078

inputs!
unknown: 
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
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_447482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_46239

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
 spiking_b_relu_1_cond_true_458388
4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast%
!spiking_b_relu_1_cond_placeholder"
spiking_b_relu_1_cond_identity?
spiking_b_relu_1/cond/IdentityIdentity4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast*
T0*/
_output_shapes
:?????????

02 
spiking_b_relu_1/cond/Identity"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?
?
+sequential_spiking_b_relu_4_cond_true_44666N
Jsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_cast0
,sequential_spiking_b_relu_4_cond_placeholder-
)sequential_spiking_b_relu_4_cond_identity?
)sequential/spiking_b_relu_4/cond/IdentityIdentityJsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_cast*
T0*'
_output_shapes
:?????????d2+
)sequential/spiking_b_relu_4/cond/Identity"_
)sequential_spiking_b_relu_4_cond_identity2sequential/spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
?
)sequential_spiking_b_relu_cond_true_44498J
Fsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_cast.
*sequential_spiking_b_relu_cond_placeholder+
'sequential_spiking_b_relu_cond_identity?
'sequential/spiking_b_relu/cond/IdentityIdentityFsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_cast*
T0*/
_output_shapes
:????????? 2)
'sequential/spiking_b_relu/cond/Identity"[
'sequential_spiking_b_relu_cond_identity0sequential/spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?
?
+sequential_spiking_b_relu_1_cond_true_44540N
Jsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_cast0
,sequential_spiking_b_relu_1_cond_placeholder-
)sequential_spiking_b_relu_1_cond_identity?
)sequential/spiking_b_relu_1/cond/IdentityIdentityJsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_cast*
T0*/
_output_shapes
:?????????

02+
)sequential/spiking_b_relu_1/cond/Identity"_
)sequential_spiking_b_relu_1_cond_identity2sequential/spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?
?
,sequential_spiking_b_relu_1_cond_false_445410
,sequential_spiking_b_relu_1_cond_placeholderW
Ssequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_clip_by_value-
)sequential_spiking_b_relu_1_cond_identity?
)sequential/spiking_b_relu_1/cond/IdentityIdentitySsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:?????????

02+
)sequential/spiking_b_relu_1/cond/Identity"_
)sequential_spiking_b_relu_1_cond_identity2sequential/spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?
\
cond_false_46292
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:?????????x2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?
?
!spiking_b_relu_2_cond_false_45883%
!spiking_b_relu_2_cond_placeholderA
=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value"
spiking_b_relu_2_cond_identity?
spiking_b_relu_2/cond/IdentityIdentity=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:?????????x2 
spiking_b_relu_2/cond/Identity"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?
R
cond_true_44918
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:?????????x2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?
?
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_44858

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y?
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????

02
GreaterEqualo
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

02
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/yi
sub_1Subinputssub_1/y:output:0*
T0*/
_output_shapes
:?????????

02
sub_1c
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:?????????

02
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yl
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:?????????

02
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????

02
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????

02
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????

0* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_44846*.
output_shapes
:?????????

0*"
then_branchR
cond_true_448452
conds
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:?????????

02
cond/Identityy
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????

02

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????

0: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:?????????

0
 
_user_specified_nameinputs
?E
?
E__inference_sequential_layer_call_and_return_conditional_losses_45480
conv2d_input&
conv2d_45433: 
conv2d_45435: 
spiking_b_relu_45438: (
conv2d_1_45442: 0
conv2d_1_45444:0 
spiking_b_relu_1_45447: 
dense_45452:	?	x
dense_45454:x 
spiking_b_relu_2_45457: 
dense_1_45460:xT
dense_1_45462:T 
spiking_b_relu_3_45465: 
dense_2_45468:Td
dense_2_45470:d 
spiking_b_relu_4_45473: '
softmax__decode_45476:d

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?'softmax__decode/StatefulPartitionedCall?&spiking_b_relu/StatefulPartitionedCall?(spiking_b_relu_1/StatefulPartitionedCall?(spiking_b_relu_2/StatefulPartitionedCall?(spiking_b_relu_3/StatefulPartitionedCall?(spiking_b_relu_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_45433conv2d_45435*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_447482 
conv2d/StatefulPartitionedCall?
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0spiking_b_relu_45438*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_447932(
&spiking_b_relu/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_448012
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_45442conv2d_1_45444*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_448132"
 conv2d_1/StatefulPartitionedCall?
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0spiking_b_relu_1_45447*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_448582*
(spiking_b_relu_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_448662!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_448742
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_45452dense_45454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_448862
dense/StatefulPartitionedCall?
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_2_45457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_449312*
(spiking_b_relu_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0dense_1_45460dense_1_45462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_449452!
dense_1/StatefulPartitionedCall?
(spiking_b_relu_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_3_45465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_449902*
(spiking_b_relu_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_3/StatefulPartitionedCall:output:0dense_2_45468dense_2_45470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_450042!
dense_2/StatefulPartitionedCall?
(spiking_b_relu_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_4_45473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_450492*
(spiking_b_relu_4/StatefulPartitionedCall?
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_4/StatefulPartitionedCall:output:0softmax__decode_45476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_450652)
'softmax__decode/StatefulPartitionedCall?
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall2T
(spiking_b_relu_3/StatefulPartitionedCall(spiking_b_relu_3/StatefulPartitionedCall2T
(spiking_b_relu_4/StatefulPartitionedCall(spiking_b_relu_4/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
|
0__inference_spiking_b_relu_3_layer_call_fn_46378

inputs
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_449902
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????T2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????T: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
!spiking_b_relu_4_cond_false_45965%
!spiking_b_relu_4_cond_placeholderA
=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value"
spiking_b_relu_4_cond_identity?
spiking_b_relu_4/cond/IdentityIdentity=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:?????????d2 
spiking_b_relu_4/cond/Identity"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_46388

inputs0
matmul_readvariableop_resource:Td-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Td*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_44866

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

0:W S
/
_output_shapes
:?????????

0
 
_user_specified_nameinputs
?
?
!spiking_b_relu_2_cond_false_45662%
!spiking_b_relu_2_cond_placeholderA
=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value"
spiking_b_relu_2_cond_identity?
spiking_b_relu_2/cond/IdentityIdentity=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:?????????x2 
spiking_b_relu_2/cond/Identity"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?
?
+sequential_spiking_b_relu_3_cond_true_44625N
Jsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_cast0
,sequential_spiking_b_relu_3_cond_placeholder-
)sequential_spiking_b_relu_3_cond_identity?
)sequential/spiking_b_relu_3/cond/IdentityIdentityJsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_cast*
T0*'
_output_shapes
:?????????T2+
)sequential/spiking_b_relu_3/cond/Identity"_
)sequential_spiking_b_relu_3_cond_identity2sequential/spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_44748

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
!spiking_b_relu_3_cond_false_45924%
!spiking_b_relu_3_cond_placeholderA
=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value"
spiking_b_relu_3_cond_identity?
spiking_b_relu_3/cond/IdentityIdentity=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:?????????T2 
spiking_b_relu_3/cond/Identity"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????T:?????????T:- )
'
_output_shapes
:?????????T:-)
'
_output_shapes
:?????????T
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46223

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

0:W S
/
_output_shapes
:?????????

0
 
_user_specified_nameinputs
?
z
.__inference_spiking_b_relu_layer_call_fn_46126

inputs
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_447932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_46206

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y?
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????

02
GreaterEqualo
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

02
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/yi
sub_1Subinputssub_1/y:output:0*
T0*/
_output_shapes
:?????????

02
sub_1c
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:?????????

02
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yl
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:?????????

02
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????

02
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????

02
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????

0* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_46194*.
output_shapes
:?????????

0*"
then_branchR
cond_true_461932
conds
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:?????????

02
cond/Identityy
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????

02

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????

0: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:?????????

0
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46218

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
J__inference_softmax__decode_layer_call_and_return_conditional_losses_46457

inputs0
matmul_readvariableop_resource:d

identity??MatMul/ReadVariableOpS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:?????????d2
mulS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/y\
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:?????????d2
sub?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:?????????
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?r
?
__inference__traced_save_46649
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop'
#savev2_variable_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop)
%savev2_variable_1_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop)
%savev2_variable_2_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop)
%savev2_variable_3_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_5_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop@
<savev2_adadelta_conv2d_kernel_accum_grad_read_readvariableop>
:savev2_adadelta_conv2d_bias_accum_grad_read_readvariableop;
7savev2_adadelta_variable_accum_grad_read_readvariableopB
>savev2_adadelta_conv2d_1_kernel_accum_grad_read_readvariableop@
<savev2_adadelta_conv2d_1_bias_accum_grad_read_readvariableop=
9savev2_adadelta_variable_accum_grad_1_read_readvariableop?
;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_dense_bias_accum_grad_read_readvariableop=
9savev2_adadelta_variable_accum_grad_2_read_readvariableopA
=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableop=
9savev2_adadelta_variable_accum_grad_3_read_readvariableopA
=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableop=
9savev2_adadelta_variable_accum_grad_4_read_readvariableop?
;savev2_adadelta_conv2d_kernel_accum_var_read_readvariableop=
9savev2_adadelta_conv2d_bias_accum_var_read_readvariableop:
6savev2_adadelta_variable_accum_var_read_readvariableopA
=savev2_adadelta_conv2d_1_kernel_accum_var_read_readvariableop?
;savev2_adadelta_conv2d_1_bias_accum_var_read_readvariableop<
8savev2_adadelta_variable_accum_var_1_read_readvariableop>
:savev2_adadelta_dense_kernel_accum_var_read_readvariableop<
8savev2_adadelta_dense_bias_accum_var_read_readvariableop<
8savev2_adadelta_variable_accum_var_2_read_readvariableop@
<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop<
8savev2_adadelta_variable_accum_var_3_read_readvariableop@
<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_2_bias_accum_var_read_readvariableop<
8savev2_adadelta_variable_accum_var_4_read_readvariableop
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
ShardedFilename?!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*? 
value? B? 7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-7/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-9/sharpness/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop#savev2_variable_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop%savev2_variable_1_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop%savev2_variable_2_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop%savev2_variable_3_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_5_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop<savev2_adadelta_conv2d_kernel_accum_grad_read_readvariableop:savev2_adadelta_conv2d_bias_accum_grad_read_readvariableop7savev2_adadelta_variable_accum_grad_read_readvariableop>savev2_adadelta_conv2d_1_kernel_accum_grad_read_readvariableop<savev2_adadelta_conv2d_1_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_1_read_readvariableop;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop9savev2_adadelta_dense_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_2_read_readvariableop=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_3_read_readvariableop=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_4_read_readvariableop;savev2_adadelta_conv2d_kernel_accum_var_read_readvariableop9savev2_adadelta_conv2d_bias_accum_var_read_readvariableop6savev2_adadelta_variable_accum_var_read_readvariableop=savev2_adadelta_conv2d_1_kernel_accum_var_read_readvariableop;savev2_adadelta_conv2d_1_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_1_read_readvariableop:savev2_adadelta_dense_kernel_accum_var_read_readvariableop8savev2_adadelta_dense_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_2_read_readvariableop<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_3_read_readvariableop<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_2_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_4_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	2
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

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : 0:0: :	?	x:x: :xT:T: :Td:d: :d
: : : : : : : : : : : : 0:0: :	?	x:x: :xT:T: :Td:d: : : : : 0:0: :	?	x:x: :xT:T: :Td:d: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:

_output_shapes
: :%!

_output_shapes
:	?	x: 

_output_shapes
:x:	

_output_shapes
: :$
 

_output_shapes

:xT: 

_output_shapes
:T:

_output_shapes
: :$ 

_output_shapes

:Td: 

_output_shapes
:d:

_output_shapes
: :$ 

_output_shapes

:d
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:

_output_shapes
: :%!

_output_shapes
:	?	x:  

_output_shapes
:x:!

_output_shapes
: :$" 

_output_shapes

:xT: #

_output_shapes
:T:$

_output_shapes
: :$% 

_output_shapes

:Td: &

_output_shapes
:d:'

_output_shapes
: :,((
&
_output_shapes
: : )

_output_shapes
: :*

_output_shapes
: :,+(
&
_output_shapes
: 0: ,

_output_shapes
:0:-

_output_shapes
: :%.!

_output_shapes
:	?	x: /

_output_shapes
:x:0

_output_shapes
: :$1 

_output_shapes

:xT: 2

_output_shapes
:T:3

_output_shapes
: :$4 

_output_shapes

:Td: 5

_output_shapes
:d:6

_output_shapes
: :7

_output_shapes
: 
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_45764

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 0
&spiking_b_relu_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: 06
(conv2d_1_biasadd_readvariableop_resource:02
(spiking_b_relu_1_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	?	x3
%dense_biasadd_readvariableop_resource:x2
(spiking_b_relu_2_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:xT5
'dense_1_biasadd_readvariableop_resource:T2
(spiking_b_relu_3_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:Td5
'dense_2_biasadd_readvariableop_resource:d2
(spiking_b_relu_4_readvariableop_resource: @
.softmax__decode_matmul_readvariableop_resource:d

identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?%softmax__decode/MatMul/ReadVariableOp?#spiking_b_relu/Equal/ReadVariableOp?spiking_b_relu/ReadVariableOp?%spiking_b_relu_1/Equal/ReadVariableOp?spiking_b_relu_1/ReadVariableOp?%spiking_b_relu_2/Equal/ReadVariableOp?spiking_b_relu_2/ReadVariableOp?%spiking_b_relu_3/Equal/ReadVariableOp?spiking_b_relu_3/ReadVariableOp?%spiking_b_relu_4/Equal/ReadVariableOp?spiking_b_relu_4/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdd?
spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu/GreaterEqual/y?
spiking_b_relu/GreaterEqualGreaterEqualconv2d/BiasAdd:output:0&spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/GreaterEqual?
spiking_b_relu/CastCastspiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
spiking_b_relu/Cast?
spiking_b_relu/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype02
spiking_b_relu/ReadVariableOpq
spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu/sub/x?
spiking_b_relu/subSubspiking_b_relu/sub/x:output:0%spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu/subq
spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu/add/y?
spiking_b_relu/addAddV2spiking_b_relu/sub:z:0spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu/addy
spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu/truediv/x?
spiking_b_relu/truedivRealDiv!spiking_b_relu/truediv/x:output:0spiking_b_relu/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu/truedivu
spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu/sub_1/y?
spiking_b_relu/sub_1Subconv2d/BiasAdd:output:0spiking_b_relu/sub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/sub_1?
spiking_b_relu/mulMulspiking_b_relu/truediv:z:0spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/mulu
spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu/add_1/y?
spiking_b_relu/add_1AddV2spiking_b_relu/mul:z:0spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/add_1?
&spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&spiking_b_relu/clip_by_value/Minimum/y?
$spiking_b_relu/clip_by_value/MinimumMinimumspiking_b_relu/add_1:z:0/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2&
$spiking_b_relu/clip_by_value/Minimum?
spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
spiking_b_relu/clip_by_value/y?
spiking_b_relu/clip_by_valueMaximum(spiking_b_relu/clip_by_value/Minimum:z:0'spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/clip_by_value?
#spiking_b_relu/Equal/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype02%
#spiking_b_relu/Equal/ReadVariableOpu
spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu/Equal/y?
spiking_b_relu/EqualEqual+spiking_b_relu/Equal/ReadVariableOp:value:0spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu/Equal?
spiking_b_relu/condStatelessIfspiking_b_relu/Equal:z:0spiking_b_relu/Cast:y:0 spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *2
else_branch#R!
spiking_b_relu_cond_false_45576*.
output_shapes
:????????? *1
then_branch"R 
spiking_b_relu_cond_true_455752
spiking_b_relu/cond?
spiking_b_relu/cond/IdentityIdentityspiking_b_relu/cond:output:0*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/cond/Identity?
max_pooling2d/MaxPoolMaxPool%spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

0*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

02
conv2d_1/BiasAdd?
spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spiking_b_relu_1/GreaterEqual/y?
spiking_b_relu_1/GreaterEqualGreaterEqualconv2d_1/BiasAdd:output:0(spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????

02
spiking_b_relu_1/GreaterEqual?
spiking_b_relu_1/CastCast!spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

02
spiking_b_relu_1/Cast?
spiking_b_relu_1/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype02!
spiking_b_relu_1/ReadVariableOpu
spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_1/sub/x?
spiking_b_relu_1/subSubspiking_b_relu_1/sub/x:output:0'spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu_1/subu
spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu_1/add/y?
spiking_b_relu_1/addAddV2spiking_b_relu_1/sub:z:0spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_1/add}
spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_1/truediv/x?
spiking_b_relu_1/truedivRealDiv#spiking_b_relu_1/truediv/x:output:0spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu_1/truedivy
spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_1/sub_1/y?
spiking_b_relu_1/sub_1Subconv2d_1/BiasAdd:output:0!spiking_b_relu_1/sub_1/y:output:0*
T0*/
_output_shapes
:?????????

02
spiking_b_relu_1/sub_1?
spiking_b_relu_1/mulMulspiking_b_relu_1/truediv:z:0spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:?????????

02
spiking_b_relu_1/muly
spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_1/add_1/y?
spiking_b_relu_1/add_1AddV2spiking_b_relu_1/mul:z:0!spiking_b_relu_1/add_1/y:output:0*
T0*/
_output_shapes
:?????????

02
spiking_b_relu_1/add_1?
(spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(spiking_b_relu_1/clip_by_value/Minimum/y?
&spiking_b_relu_1/clip_by_value/MinimumMinimumspiking_b_relu_1/add_1:z:01spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????

02(
&spiking_b_relu_1/clip_by_value/Minimum?
 spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 spiking_b_relu_1/clip_by_value/y?
spiking_b_relu_1/clip_by_valueMaximum*spiking_b_relu_1/clip_by_value/Minimum:z:0)spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????

02 
spiking_b_relu_1/clip_by_value?
%spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype02'
%spiking_b_relu_1/Equal/ReadVariableOpy
spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_1/Equal/y?
spiking_b_relu_1/EqualEqual-spiking_b_relu_1/Equal/ReadVariableOp:value:0!spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_1/Equal?
spiking_b_relu_1/condStatelessIfspiking_b_relu_1/Equal:z:0spiking_b_relu_1/Cast:y:0"spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????

0* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_1_cond_false_45618*.
output_shapes
:?????????

0*3
then_branch$R"
 spiking_b_relu_1_cond_true_456172
spiking_b_relu_1/cond?
spiking_b_relu_1/cond/IdentityIdentityspiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:?????????

02 
spiking_b_relu_1/cond/Identity?
max_pooling2d_1/MaxPoolMaxPool'spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense/BiasAdd?
spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spiking_b_relu_2/GreaterEqual/y?
spiking_b_relu_2/GreaterEqualGreaterEqualdense/BiasAdd:output:0(spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????x2
spiking_b_relu_2/GreaterEqual?
spiking_b_relu_2/CastCast!spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????x2
spiking_b_relu_2/Cast?
spiking_b_relu_2/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype02!
spiking_b_relu_2/ReadVariableOpu
spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_2/sub/x?
spiking_b_relu_2/subSubspiking_b_relu_2/sub/x:output:0'spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu_2/subu
spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu_2/add/y?
spiking_b_relu_2/addAddV2spiking_b_relu_2/sub:z:0spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_2/add}
spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_2/truediv/x?
spiking_b_relu_2/truedivRealDiv#spiking_b_relu_2/truediv/x:output:0spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu_2/truedivy
spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_2/sub_1/y?
spiking_b_relu_2/sub_1Subdense/BiasAdd:output:0!spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:?????????x2
spiking_b_relu_2/sub_1?
spiking_b_relu_2/mulMulspiking_b_relu_2/truediv:z:0spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:?????????x2
spiking_b_relu_2/muly
spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_2/add_1/y?
spiking_b_relu_2/add_1AddV2spiking_b_relu_2/mul:z:0!spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:?????????x2
spiking_b_relu_2/add_1?
(spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(spiking_b_relu_2/clip_by_value/Minimum/y?
&spiking_b_relu_2/clip_by_value/MinimumMinimumspiking_b_relu_2/add_1:z:01spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????x2(
&spiking_b_relu_2/clip_by_value/Minimum?
 spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 spiking_b_relu_2/clip_by_value/y?
spiking_b_relu_2/clip_by_valueMaximum*spiking_b_relu_2/clip_by_value/Minimum:z:0)spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????x2 
spiking_b_relu_2/clip_by_value?
%spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype02'
%spiking_b_relu_2/Equal/ReadVariableOpy
spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_2/Equal/y?
spiking_b_relu_2/EqualEqual-spiking_b_relu_2/Equal/ReadVariableOp:value:0!spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_2/Equal?
spiking_b_relu_2/condStatelessIfspiking_b_relu_2/Equal:z:0spiking_b_relu_2/Cast:y:0"spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_2_cond_false_45662*&
output_shapes
:?????????x*3
then_branch$R"
 spiking_b_relu_2_cond_true_456612
spiking_b_relu_2/cond?
spiking_b_relu_2/cond/IdentityIdentityspiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:?????????x2 
spiking_b_relu_2/cond/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul'spiking_b_relu_2/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_1/BiasAdd?
spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spiking_b_relu_3/GreaterEqual/y?
spiking_b_relu_3/GreaterEqualGreaterEqualdense_1/BiasAdd:output:0(spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T2
spiking_b_relu_3/GreaterEqual?
spiking_b_relu_3/CastCast!spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T2
spiking_b_relu_3/Cast?
spiking_b_relu_3/ReadVariableOpReadVariableOp(spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype02!
spiking_b_relu_3/ReadVariableOpu
spiking_b_relu_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_3/sub/x?
spiking_b_relu_3/subSubspiking_b_relu_3/sub/x:output:0'spiking_b_relu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu_3/subu
spiking_b_relu_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu_3/add/y?
spiking_b_relu_3/addAddV2spiking_b_relu_3/sub:z:0spiking_b_relu_3/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_3/add}
spiking_b_relu_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_3/truediv/x?
spiking_b_relu_3/truedivRealDiv#spiking_b_relu_3/truediv/x:output:0spiking_b_relu_3/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu_3/truedivy
spiking_b_relu_3/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_3/sub_1/y?
spiking_b_relu_3/sub_1Subdense_1/BiasAdd:output:0!spiking_b_relu_3/sub_1/y:output:0*
T0*'
_output_shapes
:?????????T2
spiking_b_relu_3/sub_1?
spiking_b_relu_3/mulMulspiking_b_relu_3/truediv:z:0spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:?????????T2
spiking_b_relu_3/muly
spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_3/add_1/y?
spiking_b_relu_3/add_1AddV2spiking_b_relu_3/mul:z:0!spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:?????????T2
spiking_b_relu_3/add_1?
(spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(spiking_b_relu_3/clip_by_value/Minimum/y?
&spiking_b_relu_3/clip_by_value/MinimumMinimumspiking_b_relu_3/add_1:z:01spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????T2(
&spiking_b_relu_3/clip_by_value/Minimum?
 spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 spiking_b_relu_3/clip_by_value/y?
spiking_b_relu_3/clip_by_valueMaximum*spiking_b_relu_3/clip_by_value/Minimum:z:0)spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????T2 
spiking_b_relu_3/clip_by_value?
%spiking_b_relu_3/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype02'
%spiking_b_relu_3/Equal/ReadVariableOpy
spiking_b_relu_3/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_3/Equal/y?
spiking_b_relu_3/EqualEqual-spiking_b_relu_3/Equal/ReadVariableOp:value:0!spiking_b_relu_3/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_3/Equal?
spiking_b_relu_3/condStatelessIfspiking_b_relu_3/Equal:z:0spiking_b_relu_3/Cast:y:0"spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_3_cond_false_45703*&
output_shapes
:?????????T*3
then_branch$R"
 spiking_b_relu_3_cond_true_457022
spiking_b_relu_3/cond?
spiking_b_relu_3/cond/IdentityIdentityspiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:?????????T2 
spiking_b_relu_3/cond/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul'spiking_b_relu_3/cond/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_2/BiasAdd?
spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spiking_b_relu_4/GreaterEqual/y?
spiking_b_relu_4/GreaterEqualGreaterEqualdense_2/BiasAdd:output:0(spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2
spiking_b_relu_4/GreaterEqual?
spiking_b_relu_4/CastCast!spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2
spiking_b_relu_4/Cast?
spiking_b_relu_4/ReadVariableOpReadVariableOp(spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype02!
spiking_b_relu_4/ReadVariableOpu
spiking_b_relu_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_4/sub/x?
spiking_b_relu_4/subSubspiking_b_relu_4/sub/x:output:0'spiking_b_relu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
spiking_b_relu_4/subu
spiking_b_relu_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
spiking_b_relu_4/add/y?
spiking_b_relu_4/addAddV2spiking_b_relu_4/sub:z:0spiking_b_relu_4/add/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_4/add}
spiking_b_relu_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_4/truediv/x?
spiking_b_relu_4/truedivRealDiv#spiking_b_relu_4/truediv/x:output:0spiking_b_relu_4/add:z:0*
T0*
_output_shapes
: 2
spiking_b_relu_4/truedivy
spiking_b_relu_4/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_4/sub_1/y?
spiking_b_relu_4/sub_1Subdense_2/BiasAdd:output:0!spiking_b_relu_4/sub_1/y:output:0*
T0*'
_output_shapes
:?????????d2
spiking_b_relu_4/sub_1?
spiking_b_relu_4/mulMulspiking_b_relu_4/truediv:z:0spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:?????????d2
spiking_b_relu_4/muly
spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
spiking_b_relu_4/add_1/y?
spiking_b_relu_4/add_1AddV2spiking_b_relu_4/mul:z:0!spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:?????????d2
spiking_b_relu_4/add_1?
(spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(spiking_b_relu_4/clip_by_value/Minimum/y?
&spiking_b_relu_4/clip_by_value/MinimumMinimumspiking_b_relu_4/add_1:z:01spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????d2(
&spiking_b_relu_4/clip_by_value/Minimum?
 spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 spiking_b_relu_4/clip_by_value/y?
spiking_b_relu_4/clip_by_valueMaximum*spiking_b_relu_4/clip_by_value/Minimum:z:0)spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????d2 
spiking_b_relu_4/clip_by_value?
%spiking_b_relu_4/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype02'
%spiking_b_relu_4/Equal/ReadVariableOpy
spiking_b_relu_4/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
spiking_b_relu_4/Equal/y?
spiking_b_relu_4/EqualEqual-spiking_b_relu_4/Equal/ReadVariableOp:value:0!spiking_b_relu_4/Equal/y:output:0*
T0*
_output_shapes
: 2
spiking_b_relu_4/Equal?
spiking_b_relu_4/condStatelessIfspiking_b_relu_4/Equal:z:0spiking_b_relu_4/Cast:y:0"spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_4_cond_false_45744*&
output_shapes
:?????????d*3
then_branch$R"
 spiking_b_relu_4_cond_true_457432
spiking_b_relu_4/cond?
spiking_b_relu_4/cond/IdentityIdentityspiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:?????????d2 
spiking_b_relu_4/cond/Identitys
softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
softmax__decode/mul/x?
softmax__decode/mulMulsoftmax__decode/mul/x:output:0'spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:?????????d2
softmax__decode/muls
softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
softmax__decode/sub/y?
softmax__decode/subSubsoftmax__decode/mul:z:0softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:?????????d2
softmax__decode/sub?
%softmax__decode/MatMul/ReadVariableOpReadVariableOp.softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02'
%softmax__decode/MatMul/ReadVariableOp?
softmax__decode/MatMulMatMulsoftmax__decode/sub:z:0-softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
softmax__decode/MatMul?
softmax__decode/SoftmaxSoftmax softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:?????????
2
softmax__decode/Softmax|
IdentityIdentity!softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^softmax__decode/MatMul/ReadVariableOp$^spiking_b_relu/Equal/ReadVariableOp^spiking_b_relu/ReadVariableOp&^spiking_b_relu_1/Equal/ReadVariableOp ^spiking_b_relu_1/ReadVariableOp&^spiking_b_relu_2/Equal/ReadVariableOp ^spiking_b_relu_2/ReadVariableOp&^spiking_b_relu_3/Equal/ReadVariableOp ^spiking_b_relu_3/ReadVariableOp&^spiking_b_relu_4/Equal/ReadVariableOp ^spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2N
%softmax__decode/MatMul/ReadVariableOp%softmax__decode/MatMul/ReadVariableOp2J
#spiking_b_relu/Equal/ReadVariableOp#spiking_b_relu/Equal/ReadVariableOp2>
spiking_b_relu/ReadVariableOpspiking_b_relu/ReadVariableOp2N
%spiking_b_relu_1/Equal/ReadVariableOp%spiking_b_relu_1/Equal/ReadVariableOp2B
spiking_b_relu_1/ReadVariableOpspiking_b_relu_1/ReadVariableOp2N
%spiking_b_relu_2/Equal/ReadVariableOp%spiking_b_relu_2/Equal/ReadVariableOp2B
spiking_b_relu_2/ReadVariableOpspiking_b_relu_2/ReadVariableOp2N
%spiking_b_relu_3/Equal/ReadVariableOp%spiking_b_relu_3/Equal/ReadVariableOp2B
spiking_b_relu_3/ReadVariableOpspiking_b_relu_3/ReadVariableOp2N
%spiking_b_relu_4/Equal/ReadVariableOp%spiking_b_relu_4/Equal/ReadVariableOp2B
spiking_b_relu_4/ReadVariableOpspiking_b_relu_4/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_44687
conv2d_inputJ
0sequential_conv2d_conv2d_readvariableop_resource: ?
1sequential_conv2d_biasadd_readvariableop_resource: ;
1sequential_spiking_b_relu_readvariableop_resource: L
2sequential_conv2d_1_conv2d_readvariableop_resource: 0A
3sequential_conv2d_1_biasadd_readvariableop_resource:0=
3sequential_spiking_b_relu_1_readvariableop_resource: B
/sequential_dense_matmul_readvariableop_resource:	?	x>
0sequential_dense_biasadd_readvariableop_resource:x=
3sequential_spiking_b_relu_2_readvariableop_resource: C
1sequential_dense_1_matmul_readvariableop_resource:xT@
2sequential_dense_1_biasadd_readvariableop_resource:T=
3sequential_spiking_b_relu_3_readvariableop_resource: C
1sequential_dense_2_matmul_readvariableop_resource:Td@
2sequential_dense_2_biasadd_readvariableop_resource:d=
3sequential_spiking_b_relu_4_readvariableop_resource: K
9sequential_softmax__decode_matmul_readvariableop_resource:d

identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?0sequential/softmax__decode/MatMul/ReadVariableOp?.sequential/spiking_b_relu/Equal/ReadVariableOp?(sequential/spiking_b_relu/ReadVariableOp?0sequential/spiking_b_relu_1/Equal/ReadVariableOp?*sequential/spiking_b_relu_1/ReadVariableOp?0sequential/spiking_b_relu_2/Equal/ReadVariableOp?*sequential/spiking_b_relu_2/ReadVariableOp?0sequential/spiking_b_relu_3/Equal/ReadVariableOp?*sequential/spiking_b_relu_3/ReadVariableOp?0sequential/spiking_b_relu_4/Equal/ReadVariableOp?*sequential/spiking_b_relu_4/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d/BiasAdd?
(sequential/spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(sequential/spiking_b_relu/GreaterEqual/y?
&sequential/spiking_b_relu/GreaterEqualGreaterEqual"sequential/conv2d/BiasAdd:output:01sequential/spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2(
&sequential/spiking_b_relu/GreaterEqual?
sequential/spiking_b_relu/CastCast*sequential/spiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2 
sequential/spiking_b_relu/Cast?
(sequential/spiking_b_relu/ReadVariableOpReadVariableOp1sequential_spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/spiking_b_relu/ReadVariableOp?
sequential/spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
sequential/spiking_b_relu/sub/x?
sequential/spiking_b_relu/subSub(sequential/spiking_b_relu/sub/x:output:00sequential/spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
sequential/spiking_b_relu/sub?
sequential/spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2!
sequential/spiking_b_relu/add/y?
sequential/spiking_b_relu/addAddV2!sequential/spiking_b_relu/sub:z:0(sequential/spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: 2
sequential/spiking_b_relu/add?
#sequential/spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#sequential/spiking_b_relu/truediv/x?
!sequential/spiking_b_relu/truedivRealDiv,sequential/spiking_b_relu/truediv/x:output:0!sequential/spiking_b_relu/add:z:0*
T0*
_output_shapes
: 2#
!sequential/spiking_b_relu/truediv?
!sequential/spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!sequential/spiking_b_relu/sub_1/y?
sequential/spiking_b_relu/sub_1Sub"sequential/conv2d/BiasAdd:output:0*sequential/spiking_b_relu/sub_1/y:output:0*
T0*/
_output_shapes
:????????? 2!
sequential/spiking_b_relu/sub_1?
sequential/spiking_b_relu/mulMul%sequential/spiking_b_relu/truediv:z:0#sequential/spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:????????? 2
sequential/spiking_b_relu/mul?
!sequential/spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!sequential/spiking_b_relu/add_1/y?
sequential/spiking_b_relu/add_1AddV2!sequential/spiking_b_relu/mul:z:0*sequential/spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:????????? 2!
sequential/spiking_b_relu/add_1?
1sequential/spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??23
1sequential/spiking_b_relu/clip_by_value/Minimum/y?
/sequential/spiking_b_relu/clip_by_value/MinimumMinimum#sequential/spiking_b_relu/add_1:z:0:sequential/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 21
/sequential/spiking_b_relu/clip_by_value/Minimum?
)sequential/spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)sequential/spiking_b_relu/clip_by_value/y?
'sequential/spiking_b_relu/clip_by_valueMaximum3sequential/spiking_b_relu/clip_by_value/Minimum:z:02sequential/spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2)
'sequential/spiking_b_relu/clip_by_value?
.sequential/spiking_b_relu/Equal/ReadVariableOpReadVariableOp1sequential_spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential/spiking_b_relu/Equal/ReadVariableOp?
!sequential/spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!sequential/spiking_b_relu/Equal/y?
sequential/spiking_b_relu/EqualEqual6sequential/spiking_b_relu/Equal/ReadVariableOp:value:0*sequential/spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu/Equal?
sequential/spiking_b_relu/condStatelessIf#sequential/spiking_b_relu/Equal:z:0"sequential/spiking_b_relu/Cast:y:0+sequential/spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *=
else_branch.R,
*sequential_spiking_b_relu_cond_false_44499*.
output_shapes
:????????? *<
then_branch-R+
)sequential_spiking_b_relu_cond_true_444982 
sequential/spiking_b_relu/cond?
'sequential/spiking_b_relu/cond/IdentityIdentity'sequential/spiking_b_relu/cond:output:0*
T0*/
_output_shapes
:????????? 2)
'sequential/spiking_b_relu/cond/Identity?
 sequential/max_pooling2d/MaxPoolMaxPool0sequential/spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

0*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

02
sequential/conv2d_1/BiasAdd?
*sequential/spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*sequential/spiking_b_relu_1/GreaterEqual/y?
(sequential/spiking_b_relu_1/GreaterEqualGreaterEqual$sequential/conv2d_1/BiasAdd:output:03sequential/spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????

02*
(sequential/spiking_b_relu_1/GreaterEqual?
 sequential/spiking_b_relu_1/CastCast,sequential/spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

02"
 sequential/spiking_b_relu_1/Cast?
*sequential/spiking_b_relu_1/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/spiking_b_relu_1/ReadVariableOp?
!sequential/spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!sequential/spiking_b_relu_1/sub/x?
sequential/spiking_b_relu_1/subSub*sequential/spiking_b_relu_1/sub/x:output:02sequential/spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu_1/sub?
!sequential/spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!sequential/spiking_b_relu_1/add/y?
sequential/spiking_b_relu_1/addAddV2#sequential/spiking_b_relu_1/sub:z:0*sequential/spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu_1/add?
%sequential/spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%sequential/spiking_b_relu_1/truediv/x?
#sequential/spiking_b_relu_1/truedivRealDiv.sequential/spiking_b_relu_1/truediv/x:output:0#sequential/spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: 2%
#sequential/spiking_b_relu_1/truediv?
#sequential/spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential/spiking_b_relu_1/sub_1/y?
!sequential/spiking_b_relu_1/sub_1Sub$sequential/conv2d_1/BiasAdd:output:0,sequential/spiking_b_relu_1/sub_1/y:output:0*
T0*/
_output_shapes
:?????????

02#
!sequential/spiking_b_relu_1/sub_1?
sequential/spiking_b_relu_1/mulMul'sequential/spiking_b_relu_1/truediv:z:0%sequential/spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:?????????

02!
sequential/spiking_b_relu_1/mul?
#sequential/spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential/spiking_b_relu_1/add_1/y?
!sequential/spiking_b_relu_1/add_1AddV2#sequential/spiking_b_relu_1/mul:z:0,sequential/spiking_b_relu_1/add_1/y:output:0*
T0*/
_output_shapes
:?????????

02#
!sequential/spiking_b_relu_1/add_1?
3sequential/spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??25
3sequential/spiking_b_relu_1/clip_by_value/Minimum/y?
1sequential/spiking_b_relu_1/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_1/add_1:z:0<sequential/spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????

023
1sequential/spiking_b_relu_1/clip_by_value/Minimum?
+sequential/spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+sequential/spiking_b_relu_1/clip_by_value/y?
)sequential/spiking_b_relu_1/clip_by_valueMaximum5sequential/spiking_b_relu_1/clip_by_value/Minimum:z:04sequential/spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????

02+
)sequential/spiking_b_relu_1/clip_by_value?
0sequential/spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential/spiking_b_relu_1/Equal/ReadVariableOp?
#sequential/spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#sequential/spiking_b_relu_1/Equal/y?
!sequential/spiking_b_relu_1/EqualEqual8sequential/spiking_b_relu_1/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: 2#
!sequential/spiking_b_relu_1/Equal?
 sequential/spiking_b_relu_1/condStatelessIf%sequential/spiking_b_relu_1/Equal:z:0$sequential/spiking_b_relu_1/Cast:y:0-sequential/spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????

0* 
_read_only_resource_inputs
 *?
else_branch0R.
,sequential_spiking_b_relu_1_cond_false_44541*.
output_shapes
:?????????

0*>
then_branch/R-
+sequential_spiking_b_relu_1_cond_true_445402"
 sequential/spiking_b_relu_1/cond?
)sequential/spiking_b_relu_1/cond/IdentityIdentity)sequential/spiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:?????????

02+
)sequential/spiking_b_relu_1/cond/Identity?
"sequential/max_pooling2d_1/MaxPoolMaxPool2sequential/spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_1/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
sequential/dense/BiasAdd?
*sequential/spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*sequential/spiking_b_relu_2/GreaterEqual/y?
(sequential/spiking_b_relu_2/GreaterEqualGreaterEqual!sequential/dense/BiasAdd:output:03sequential/spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????x2*
(sequential/spiking_b_relu_2/GreaterEqual?
 sequential/spiking_b_relu_2/CastCast,sequential/spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????x2"
 sequential/spiking_b_relu_2/Cast?
*sequential/spiking_b_relu_2/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/spiking_b_relu_2/ReadVariableOp?
!sequential/spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!sequential/spiking_b_relu_2/sub/x?
sequential/spiking_b_relu_2/subSub*sequential/spiking_b_relu_2/sub/x:output:02sequential/spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu_2/sub?
!sequential/spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!sequential/spiking_b_relu_2/add/y?
sequential/spiking_b_relu_2/addAddV2#sequential/spiking_b_relu_2/sub:z:0*sequential/spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu_2/add?
%sequential/spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%sequential/spiking_b_relu_2/truediv/x?
#sequential/spiking_b_relu_2/truedivRealDiv.sequential/spiking_b_relu_2/truediv/x:output:0#sequential/spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: 2%
#sequential/spiking_b_relu_2/truediv?
#sequential/spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential/spiking_b_relu_2/sub_1/y?
!sequential/spiking_b_relu_2/sub_1Sub!sequential/dense/BiasAdd:output:0,sequential/spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:?????????x2#
!sequential/spiking_b_relu_2/sub_1?
sequential/spiking_b_relu_2/mulMul'sequential/spiking_b_relu_2/truediv:z:0%sequential/spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:?????????x2!
sequential/spiking_b_relu_2/mul?
#sequential/spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential/spiking_b_relu_2/add_1/y?
!sequential/spiking_b_relu_2/add_1AddV2#sequential/spiking_b_relu_2/mul:z:0,sequential/spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:?????????x2#
!sequential/spiking_b_relu_2/add_1?
3sequential/spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??25
3sequential/spiking_b_relu_2/clip_by_value/Minimum/y?
1sequential/spiking_b_relu_2/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_2/add_1:z:0<sequential/spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????x23
1sequential/spiking_b_relu_2/clip_by_value/Minimum?
+sequential/spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+sequential/spiking_b_relu_2/clip_by_value/y?
)sequential/spiking_b_relu_2/clip_by_valueMaximum5sequential/spiking_b_relu_2/clip_by_value/Minimum:z:04sequential/spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????x2+
)sequential/spiking_b_relu_2/clip_by_value?
0sequential/spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential/spiking_b_relu_2/Equal/ReadVariableOp?
#sequential/spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#sequential/spiking_b_relu_2/Equal/y?
!sequential/spiking_b_relu_2/EqualEqual8sequential/spiking_b_relu_2/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: 2#
!sequential/spiking_b_relu_2/Equal?
 sequential/spiking_b_relu_2/condStatelessIf%sequential/spiking_b_relu_2/Equal:z:0$sequential/spiking_b_relu_2/Cast:y:0-sequential/spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *?
else_branch0R.
,sequential_spiking_b_relu_2_cond_false_44585*&
output_shapes
:?????????x*>
then_branch/R-
+sequential_spiking_b_relu_2_cond_true_445842"
 sequential/spiking_b_relu_2/cond?
)sequential/spiking_b_relu_2/cond/IdentityIdentity)sequential/spiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:?????????x2+
)sequential/spiking_b_relu_2/cond/Identity?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul2sequential/spiking_b_relu_2/cond/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
sequential/dense_1/BiasAdd?
*sequential/spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*sequential/spiking_b_relu_3/GreaterEqual/y?
(sequential/spiking_b_relu_3/GreaterEqualGreaterEqual#sequential/dense_1/BiasAdd:output:03sequential/spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T2*
(sequential/spiking_b_relu_3/GreaterEqual?
 sequential/spiking_b_relu_3/CastCast,sequential/spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T2"
 sequential/spiking_b_relu_3/Cast?
*sequential/spiking_b_relu_3/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/spiking_b_relu_3/ReadVariableOp?
!sequential/spiking_b_relu_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!sequential/spiking_b_relu_3/sub/x?
sequential/spiking_b_relu_3/subSub*sequential/spiking_b_relu_3/sub/x:output:02sequential/spiking_b_relu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu_3/sub?
!sequential/spiking_b_relu_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!sequential/spiking_b_relu_3/add/y?
sequential/spiking_b_relu_3/addAddV2#sequential/spiking_b_relu_3/sub:z:0*sequential/spiking_b_relu_3/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu_3/add?
%sequential/spiking_b_relu_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%sequential/spiking_b_relu_3/truediv/x?
#sequential/spiking_b_relu_3/truedivRealDiv.sequential/spiking_b_relu_3/truediv/x:output:0#sequential/spiking_b_relu_3/add:z:0*
T0*
_output_shapes
: 2%
#sequential/spiking_b_relu_3/truediv?
#sequential/spiking_b_relu_3/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential/spiking_b_relu_3/sub_1/y?
!sequential/spiking_b_relu_3/sub_1Sub#sequential/dense_1/BiasAdd:output:0,sequential/spiking_b_relu_3/sub_1/y:output:0*
T0*'
_output_shapes
:?????????T2#
!sequential/spiking_b_relu_3/sub_1?
sequential/spiking_b_relu_3/mulMul'sequential/spiking_b_relu_3/truediv:z:0%sequential/spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:?????????T2!
sequential/spiking_b_relu_3/mul?
#sequential/spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential/spiking_b_relu_3/add_1/y?
!sequential/spiking_b_relu_3/add_1AddV2#sequential/spiking_b_relu_3/mul:z:0,sequential/spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:?????????T2#
!sequential/spiking_b_relu_3/add_1?
3sequential/spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??25
3sequential/spiking_b_relu_3/clip_by_value/Minimum/y?
1sequential/spiking_b_relu_3/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_3/add_1:z:0<sequential/spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????T23
1sequential/spiking_b_relu_3/clip_by_value/Minimum?
+sequential/spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+sequential/spiking_b_relu_3/clip_by_value/y?
)sequential/spiking_b_relu_3/clip_by_valueMaximum5sequential/spiking_b_relu_3/clip_by_value/Minimum:z:04sequential/spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????T2+
)sequential/spiking_b_relu_3/clip_by_value?
0sequential/spiking_b_relu_3/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential/spiking_b_relu_3/Equal/ReadVariableOp?
#sequential/spiking_b_relu_3/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#sequential/spiking_b_relu_3/Equal/y?
!sequential/spiking_b_relu_3/EqualEqual8sequential/spiking_b_relu_3/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_3/Equal/y:output:0*
T0*
_output_shapes
: 2#
!sequential/spiking_b_relu_3/Equal?
 sequential/spiking_b_relu_3/condStatelessIf%sequential/spiking_b_relu_3/Equal:z:0$sequential/spiking_b_relu_3/Cast:y:0-sequential/spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *?
else_branch0R.
,sequential_spiking_b_relu_3_cond_false_44626*&
output_shapes
:?????????T*>
then_branch/R-
+sequential_spiking_b_relu_3_cond_true_446252"
 sequential/spiking_b_relu_3/cond?
)sequential/spiking_b_relu_3/cond/IdentityIdentity)sequential/spiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:?????????T2+
)sequential/spiking_b_relu_3/cond/Identity?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMul2sequential/spiking_b_relu_3/cond/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential/dense_2/BiasAdd?
*sequential/spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*sequential/spiking_b_relu_4/GreaterEqual/y?
(sequential/spiking_b_relu_4/GreaterEqualGreaterEqual#sequential/dense_2/BiasAdd:output:03sequential/spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d2*
(sequential/spiking_b_relu_4/GreaterEqual?
 sequential/spiking_b_relu_4/CastCast,sequential/spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d2"
 sequential/spiking_b_relu_4/Cast?
*sequential/spiking_b_relu_4/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/spiking_b_relu_4/ReadVariableOp?
!sequential/spiking_b_relu_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!sequential/spiking_b_relu_4/sub/x?
sequential/spiking_b_relu_4/subSub*sequential/spiking_b_relu_4/sub/x:output:02sequential/spiking_b_relu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu_4/sub?
!sequential/spiking_b_relu_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!sequential/spiking_b_relu_4/add/y?
sequential/spiking_b_relu_4/addAddV2#sequential/spiking_b_relu_4/sub:z:0*sequential/spiking_b_relu_4/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/spiking_b_relu_4/add?
%sequential/spiking_b_relu_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%sequential/spiking_b_relu_4/truediv/x?
#sequential/spiking_b_relu_4/truedivRealDiv.sequential/spiking_b_relu_4/truediv/x:output:0#sequential/spiking_b_relu_4/add:z:0*
T0*
_output_shapes
: 2%
#sequential/spiking_b_relu_4/truediv?
#sequential/spiking_b_relu_4/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential/spiking_b_relu_4/sub_1/y?
!sequential/spiking_b_relu_4/sub_1Sub#sequential/dense_2/BiasAdd:output:0,sequential/spiking_b_relu_4/sub_1/y:output:0*
T0*'
_output_shapes
:?????????d2#
!sequential/spiking_b_relu_4/sub_1?
sequential/spiking_b_relu_4/mulMul'sequential/spiking_b_relu_4/truediv:z:0%sequential/spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:?????????d2!
sequential/spiking_b_relu_4/mul?
#sequential/spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential/spiking_b_relu_4/add_1/y?
!sequential/spiking_b_relu_4/add_1AddV2#sequential/spiking_b_relu_4/mul:z:0,sequential/spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:?????????d2#
!sequential/spiking_b_relu_4/add_1?
3sequential/spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??25
3sequential/spiking_b_relu_4/clip_by_value/Minimum/y?
1sequential/spiking_b_relu_4/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_4/add_1:z:0<sequential/spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????d23
1sequential/spiking_b_relu_4/clip_by_value/Minimum?
+sequential/spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+sequential/spiking_b_relu_4/clip_by_value/y?
)sequential/spiking_b_relu_4/clip_by_valueMaximum5sequential/spiking_b_relu_4/clip_by_value/Minimum:z:04sequential/spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????d2+
)sequential/spiking_b_relu_4/clip_by_value?
0sequential/spiking_b_relu_4/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential/spiking_b_relu_4/Equal/ReadVariableOp?
#sequential/spiking_b_relu_4/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#sequential/spiking_b_relu_4/Equal/y?
!sequential/spiking_b_relu_4/EqualEqual8sequential/spiking_b_relu_4/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_4/Equal/y:output:0*
T0*
_output_shapes
: 2#
!sequential/spiking_b_relu_4/Equal?
 sequential/spiking_b_relu_4/condStatelessIf%sequential/spiking_b_relu_4/Equal:z:0$sequential/spiking_b_relu_4/Cast:y:0-sequential/spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *?
else_branch0R.
,sequential_spiking_b_relu_4_cond_false_44667*&
output_shapes
:?????????d*>
then_branch/R-
+sequential_spiking_b_relu_4_cond_true_446662"
 sequential/spiking_b_relu_4/cond?
)sequential/spiking_b_relu_4/cond/IdentityIdentity)sequential/spiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:?????????d2+
)sequential/spiking_b_relu_4/cond/Identity?
 sequential/softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 sequential/softmax__decode/mul/x?
sequential/softmax__decode/mulMul)sequential/softmax__decode/mul/x:output:02sequential/spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:?????????d2 
sequential/softmax__decode/mul?
 sequential/softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 sequential/softmax__decode/sub/y?
sequential/softmax__decode/subSub"sequential/softmax__decode/mul:z:0)sequential/softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:?????????d2 
sequential/softmax__decode/sub?
0sequential/softmax__decode/MatMul/ReadVariableOpReadVariableOp9sequential_softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype022
0sequential/softmax__decode/MatMul/ReadVariableOp?
!sequential/softmax__decode/MatMulMatMul"sequential/softmax__decode/sub:z:08sequential/softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2#
!sequential/softmax__decode/MatMul?
"sequential/softmax__decode/SoftmaxSoftmax+sequential/softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:?????????
2$
"sequential/softmax__decode/Softmax?
IdentityIdentity,sequential/softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp1^sequential/softmax__decode/MatMul/ReadVariableOp/^sequential/spiking_b_relu/Equal/ReadVariableOp)^sequential/spiking_b_relu/ReadVariableOp1^sequential/spiking_b_relu_1/Equal/ReadVariableOp+^sequential/spiking_b_relu_1/ReadVariableOp1^sequential/spiking_b_relu_2/Equal/ReadVariableOp+^sequential/spiking_b_relu_2/ReadVariableOp1^sequential/spiking_b_relu_3/Equal/ReadVariableOp+^sequential/spiking_b_relu_3/ReadVariableOp1^sequential/spiking_b_relu_4/Equal/ReadVariableOp+^sequential/spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2d
0sequential/softmax__decode/MatMul/ReadVariableOp0sequential/softmax__decode/MatMul/ReadVariableOp2`
.sequential/spiking_b_relu/Equal/ReadVariableOp.sequential/spiking_b_relu/Equal/ReadVariableOp2T
(sequential/spiking_b_relu/ReadVariableOp(sequential/spiking_b_relu/ReadVariableOp2d
0sequential/spiking_b_relu_1/Equal/ReadVariableOp0sequential/spiking_b_relu_1/Equal/ReadVariableOp2X
*sequential/spiking_b_relu_1/ReadVariableOp*sequential/spiking_b_relu_1/ReadVariableOp2d
0sequential/spiking_b_relu_2/Equal/ReadVariableOp0sequential/spiking_b_relu_2/Equal/ReadVariableOp2X
*sequential/spiking_b_relu_2/ReadVariableOp*sequential/spiking_b_relu_2/ReadVariableOp2d
0sequential/spiking_b_relu_3/Equal/ReadVariableOp0sequential/spiking_b_relu_3/Equal/ReadVariableOp2X
*sequential/spiking_b_relu_3/ReadVariableOp*sequential/spiking_b_relu_3/ReadVariableOp2d
0sequential/spiking_b_relu_4/Equal/ReadVariableOp0sequential/spiking_b_relu_4/Equal/ReadVariableOp2X
*sequential/spiking_b_relu_4/ReadVariableOp*sequential/spiking_b_relu_4/ReadVariableOp:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_44813

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

02	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????

02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
\
cond_false_46194
cond_placeholder
cond_identity_clip_by_value
cond_identity?
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:?????????

02
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????

0:?????????

0:5 1
/
_output_shapes
:?????????

0:51
/
_output_shapes
:?????????

0
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_46069

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
 spiking_b_relu_4_cond_true_457438
4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast%
!spiking_b_relu_4_cond_placeholder"
spiking_b_relu_4_cond_identity?
spiking_b_relu_4/cond/IdentityIdentity4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast*
T0*'
_output_shapes
:?????????d2 
spiking_b_relu_4/cond/Identity"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
?
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_46371

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T2
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/ya
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:?????????T2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:?????????T2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yd
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:?????????T2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????T2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????T2
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_46359*&
output_shapes
:?????????T*"
then_branchR
cond_true_463582
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:?????????T2
cond/Identityq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:?????????T2

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????T: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46131

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46136

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_44931

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????x2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????x2
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/ya
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:?????????x2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:?????????x2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yd
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:?????????x2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????x2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????x2
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_44919*&
output_shapes
:?????????x*"
then_branchR
cond_true_449182
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:?????????x2
cond/Identityq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:?????????x2

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_46228

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_447182
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_softmax__decode_layer_call_fn_46464

inputs
unknown:d

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_450652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
spiking_b_relu_cond_true_457964
0spiking_b_relu_cond_identity_spiking_b_relu_cast#
spiking_b_relu_cond_placeholder 
spiking_b_relu_cond_identity?
spiking_b_relu/cond/IdentityIdentity0spiking_b_relu_cond_identity_spiking_b_relu_cast*
T0*/
_output_shapes
:????????? 2
spiking_b_relu/cond/Identity"E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?
?
*sequential_spiking_b_relu_cond_false_44499.
*sequential_spiking_b_relu_cond_placeholderS
Osequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_clip_by_value+
'sequential_spiking_b_relu_cond_identity?
'sequential/spiking_b_relu/cond/IdentityIdentityOsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:????????? 2)
'sequential/spiking_b_relu/cond/Identity"[
'sequential_spiking_b_relu_cond_identity0sequential/spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?

?
J__inference_softmax__decode_layer_call_and_return_conditional_losses_45065

inputs0
matmul_readvariableop_resource:d

identity??MatMul/ReadVariableOpS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/x[
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:?????????d2
mulS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/y\
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:?????????d2
sub?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:?????????
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
R
cond_true_44780
cond_identity_cast
cond_placeholder
cond_identityx
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:????????? 2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?E
?
E__inference_sequential_layer_call_and_return_conditional_losses_45430
conv2d_input&
conv2d_45383: 
conv2d_45385: 
spiking_b_relu_45388: (
conv2d_1_45392: 0
conv2d_1_45394:0 
spiking_b_relu_1_45397: 
dense_45402:	?	x
dense_45404:x 
spiking_b_relu_2_45407: 
dense_1_45410:xT
dense_1_45412:T 
spiking_b_relu_3_45415: 
dense_2_45418:Td
dense_2_45420:d 
spiking_b_relu_4_45423: '
softmax__decode_45426:d

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?'softmax__decode/StatefulPartitionedCall?&spiking_b_relu/StatefulPartitionedCall?(spiking_b_relu_1/StatefulPartitionedCall?(spiking_b_relu_2/StatefulPartitionedCall?(spiking_b_relu_3/StatefulPartitionedCall?(spiking_b_relu_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_45383conv2d_45385*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_447482 
conv2d/StatefulPartitionedCall?
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0spiking_b_relu_45388*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_447932(
&spiking_b_relu/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_448012
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_45392conv2d_1_45394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_448132"
 conv2d_1/StatefulPartitionedCall?
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0spiking_b_relu_1_45397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_448582*
(spiking_b_relu_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_448662!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_448742
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_45402dense_45404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_448862
dense/StatefulPartitionedCall?
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_2_45407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_449312*
(spiking_b_relu_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0dense_1_45410dense_1_45412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_449452!
dense_1/StatefulPartitionedCall?
(spiking_b_relu_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_3_45415*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_449902*
(spiking_b_relu_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_3/StatefulPartitionedCall:output:0dense_2_45418dense_2_45420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_450042!
dense_2/StatefulPartitionedCall?
(spiking_b_relu_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_4_45423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_450492*
(spiking_b_relu_4/StatefulPartitionedCall?
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_4/StatefulPartitionedCall:output:0softmax__decode_45426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_450652)
'softmax__decode/StatefulPartitionedCall?
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall2T
(spiking_b_relu_3/StatefulPartitionedCall(spiking_b_relu_3/StatefulPartitionedCall2T
(spiking_b_relu_4/StatefulPartitionedCall(spiking_b_relu_4/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
R
cond_true_46291
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:?????????x2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x
?
?
,sequential_spiking_b_relu_4_cond_false_446670
,sequential_spiking_b_relu_4_cond_placeholderW
Ssequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_clip_by_value-
)sequential_spiking_b_relu_4_cond_identity?
)sequential/spiking_b_relu_4/cond/IdentityIdentitySsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:?????????d2+
)sequential/spiking_b_relu_4/cond/Identity"_
)sequential_spiking_b_relu_4_cond_identity2sequential/spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_44718

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
\
cond_false_44781
cond_placeholder
cond_identity_clip_by_value
cond_identity?
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:????????? 2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?E
?
E__inference_sequential_layer_call_and_return_conditional_losses_45308

inputs&
conv2d_45261: 
conv2d_45263: 
spiking_b_relu_45266: (
conv2d_1_45270: 0
conv2d_1_45272:0 
spiking_b_relu_1_45275: 
dense_45280:	?	x
dense_45282:x 
spiking_b_relu_2_45285: 
dense_1_45288:xT
dense_1_45290:T 
spiking_b_relu_3_45293: 
dense_2_45296:Td
dense_2_45298:d 
spiking_b_relu_4_45301: '
softmax__decode_45304:d

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?'softmax__decode/StatefulPartitionedCall?&spiking_b_relu/StatefulPartitionedCall?(spiking_b_relu_1/StatefulPartitionedCall?(spiking_b_relu_2/StatefulPartitionedCall?(spiking_b_relu_3/StatefulPartitionedCall?(spiking_b_relu_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_45261conv2d_45263*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_447482 
conv2d/StatefulPartitionedCall?
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0spiking_b_relu_45266*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_447932(
&spiking_b_relu/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_448012
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_45270conv2d_1_45272*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_448132"
 conv2d_1/StatefulPartitionedCall?
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0spiking_b_relu_1_45275*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_448582*
(spiking_b_relu_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_448662!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_448742
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_45280dense_45282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_448862
dense/StatefulPartitionedCall?
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_2_45285*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_449312*
(spiking_b_relu_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0dense_1_45288dense_1_45290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_449452!
dense_1/StatefulPartitionedCall?
(spiking_b_relu_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_3_45293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_449902*
(spiking_b_relu_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_3/StatefulPartitionedCall:output:0dense_2_45296dense_2_45298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_450042!
dense_2/StatefulPartitionedCall?
(spiking_b_relu_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_4_45301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_450492*
(spiking_b_relu_4/StatefulPartitionedCall?
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_4/StatefulPartitionedCall:output:0softmax__decode_45304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_450652)
'softmax__decode/StatefulPartitionedCall?
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall2T
(spiking_b_relu_3/StatefulPartitionedCall(spiking_b_relu_3/StatefulPartitionedCall2T
(spiking_b_relu_4/StatefulPartitionedCall(spiking_b_relu_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_44793

inputs!
readvariableop_resource: 
identity??Equal/ReadVariableOp?ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y?
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
GreaterEqualo
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
Castp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xZ
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: 2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yM
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: 2
add[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	truediv/x[
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: 2	
truedivW
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
sub_1/yi
sub_1Subinputssub_1/y:output:0*
T0*/
_output_shapes
:????????? 2
sub_1c
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:????????? 2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/yl
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:????????? 2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:????????? 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:????????? 2
clip_by_value|
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
Equal/ReadVariableOpW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal?
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 *#
else_branchR
cond_false_44781*.
output_shapes
:????????? *"
then_branchR
cond_true_447802
conds
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:????????? 2
cond/Identityy
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:????????? 2

Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? : 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_46059

inputs!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	?	x
	unknown_6:x
	unknown_7: 
	unknown_8:xT
	unknown_9:T

unknown_10: 

unknown_11:Td

unknown_12:d

unknown_13: 

unknown_14:d

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_453082
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
R
cond_true_46106
cond_identity_cast
cond_placeholder
cond_identityx
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:????????? 2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:????????? :????????? :5 1
/
_output_shapes
:????????? :51
/
_output_shapes
:????????? 
?
?
%__inference_dense_layer_call_fn_46263

inputs
unknown:	?	x
	unknown_0:x
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_448862
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????x2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
\
cond_false_45037
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:?????????d2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????d:?????????d:- )
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d
?
?
+sequential_spiking_b_relu_2_cond_true_44584N
Jsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_cast0
,sequential_spiking_b_relu_2_cond_placeholder-
)sequential_spiking_b_relu_2_cond_identity?
)sequential/spiking_b_relu_2/cond/IdentityIdentityJsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_cast*
T0*'
_output_shapes
:?????????x2+
)sequential/spiking_b_relu_2/cond/Identity"_
)sequential_spiking_b_relu_2_cond_identity2sequential/spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????x:?????????x:- )
'
_output_shapes
:?????????x:-)
'
_output_shapes
:?????????x"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
conv2d_input=
serving_default_conv2d_input:0?????????C
softmax__decode0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_sequential
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	sharpness
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
*	sharpness
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
/	variables
0regularization_losses
1trainable_variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
3	variables
4regularization_losses
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
=	sharpness
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Bkernel
Cbias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
H	sharpness
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Mkernel
Nbias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
S	sharpness
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
X_rescaled_key
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
]iter
	^decay
_learning_rate
`rho
accum_grad?
accum_grad?
accum_grad?$
accum_grad?%
accum_grad?*
accum_grad?7
accum_grad?8
accum_grad?=
accum_grad?B
accum_grad?C
accum_grad?H
accum_grad?M
accum_grad?N
accum_grad?S
accum_grad?	accum_var?	accum_var?	accum_var?$	accum_var?%	accum_var?*	accum_var?7	accum_var?8	accum_var?=	accum_var?B	accum_var?C	accum_var?H	accum_var?M	accum_var?N	accum_var?S	accum_var?"
	optimizer
?
0
1
2
$3
%4
*5
76
87
=8
B9
C10
H11
M12
N13
S14
X15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
$3
%4
*5
76
87
=8
B9
C10
H11
M12
N13
S14"
trackable_list_wrapper
?

alayers
bmetrics
cnon_trainable_variables
dlayer_metrics
	variables
regularization_losses
trainable_variables
elayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

flayers
gmetrics
hnon_trainable_variables
ilayer_metrics
	variables
regularization_losses
trainable_variables
jlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?

klayers
lmetrics
mnon_trainable_variables
nlayer_metrics
	variables
regularization_losses
trainable_variables
olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

players
qmetrics
rnon_trainable_variables
slayer_metrics
 	variables
!regularization_losses
"trainable_variables
tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_1/kernel
:02conv2d_1/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?

ulayers
vmetrics
wnon_trainable_variables
xlayer_metrics
&	variables
'regularization_losses
(trainable_variables
ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
?

zlayers
{metrics
|non_trainable_variables
}layer_metrics
+	variables
,regularization_losses
-trainable_variables
~layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

layers
?metrics
?non_trainable_variables
?layer_metrics
/	variables
0regularization_losses
1trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
3	variables
4regularization_losses
5trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?	x2dense/kernel
:x2
dense/bias
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
9	variables
:regularization_losses
;trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
>	variables
?regularization_losses
@trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :xT2dense_1/kernel
:T2dense_1/bias
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
D	variables
Eregularization_losses
Ftrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
I	variables
Jregularization_losses
Ktrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :Td2dense_2/kernel
:d2dense_2/bias
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
O	variables
Pregularization_losses
Qtrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
S0"
trackable_list_wrapper
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
T	variables
Uregularization_losses
Vtrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:d
2Variable
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
?non_trainable_variables
?layer_metrics
Y	variables
Zregularization_losses
[trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
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
13"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
'
X0"
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
'
X0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
9:7 2!Adadelta/conv2d/kernel/accum_grad
+:) 2Adadelta/conv2d/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
;:9 02#Adadelta/conv2d_1/kernel/accum_grad
-:+02!Adadelta/conv2d_1/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
1:/	?	x2 Adadelta/dense/kernel/accum_grad
*:(x2Adadelta/dense/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
2:0xT2"Adadelta/dense_1/kernel/accum_grad
,:*T2 Adadelta/dense_1/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
2:0Td2"Adadelta/dense_2/kernel/accum_grad
,:*d2 Adadelta/dense_2/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
8:6 2 Adadelta/conv2d/kernel/accum_var
*:( 2Adadelta/conv2d/bias/accum_var
#:! 2Adadelta/Variable/accum_var
::8 02"Adadelta/conv2d_1/kernel/accum_var
,:*02 Adadelta/conv2d_1/bias/accum_var
#:! 2Adadelta/Variable/accum_var
0:.	?	x2Adadelta/dense/kernel/accum_var
):'x2Adadelta/dense/bias/accum_var
#:! 2Adadelta/Variable/accum_var
1:/xT2!Adadelta/dense_1/kernel/accum_var
+:)T2Adadelta/dense_1/bias/accum_var
#:! 2Adadelta/Variable/accum_var
1:/Td2!Adadelta/dense_2/kernel/accum_var
+:)d2Adadelta/dense_2/bias/accum_var
#:! 2Adadelta/Variable/accum_var
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_45764
E__inference_sequential_layer_call_and_return_conditional_losses_45985
E__inference_sequential_layer_call_and_return_conditional_losses_45430
E__inference_sequential_layer_call_and_return_conditional_losses_45480?
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
 __inference__wrapped_model_44687conv2d_input"?
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
?2?
*__inference_sequential_layer_call_fn_45105
*__inference_sequential_layer_call_fn_46022
*__inference_sequential_layer_call_fn_46059
*__inference_sequential_layer_call_fn_45380?
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
A__inference_conv2d_layer_call_and_return_conditional_losses_46069?
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
&__inference_conv2d_layer_call_fn_46078?
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
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_46119?
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
.__inference_spiking_b_relu_layer_call_fn_46126?
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46131
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46136?
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
-__inference_max_pooling2d_layer_call_fn_46141
-__inference_max_pooling2d_layer_call_fn_46146?
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_46156?
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
(__inference_conv2d_1_layer_call_fn_46165?
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
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_46206?
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
0__inference_spiking_b_relu_1_layer_call_fn_46213?
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46218
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46223?
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
/__inference_max_pooling2d_1_layer_call_fn_46228
/__inference_max_pooling2d_1_layer_call_fn_46233?
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
B__inference_flatten_layer_call_and_return_conditional_losses_46239?
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
'__inference_flatten_layer_call_fn_46244?
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
@__inference_dense_layer_call_and_return_conditional_losses_46254?
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
%__inference_dense_layer_call_fn_46263?
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
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_46304?
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
0__inference_spiking_b_relu_2_layer_call_fn_46311?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_46321?
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
'__inference_dense_1_layer_call_fn_46330?
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
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_46371?
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
0__inference_spiking_b_relu_3_layer_call_fn_46378?
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
B__inference_dense_2_layer_call_and_return_conditional_losses_46388?
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
'__inference_dense_2_layer_call_fn_46397?
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
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_46438?
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
0__inference_spiking_b_relu_4_layer_call_fn_46445?
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
J__inference_softmax__decode_layer_call_and_return_conditional_losses_46457?
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
/__inference_softmax__decode_layer_call_fn_46464?
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
#__inference_signature_wrapper_45543conv2d_input"?
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
 ?
 __inference__wrapped_model_44687?$%*78=BCHMNSX=?:
3?0
.?+
conv2d_input?????????
? "A?>
<
softmax__decode)?&
softmax__decode?????????
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_46156l$%7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????

0
? ?
(__inference_conv2d_1_layer_call_fn_46165_$%7?4
-?*
(?%
inputs????????? 
? " ??????????

0?
A__inference_conv2d_layer_call_and_return_conditional_losses_46069l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
&__inference_conv2d_layer_call_fn_46078_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
B__inference_dense_1_layer_call_and_return_conditional_losses_46321\BC/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????T
? z
'__inference_dense_1_layer_call_fn_46330OBC/?,
%?"
 ?
inputs?????????x
? "??????????T?
B__inference_dense_2_layer_call_and_return_conditional_losses_46388\MN/?,
%?"
 ?
inputs?????????T
? "%?"
?
0?????????d
? z
'__inference_dense_2_layer_call_fn_46397OMN/?,
%?"
 ?
inputs?????????T
? "??????????d?
@__inference_dense_layer_call_and_return_conditional_losses_46254]780?-
&?#
!?
inputs??????????	
? "%?"
?
0?????????x
? y
%__inference_dense_layer_call_fn_46263P780?-
&?#
!?
inputs??????????	
? "??????????x?
B__inference_flatten_layer_call_and_return_conditional_losses_46239a7?4
-?*
(?%
inputs?????????0
? "&?#
?
0??????????	
? 
'__inference_flatten_layer_call_fn_46244T7?4
-?*
(?%
inputs?????????0
? "???????????	?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46218?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46223h7?4
-?*
(?%
inputs?????????

0
? "-?*
#? 
0?????????0
? ?
/__inference_max_pooling2d_1_layer_call_fn_46228?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
/__inference_max_pooling2d_1_layer_call_fn_46233[7?4
-?*
(?%
inputs?????????

0
? " ??????????0?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46131?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46136h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
-__inference_max_pooling2d_layer_call_fn_46141?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
-__inference_max_pooling2d_layer_call_fn_46146[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
E__inference_sequential_layer_call_and_return_conditional_losses_45430?$%*78=BCHMNSXE?B
;?8
.?+
conv2d_input?????????
p 

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_45480?$%*78=BCHMNSXE?B
;?8
.?+
conv2d_input?????????
p

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_45764z$%*78=BCHMNSX??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_45985z$%*78=BCHMNSX??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
*__inference_sequential_layer_call_fn_45105s$%*78=BCHMNSXE?B
;?8
.?+
conv2d_input?????????
p 

 
? "??????????
?
*__inference_sequential_layer_call_fn_45380s$%*78=BCHMNSXE?B
;?8
.?+
conv2d_input?????????
p

 
? "??????????
?
*__inference_sequential_layer_call_fn_46022m$%*78=BCHMNSX??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
*__inference_sequential_layer_call_fn_46059m$%*78=BCHMNSX??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
#__inference_signature_wrapper_45543?$%*78=BCHMNSXM?J
? 
C?@
>
conv2d_input.?+
conv2d_input?????????"A?>
<
softmax__decode)?&
softmax__decode?????????
?
J__inference_softmax__decode_layer_call_and_return_conditional_losses_46457[X/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????

? ?
/__inference_softmax__decode_layer_call_fn_46464NX/?,
%?"
 ?
inputs?????????d
? "??????????
?
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_46206k*7?4
-?*
(?%
inputs?????????

0
? "-?*
#? 
0?????????

0
? ?
0__inference_spiking_b_relu_1_layer_call_fn_46213^*7?4
-?*
(?%
inputs?????????

0
? " ??????????

0?
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_46304[=/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????x
? ?
0__inference_spiking_b_relu_2_layer_call_fn_46311N=/?,
%?"
 ?
inputs?????????x
? "??????????x?
K__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_46371[H/?,
%?"
 ?
inputs?????????T
? "%?"
?
0?????????T
? ?
0__inference_spiking_b_relu_3_layer_call_fn_46378NH/?,
%?"
 ?
inputs?????????T
? "??????????T?
K__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_46438[S/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????d
? ?
0__inference_spiking_b_relu_4_layer_call_fn_46445NS/?,
%?"
 ?
inputs?????????d
? "??????????d?
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_46119k7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_spiking_b_relu_layer_call_fn_46126^7?4
-?*
(?%
inputs????????? 
? " ?????????? 