є
їШ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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
incompatible_shape_errorbool(
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

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
delete_old_dirsbool(
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
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
С
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
executor_typestring Ј
Р
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Эю
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

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
shape:	А	x*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	А	x*
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

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
І
!Adadelta/conv2d/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d/kernel/accum_grad

5Adadelta/conv2d/kernel/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d/kernel/accum_grad*&
_output_shapes
: *
dtype0

Adadelta/conv2d/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adadelta/conv2d/bias/accum_grad

3Adadelta/conv2d/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/conv2d/bias/accum_grad*
_output_shapes
: *
dtype0

Adadelta/Variable/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdadelta/Variable/accum_grad

0Adadelta/Variable/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad*
_output_shapes
: *
dtype0
Њ
#Adadelta/conv2d_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*4
shared_name%#Adadelta/conv2d_1/kernel/accum_grad
Ѓ
7Adadelta/conv2d_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_1/kernel/accum_grad*&
_output_shapes
: 0*
dtype0

!Adadelta/conv2d_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adadelta/conv2d_1/bias/accum_grad

5Adadelta/conv2d_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_1/bias/accum_grad*
_output_shapes
:0*
dtype0

Adadelta/Variable/accum_grad_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_1

2Adadelta/Variable/accum_grad_1/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_1*
_output_shapes
: *
dtype0

 Adadelta/dense/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А	x*1
shared_name" Adadelta/dense/kernel/accum_grad

4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense/kernel/accum_grad*
_output_shapes
:	А	x*
dtype0

Adadelta/dense/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*/
shared_name Adadelta/dense/bias/accum_grad

2Adadelta/dense/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_grad*
_output_shapes
:x*
dtype0

Adadelta/Variable/accum_grad_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_2

2Adadelta/Variable/accum_grad_2/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_2*
_output_shapes
: *
dtype0
 
"Adadelta/dense_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*3
shared_name$"Adadelta/dense_1/kernel/accum_grad

6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_1/kernel/accum_grad*
_output_shapes

:xT*
dtype0

 Adadelta/dense_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*1
shared_name" Adadelta/dense_1/bias/accum_grad

4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_1/bias/accum_grad*
_output_shapes
:T*
dtype0

Adadelta/Variable/accum_grad_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_3

2Adadelta/Variable/accum_grad_3/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_3*
_output_shapes
: *
dtype0
 
"Adadelta/dense_2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Td*3
shared_name$"Adadelta/dense_2/kernel/accum_grad

6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_2/kernel/accum_grad*
_output_shapes

:Td*
dtype0

 Adadelta/dense_2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*1
shared_name" Adadelta/dense_2/bias/accum_grad

4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_2/bias/accum_grad*
_output_shapes
:d*
dtype0

Adadelta/Variable/accum_grad_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_4

2Adadelta/Variable/accum_grad_4/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_4*
_output_shapes
: *
dtype0
Є
 Adadelta/conv2d/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adadelta/conv2d/kernel/accum_var

4Adadelta/conv2d/kernel/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/conv2d/kernel/accum_var*&
_output_shapes
: *
dtype0

Adadelta/conv2d/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/conv2d/bias/accum_var

2Adadelta/conv2d/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv2d/bias/accum_var*
_output_shapes
: *
dtype0

Adadelta/Variable/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdadelta/Variable/accum_var

/Adadelta/Variable/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var*
_output_shapes
: *
dtype0
Ј
"Adadelta/conv2d_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*3
shared_name$"Adadelta/conv2d_1/kernel/accum_var
Ё
6Adadelta/conv2d_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_1/kernel/accum_var*&
_output_shapes
: 0*
dtype0

 Adadelta/conv2d_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Adadelta/conv2d_1/bias/accum_var

4Adadelta/conv2d_1/bias/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/conv2d_1/bias/accum_var*
_output_shapes
:0*
dtype0

Adadelta/Variable/accum_var_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_1

1Adadelta/Variable/accum_var_1/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_1*
_output_shapes
: *
dtype0

Adadelta/dense/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А	x*0
shared_name!Adadelta/dense/kernel/accum_var

3Adadelta/dense/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/kernel/accum_var*
_output_shapes
:	А	x*
dtype0

Adadelta/dense/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*.
shared_nameAdadelta/dense/bias/accum_var

1Adadelta/dense/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_var*
_output_shapes
:x*
dtype0

Adadelta/Variable/accum_var_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_2

1Adadelta/Variable/accum_var_2/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_2*
_output_shapes
: *
dtype0

!Adadelta/dense_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*2
shared_name#!Adadelta/dense_1/kernel/accum_var

5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_1/kernel/accum_var*
_output_shapes

:xT*
dtype0

Adadelta/dense_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*0
shared_name!Adadelta/dense_1/bias/accum_var

3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_1/bias/accum_var*
_output_shapes
:T*
dtype0

Adadelta/Variable/accum_var_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_3

1Adadelta/Variable/accum_var_3/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_3*
_output_shapes
: *
dtype0

!Adadelta/dense_2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Td*2
shared_name#!Adadelta/dense_2/kernel/accum_var

5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_2/kernel/accum_var*
_output_shapes

:Td*
dtype0

Adadelta/dense_2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adadelta/dense_2/bias/accum_var

3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_2/bias/accum_var*
_output_shapes
:d*
dtype0

Adadelta/Variable/accum_var_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_4

1Adadelta/Variable/accum_var_4/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_4*
_output_shapes
: *
dtype0

NoOpNoOp
~
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*к}
valueа}BЭ} BЦ}
Ѕ
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
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
І

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

 	sharpness
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*

'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
І

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*

5	sharpness
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*

<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 

B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
І

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses*

P	sharpness
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses*
І

Wkernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses*

_	sharpness
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses*
І

fkernel
gbias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*

n	sharpness
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses*
Ѓ
u_rescaled_key
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses*
р
|iter
	}decay
~learning_rate
rho
accum_gradз
accum_gradи 
accum_gradй-
accum_gradк.
accum_gradл5
accum_gradмH
accum_gradнI
accum_gradоP
accum_gradпW
accum_gradрX
accum_gradс_
accum_gradтf
accum_gradуg
accum_gradфn
accum_gradх	accum_varц	accum_varч 	accum_varш-	accum_varщ.	accum_varъ5	accum_varыH	accum_varьI	accum_varэP	accum_varюW	accum_varяX	accum_var№_	accum_varёf	accum_varђg	accum_varѓn	accum_varє*
z
0
1
 2
-3
.4
55
H6
I7
P8
W9
X10
_11
f12
g13
n14
u15*
r
0
1
 2
-3
.4
55
H6
I7
P8
W9
X10
_11
f12
g13
n14*
* 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
[U
VARIABLE_VALUEVariable9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUE*

 0*

 0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUE
Variable_19layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUE*

50*

50*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

H0
I1*
* 

Љnon_trainable_variables
Њlayers
Ћmetrics
 Ќlayer_regularization_losses
­layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUE
Variable_29layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUE*

P0*

P0*
* 

Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

W0
X1*
* 

Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUE
Variable_39layer_with_weights-7/sharpness/.ATTRIBUTES/VARIABLE_VALUE*

_0*

_0*
* 

Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

f0
g1*

f0
g1*
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUE
Variable_49layer_with_weights-9/sharpness/.ATTRIBUTES/VARIABLE_VALUE*

n0*

n0*
* 

Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE
Variable_5>layer_with_weights-10/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUE*

u0*
* 
* 

Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*
* 
* 
PJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*

u0*
j
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
13*

Ь0
Э1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

u0*
* 
* 
* 
* 
<

Юtotal

Яcount
а	variables
б	keras_api*
M

вtotal

гcount
д
_fn_kwargs
е	variables
ж	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ю0
Я1*

а	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

в0
г1*

е	variables*

VARIABLE_VALUE!Adadelta/conv2d/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/conv2d/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_grad^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adadelta/conv2d_1/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adadelta/conv2d_1/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_grad_1^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adadelta/dense/kernel/accum_grad[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/dense/bias/accum_gradYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_grad_2^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adadelta/dense_1/kernel/accum_grad[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adadelta/dense_1/bias/accum_gradYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_grad_3^layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adadelta/dense_2/kernel/accum_grad[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adadelta/dense_2/bias/accum_gradYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_grad_4^layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adadelta/conv2d/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/conv2d/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_var]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adadelta/conv2d_1/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adadelta/conv2d_1/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_var_1]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/dense/kernel/accum_varZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/dense/bias/accum_varXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_var_2]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adadelta/dense_1/kernel/accum_varZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/dense_1/bias/accum_varXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_var_3]layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adadelta/dense_2/kernel/accum_varZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/dense_2/bias/accum_varXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdadelta/Variable/accum_var_4]layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_conv2d_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
Џ
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
:џџџџџџџџџ
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_192464
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ї
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
GPU 2J 8 *(
f#R!
__inference__traced_save_193034
ъ
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_193206Ди
Ђ
S
cond_true_192817
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:џџџџџџџџџd"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd
с
В
"spiking_b_relu_2_cond_false_192323%
!spiking_b_relu_2_cond_placeholderA
=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value"
spiking_b_relu_2_cond_identity
spiking_b_relu_2/cond/IdentityIdentity=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџx"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx
Ђ
S
cond_true_191451
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:џџџџџџџџџd"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd
Ю
Ј
!spiking_b_relu_4_cond_true_1921838
4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast%
!spiking_b_relu_4_cond_placeholder"
spiking_b_relu_4_cond_identity
spiking_b_relu_4/cond/IdentityIdentity4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast*
T0*'
_output_shapes
:џџџџџџџџџd"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd
Ј

§
D__inference_conv2d_1_layer_call_and_return_conditional_losses_191233

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ

0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ц	
є
C__inference_dense_2_layer_call_and_return_conditional_losses_191419

inputs0
matmul_readvariableop_resource:Td-
biasadd_readvariableop_resource:d
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Td*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџT
 
_user_specified_nameinputs
Ё

+__inference_sequential_layer_call_fn_191983

inputs!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	А	x
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
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_191713o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г

+__inference_sequential_layer_call_fn_191785
conv2d_input!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	А	x
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
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_191713o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ю
Ј
!spiking_b_relu_4_cond_true_1924048
4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast%
!spiking_b_relu_4_cond_placeholder"
spiking_b_relu_4_cond_identity
spiking_b_relu_4/cond/IdentityIdentity4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast*
T0*'
_output_shapes
:џџџџџџџџџd"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd
БA
Ж
F__inference_sequential_layer_call_and_return_conditional_losses_191885
conv2d_input'
conv2d_191838: 
conv2d_191840: 
spiking_b_relu_191843: )
conv2d_1_191847: 0
conv2d_1_191849:0!
spiking_b_relu_1_191852: 
dense_191857:	А	x
dense_191859:x!
spiking_b_relu_2_191862:  
dense_1_191865:xT
dense_1_191867:T!
spiking_b_relu_3_191870:  
dense_2_191873:Td
dense_2_191875:d!
spiking_b_relu_4_191878: (
softmax__decode_191881:d

identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ'softmax__decode/StatefulPartitionedCallЂ&spiking_b_relu/StatefulPartitionedCallЂ(spiking_b_relu_1/StatefulPartitionedCallЂ(spiking_b_relu_2/StatefulPartitionedCallЂ(spiking_b_relu_3/StatefulPartitionedCallЂ(spiking_b_relu_4/StatefulPartitionedCallі
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_191838conv2d_191840*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_191173
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0spiking_b_relu_191843*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218ѓ
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_191141
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_191847conv2d_1_191849*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_191233 
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0spiking_b_relu_1_191852*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278љ
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153й
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџА	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289ў
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_191857dense_191859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_191301
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_2_191862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_191346
dense_1/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0dense_1_191865dense_1_191867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_191360
(spiking_b_relu_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_3_191870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_191405
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_3/StatefulPartitionedCall:output:0dense_2_191873dense_2_191875*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_191419
(spiking_b_relu_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_4_191878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_191464
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_4/StatefulPartitionedCall:output:0softmax__decode_191881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_softmax__decode_layer_call_and_return_conditional_losses_191480
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
э
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 2@
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
:џџџџџџџџџ
&
_user_specified_nameconv2d_input
A
А
F__inference_sequential_layer_call_and_return_conditional_losses_191485

inputs'
conv2d_191174: 
conv2d_191176: 
spiking_b_relu_191219: )
conv2d_1_191234: 0
conv2d_1_191236:0!
spiking_b_relu_1_191279: 
dense_191302:	А	x
dense_191304:x!
spiking_b_relu_2_191347:  
dense_1_191361:xT
dense_1_191363:T!
spiking_b_relu_3_191406:  
dense_2_191420:Td
dense_2_191422:d!
spiking_b_relu_4_191465: (
softmax__decode_191481:d

identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ'softmax__decode/StatefulPartitionedCallЂ&spiking_b_relu/StatefulPartitionedCallЂ(spiking_b_relu_1/StatefulPartitionedCallЂ(spiking_b_relu_2/StatefulPartitionedCallЂ(spiking_b_relu_3/StatefulPartitionedCallЂ(spiking_b_relu_4/StatefulPartitionedCall№
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_191174conv2d_191176*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_191173
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0spiking_b_relu_191219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218ѓ
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_191141
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_191234conv2d_1_191236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_191233 
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0spiking_b_relu_1_191279*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278љ
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153й
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџА	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289ў
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_191302dense_191304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_191301
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_2_191347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_191346
dense_1/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0dense_1_191361dense_1_191363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_191360
(spiking_b_relu_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_3_191406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_191405
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_3/StatefulPartitionedCall:output:0dense_2_191420dense_2_191422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_191419
(spiking_b_relu_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_4_191465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_191464
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_4/StatefulPartitionedCall:output:0softmax__decode_191481*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_softmax__decode_layer_call_and_return_conditional_losses_191480
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
э
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 2@
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
:џџџџџџџџџ
 
_user_specified_nameinputs
A
А
F__inference_sequential_layer_call_and_return_conditional_losses_191713

inputs'
conv2d_191666: 
conv2d_191668: 
spiking_b_relu_191671: )
conv2d_1_191675: 0
conv2d_1_191677:0!
spiking_b_relu_1_191680: 
dense_191685:	А	x
dense_191687:x!
spiking_b_relu_2_191690:  
dense_1_191693:xT
dense_1_191695:T!
spiking_b_relu_3_191698:  
dense_2_191701:Td
dense_2_191703:d!
spiking_b_relu_4_191706: (
softmax__decode_191709:d

identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ'softmax__decode/StatefulPartitionedCallЂ&spiking_b_relu/StatefulPartitionedCallЂ(spiking_b_relu_1/StatefulPartitionedCallЂ(spiking_b_relu_2/StatefulPartitionedCallЂ(spiking_b_relu_3/StatefulPartitionedCallЂ(spiking_b_relu_4/StatefulPartitionedCall№
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_191666conv2d_191668*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_191173
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0spiking_b_relu_191671*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218ѓ
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_191141
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_191675conv2d_1_191677*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_191233 
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0spiking_b_relu_1_191680*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278љ
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153й
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџА	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289ў
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_191685dense_191687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_191301
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_2_191690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_191346
dense_1/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0dense_1_191693dense_1_191695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_191360
(spiking_b_relu_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_3_191698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_191405
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_3/StatefulPartitionedCall:output:0dense_2_191701dense_2_191703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_191419
(spiking_b_relu_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_4_191706*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_191464
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_4/StatefulPartitionedCall:output:0softmax__decode_191709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_softmax__decode_layer_call_and_return_conditional_losses_191480
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
э
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 2@
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
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј

0__inference_softmax__decode_layer_call_fn_192837

inputs
unknown:d

identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_softmax__decode_layer_call_and_return_conditional_losses_191480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Г

+__inference_sequential_layer_call_fn_191520
conv2d_input!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	А	x
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
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_191485o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input
М
п
,sequential_spiking_b_relu_3_cond_true_191070N
Jsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_cast0
,sequential_spiking_b_relu_3_cond_placeholder-
)sequential_spiking_b_relu_3_cond_identityГ
)sequential/spiking_b_relu_3/cond/IdentityIdentityJsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_cast*
T0*'
_output_shapes
:џџџџџџџџџT"_
)sequential_spiking_b_relu_3_cond_identity2sequential/spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
Е
]
cond_false_191393
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџT"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
і
Ј
!spiking_b_relu_1_cond_true_1920578
4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast%
!spiking_b_relu_1_cond_placeholder"
spiking_b_relu_1_cond_identity
spiking_b_relu_1/cond/IdentityIdentity4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast*
T0*/
_output_shapes
:џџџџџџџџџ

0"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0
Е
]
cond_false_192684
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџx"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx

В
"spiking_b_relu_1_cond_false_192279%
!spiking_b_relu_1_cond_placeholderA
=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value"
spiking_b_relu_1_cond_identityЃ
spiking_b_relu_1/cond/IdentityIdentity=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ

0"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_192541

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ч
Т
F__inference_sequential_layer_call_and_return_conditional_losses_192204

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 0
&spiking_b_relu_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: 06
(conv2d_1_biasadd_readvariableop_resource:02
(spiking_b_relu_1_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	А	x3
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
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂ%softmax__decode/MatMul/ReadVariableOpЂ#spiking_b_relu/Equal/ReadVariableOpЂspiking_b_relu/ReadVariableOpЂ%spiking_b_relu_1/Equal/ReadVariableOpЂspiking_b_relu_1/ReadVariableOpЂ%spiking_b_relu_2/Equal/ReadVariableOpЂspiking_b_relu_2/ReadVariableOpЂ%spiking_b_relu_3/Equal/ReadVariableOpЂspiking_b_relu_3/ReadVariableOpЂ%spiking_b_relu_4/Equal/ReadVariableOpЂspiking_b_relu_4/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ b
spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?І
spiking_b_relu/GreaterEqualGreaterEqualconv2d/BiasAdd:output:0&spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
spiking_b_relu/CastCastspiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ |
spiking_b_relu/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0Y
spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu/subSubspiking_b_relu/sub/x:output:0%spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: Y
spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:s
spiking_b_relu/addAddV2spiking_b_relu/sub:z:0spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: ]
spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
spiking_b_relu/truedivRealDiv!spiking_b_relu/truediv/x:output:0spiking_b_relu/add:z:0*
T0*
_output_shapes
: [
spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu/sub_1Subconv2d/BiasAdd:output:0spiking_b_relu/sub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
spiking_b_relu/mulMulspiking_b_relu/truediv:z:0spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ [
spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu/add_1AddV2spiking_b_relu/mul:z:0spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ k
&spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
$spiking_b_relu/clip_by_value/MinimumMinimumspiking_b_relu/add_1:z:0/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ c
spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
spiking_b_relu/clip_by_valueMaximum(spiking_b_relu/clip_by_value/Minimum:z:0'spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
#spiking_b_relu/Equal/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu/EqualEqual+spiking_b_relu/Equal/ReadVariableOp:value:0spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: Џ
spiking_b_relu/condStatelessIfspiking_b_relu/Equal:z:0spiking_b_relu/Cast:y:0 spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *3
else_branch$R"
 spiking_b_relu_cond_false_192016*.
output_shapes
:џџџџџџџџџ *2
then_branch#R!
spiking_b_relu_cond_true_192015
spiking_b_relu/cond/IdentityIdentityspiking_b_relu/cond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Д
max_pooling2d/MaxPoolMaxPool%spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0Ф
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0d
spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ќ
spiking_b_relu_1/GreaterEqualGreaterEqualconv2d_1/BiasAdd:output:0(spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0
spiking_b_relu_1/CastCast!spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

0
spiking_b_relu_1/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_1/subSubspiking_b_relu_1/sub/x:output:0'spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:y
spiking_b_relu_1/addAddV2spiking_b_relu_1/sub:z:0spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_1/truedivRealDiv#spiking_b_relu_1/truediv/x:output:0spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_1/sub_1Subconv2d_1/BiasAdd:output:0!spiking_b_relu_1/sub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0
spiking_b_relu_1/mulMulspiking_b_relu_1/truediv:z:0spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ

0]
spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_1/add_1AddV2spiking_b_relu_1/mul:z:0!spiking_b_relu_1/add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0m
(spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?К
&spiking_b_relu_1/clip_by_value/MinimumMinimumspiking_b_relu_1/add_1:z:01spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0e
 spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    К
spiking_b_relu_1/clip_by_valueMaximum*spiking_b_relu_1/clip_by_value/Minimum:z:0)spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0
%spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_1/EqualEqual-spiking_b_relu_1/Equal/ReadVariableOp:value:0!spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: Л
spiking_b_relu_1/condStatelessIfspiking_b_relu_1/Equal:z:0spiking_b_relu_1/Cast:y:0"spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ

0* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_1_cond_false_192058*.
output_shapes
:џџџџџџџџџ

0*4
then_branch%R#
!spiking_b_relu_1_cond_true_192057
spiking_b_relu_1/cond/IdentityIdentityspiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0И
max_pooling2d_1/MaxPoolMaxPool'spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџА  
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџА	
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А	x*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxd
spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ё
spiking_b_relu_2/GreaterEqualGreaterEqualdense/BiasAdd:output:0(spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
spiking_b_relu_2/CastCast!spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџx
spiking_b_relu_2/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_2/subSubspiking_b_relu_2/sub/x:output:0'spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:y
spiking_b_relu_2/addAddV2spiking_b_relu_2/sub:z:0spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_2/truedivRealDiv#spiking_b_relu_2/truediv/x:output:0spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_2/sub_1Subdense/BiasAdd:output:0!spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
spiking_b_relu_2/mulMulspiking_b_relu_2/truediv:z:0spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџx]
spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_2/add_1AddV2spiking_b_relu_2/mul:z:0!spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxm
(spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?В
&spiking_b_relu_2/clip_by_value/MinimumMinimumspiking_b_relu_2/add_1:z:01spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxe
 spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    В
spiking_b_relu_2/clip_by_valueMaximum*spiking_b_relu_2/clip_by_value/Minimum:z:0)spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
%spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_2/EqualEqual-spiking_b_relu_2/Equal/ReadVariableOp:value:0!spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: Ћ
spiking_b_relu_2/condStatelessIfspiking_b_relu_2/Equal:z:0spiking_b_relu_2/Cast:y:0"spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџx* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_2_cond_false_192102*&
output_shapes
:џџџџџџџџџx*4
then_branch%R#
!spiking_b_relu_2_cond_true_192101|
spiking_b_relu_2/cond/IdentityIdentityspiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0
dense_1/MatMulMatMul'spiking_b_relu_2/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџT
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџTd
spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѓ
spiking_b_relu_3/GreaterEqualGreaterEqualdense_1/BiasAdd:output:0(spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
spiking_b_relu_3/CastCast!spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџT
spiking_b_relu_3/ReadVariableOpReadVariableOp(spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_3/subSubspiking_b_relu_3/sub/x:output:0'spiking_b_relu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:y
spiking_b_relu_3/addAddV2spiking_b_relu_3/sub:z:0spiking_b_relu_3/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_3/truedivRealDiv#spiking_b_relu_3/truediv/x:output:0spiking_b_relu_3/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_3/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_3/sub_1Subdense_1/BiasAdd:output:0!spiking_b_relu_3/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
spiking_b_relu_3/mulMulspiking_b_relu_3/truediv:z:0spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџT]
spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_3/add_1AddV2spiking_b_relu_3/mul:z:0!spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTm
(spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?В
&spiking_b_relu_3/clip_by_value/MinimumMinimumspiking_b_relu_3/add_1:z:01spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTe
 spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    В
spiking_b_relu_3/clip_by_valueMaximum*spiking_b_relu_3/clip_by_value/Minimum:z:0)spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
%spiking_b_relu_3/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_3/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_3/EqualEqual-spiking_b_relu_3/Equal/ReadVariableOp:value:0!spiking_b_relu_3/Equal/y:output:0*
T0*
_output_shapes
: Ћ
spiking_b_relu_3/condStatelessIfspiking_b_relu_3/Equal:z:0spiking_b_relu_3/Cast:y:0"spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџT* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_3_cond_false_192143*&
output_shapes
:џџџџџџџџџT*4
then_branch%R#
!spiking_b_relu_3_cond_true_192142|
spiking_b_relu_3/cond/IdentityIdentityspiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype0
dense_2/MatMulMatMul'spiking_b_relu_3/cond/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdd
spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѓ
spiking_b_relu_4/GreaterEqualGreaterEqualdense_2/BiasAdd:output:0(spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
spiking_b_relu_4/CastCast!spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd
spiking_b_relu_4/ReadVariableOpReadVariableOp(spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_4/subSubspiking_b_relu_4/sub/x:output:0'spiking_b_relu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:y
spiking_b_relu_4/addAddV2spiking_b_relu_4/sub:z:0spiking_b_relu_4/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_4/truedivRealDiv#spiking_b_relu_4/truediv/x:output:0spiking_b_relu_4/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_4/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_4/sub_1Subdense_2/BiasAdd:output:0!spiking_b_relu_4/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
spiking_b_relu_4/mulMulspiking_b_relu_4/truediv:z:0spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd]
spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_4/add_1AddV2spiking_b_relu_4/mul:z:0!spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdm
(spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?В
&spiking_b_relu_4/clip_by_value/MinimumMinimumspiking_b_relu_4/add_1:z:01spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
 spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    В
spiking_b_relu_4/clip_by_valueMaximum*spiking_b_relu_4/clip_by_value/Minimum:z:0)spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
%spiking_b_relu_4/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_4/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_4/EqualEqual-spiking_b_relu_4/Equal/ReadVariableOp:value:0!spiking_b_relu_4/Equal/y:output:0*
T0*
_output_shapes
: Ћ
spiking_b_relu_4/condStatelessIfspiking_b_relu_4/Equal:z:0spiking_b_relu_4/Cast:y:0"spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_4_cond_false_192184*&
output_shapes
:џџџџџџџџџd*4
then_branch%R#
!spiking_b_relu_4_cond_true_192183|
spiking_b_relu_4/cond/IdentityIdentityspiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџdZ
softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
softmax__decode/mulMulsoftmax__decode/mul/x:output:0'spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџdZ
softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
softmax__decode/subSubsoftmax__decode/mul:z:0softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
%softmax__decode/MatMul/ReadVariableOpReadVariableOp.softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0
softmax__decode/MatMulMatMulsoftmax__decode/sub:z:0-softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
v
softmax__decode/SoftmaxSoftmax softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
p
IdentityIdentity!softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^softmax__decode/MatMul/ReadVariableOp$^spiking_b_relu/Equal/ReadVariableOp^spiking_b_relu/ReadVariableOp&^spiking_b_relu_1/Equal/ReadVariableOp ^spiking_b_relu_1/ReadVariableOp&^spiking_b_relu_2/Equal/ReadVariableOp ^spiking_b_relu_2/ReadVariableOp&^spiking_b_relu_3/Equal/ReadVariableOp ^spiking_b_relu_3/ReadVariableOp&^spiking_b_relu_4/Equal/ReadVariableOp ^spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 2>
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
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю
Ј
!spiking_b_relu_3_cond_true_1921428
4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast%
!spiking_b_relu_3_cond_placeholder"
spiking_b_relu_3_cond_identity
spiking_b_relu_3/cond/IdentityIdentity4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast*
T0*'
_output_shapes
:џџџџџџџџџT"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
БA
Ж
F__inference_sequential_layer_call_and_return_conditional_losses_191835
conv2d_input'
conv2d_191788: 
conv2d_191790: 
spiking_b_relu_191793: )
conv2d_1_191797: 0
conv2d_1_191799:0!
spiking_b_relu_1_191802: 
dense_191807:	А	x
dense_191809:x!
spiking_b_relu_2_191812:  
dense_1_191815:xT
dense_1_191817:T!
spiking_b_relu_3_191820:  
dense_2_191823:Td
dense_2_191825:d!
spiking_b_relu_4_191828: (
softmax__decode_191831:d

identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ'softmax__decode/StatefulPartitionedCallЂ&spiking_b_relu/StatefulPartitionedCallЂ(spiking_b_relu_1/StatefulPartitionedCallЂ(spiking_b_relu_2/StatefulPartitionedCallЂ(spiking_b_relu_3/StatefulPartitionedCallЂ(spiking_b_relu_4/StatefulPartitionedCallі
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_191788conv2d_191790*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_191173
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0spiking_b_relu_191793*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218ѓ
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_191141
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_191797conv2d_1_191799*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_191233 
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0spiking_b_relu_1_191802*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278љ
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153й
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџА	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289ў
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_191807dense_191809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_191301
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_2_191812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_191346
dense_1/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0dense_1_191815dense_1_191817*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_191360
(spiking_b_relu_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_3_191820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_191405
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_3/StatefulPartitionedCall:output:0dense_2_191823dense_2_191825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_191419
(spiking_b_relu_4/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_4_191828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_191464
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_4/StatefulPartitionedCall:output:0softmax__decode_191831*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_softmax__decode_layer_call_and_return_conditional_losses_191480
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
э
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 2@
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
:џџџџџџџџџ
&
_user_specified_nameconv2d_input
Д
J
.__inference_max_pooling2d_layer_call_fn_192536

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_191141
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ц

'__inference_conv2d_layer_call_fn_192473

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_191173w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н
{
/__inference_spiking_b_relu_layer_call_fn_192492

inputs
unknown: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

Ж
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_192696

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџx^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџxL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxd
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: Х
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџx* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192684*&
output_shapes
:џџџџџџџџџx*#
then_branchR
cond_true_192683Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:џџџџџџџџџxe
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџxn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџx: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
с
В
"spiking_b_relu_2_cond_false_192102%
!spiking_b_relu_2_cond_placeholderA
=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value"
spiking_b_relu_2_cond_identity
spiking_b_relu_2/cond/IdentityIdentity=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџx"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx
с
В
"spiking_b_relu_4_cond_false_192405%
!spiking_b_relu_4_cond_placeholderA
=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value"
spiking_b_relu_4_cond_identity
spiking_b_relu_4/cond/IdentityIdentity=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџd"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd

В
"spiking_b_relu_1_cond_false_192058%
!spiking_b_relu_1_cond_placeholderA
=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value"
spiking_b_relu_1_cond_identityЃ
spiking_b_relu_1/cond/IdentityIdentity=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ

0"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0
И
L
0__inference_max_pooling2d_1_layer_call_fn_192613

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц	
є
C__inference_dense_2_layer_call_and_return_conditional_losses_192782

inputs0
matmul_readvariableop_resource:Td-
biasadd_readvariableop_resource:d
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Td*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџT
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_191141

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е
]
cond_false_191452
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџd"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd
Ё
}
1__inference_spiking_b_relu_3_layer_call_fn_192724

inputs
unknown: 
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_191405o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџT`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџT: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџT
 
_user_specified_nameinputs
ї
щ
-sequential_spiking_b_relu_1_cond_false_1909860
,sequential_spiking_b_relu_1_cond_placeholderW
Ssequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_clip_by_value-
)sequential_spiking_b_relu_1_cond_identityФ
)sequential/spiking_b_relu_1/cond/IdentityIdentitySsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ

0"_
)sequential_spiking_b_relu_1_cond_identity2sequential/spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0
Ъ
S
cond_true_191265
cond_identity_cast
cond_placeholder
cond_identityg
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:џџџџџџџџџ

0"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0
Ъp

__inference__traced_save_193034
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

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: !
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Ў 
valueЄ BЁ 7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-7/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-9/sharpness/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHм
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ж
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop#savev2_variable_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop%savev2_variable_1_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop%savev2_variable_2_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop%savev2_variable_3_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_5_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop<savev2_adadelta_conv2d_kernel_accum_grad_read_readvariableop:savev2_adadelta_conv2d_bias_accum_grad_read_readvariableop7savev2_adadelta_variable_accum_grad_read_readvariableop>savev2_adadelta_conv2d_1_kernel_accum_grad_read_readvariableop<savev2_adadelta_conv2d_1_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_1_read_readvariableop;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop9savev2_adadelta_dense_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_2_read_readvariableop=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_3_read_readvariableop=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_4_read_readvariableop;savev2_adadelta_conv2d_kernel_accum_var_read_readvariableop9savev2_adadelta_conv2d_bias_accum_var_read_readvariableop6savev2_adadelta_variable_accum_var_read_readvariableop=savev2_adadelta_conv2d_1_kernel_accum_var_read_readvariableop;savev2_adadelta_conv2d_1_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_1_read_readvariableop:savev2_adadelta_dense_kernel_accum_var_read_readvariableop8savev2_adadelta_dense_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_2_read_readvariableop<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_3_read_readvariableop<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_2_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_4_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*є
_input_shapesт
п: : : : : 0:0: :	А	x:x: :xT:T: :Td:d: :d
: : : : : : : : : : : : 0:0: :	А	x:x: :xT:T: :Td:d: : : : : 0:0: :	А	x:x: :xT:T: :Td:d: : 2(
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
:	А	x: 
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
:	А	x:  
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
:	А	x: /
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
Ђ
S
cond_true_191392
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:џџџџџџџџџT"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
Ъ
S
cond_true_192595
cond_identity_cast
cond_placeholder
cond_identityg
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:џџџџџџџџџ

0"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0
Х
_
C__inference_flatten_layer_call_and_return_conditional_losses_191289

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџА  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџА	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџА	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ0:W S
/
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Ѕ

ћ
B__inference_conv2d_layer_call_and_return_conditional_losses_191173

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
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
:џџџџџџџџџ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т

spiking_b_relu_cond_true_1922364
0spiking_b_relu_cond_identity_spiking_b_relu_cast#
spiking_b_relu_cond_placeholder 
spiking_b_relu_cond_identity
spiking_b_relu/cond/IdentityIdentity0spiking_b_relu_cond_identity_spiking_b_relu_cast*
T0*/
_output_shapes
:џџџџџџџџџ "E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 
Е
]
cond_false_192751
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџT"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
Ю
Ј
!spiking_b_relu_2_cond_true_1923228
4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast%
!spiking_b_relu_2_cond_placeholder"
spiking_b_relu_2_cond_identity
spiking_b_relu_2/cond/IdentityIdentity4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast*
T0*'
_output_shapes
:џџџџџџџџџx"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx

Ж
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_192830

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdd
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: Х
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192818*&
output_shapes
:џџџџџџџџџd*#
then_branchR
cond_true_192817Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ё

+__inference_sequential_layer_call_fn_191946

inputs!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	А	x
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
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_191485o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
п
,sequential_spiking_b_relu_2_cond_true_191029N
Jsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_cast0
,sequential_spiking_b_relu_2_cond_placeholder-
)sequential_spiking_b_relu_2_cond_identityГ
)sequential/spiking_b_relu_2/cond/IdentityIdentityJsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_cast*
T0*'
_output_shapes
:џџџџџџџџџx"_
)sequential_spiking_b_relu_2_cond_identity2sequential/spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx
Я
щ
-sequential_spiking_b_relu_4_cond_false_1911120
,sequential_spiking_b_relu_4_cond_placeholderW
Ssequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_clip_by_value-
)sequential_spiking_b_relu_4_cond_identityМ
)sequential/spiking_b_relu_4/cond/IdentityIdentitySsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџd"_
)sequential_spiking_b_relu_4_cond_identity2sequential/spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd

Ж
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_191405

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџT^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџTL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTd
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: Х
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџT* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191393*&
output_shapes
:џџџџџџџџџT*#
then_branchR
cond_true_191392Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:џџџџџџџџџTe
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџTn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџT: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџT
 
_user_specified_nameinputs
н
]
cond_false_192519
cond_placeholder
cond_identity_clip_by_value
cond_identityp
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 
Ш	
ѓ
A__inference_dense_layer_call_and_return_conditional_losses_191301

inputs1
matmul_readvariableop_resource:	А	x-
biasadd_readvariableop_resource:x
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А	x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџxw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџА	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџА	
 
_user_specified_nameinputs
ъ

)__inference_conv2d_1_layer_call_fn_192550

inputs!
unknown: 0
	unknown_0:0
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_191233w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ

0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ђ
S
cond_true_192750
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:џџџџџџџџџT"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT

Ж
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_192608

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0g
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

0^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?`
sub_1Subinputssub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0\
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ

0L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0d
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: е
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ

0* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192596*.
output_shapes
:џџџџџџџџџ

0*#
then_branchR
cond_true_192595b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ

0n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ

0: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ

0
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_192618

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е
]
cond_false_192818
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџd"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd
Ђ
S
cond_true_191333
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:џџџџџџџџџx"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx
Ц	
є
C__inference_dense_1_layer_call_and_return_conditional_losses_192715

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџTr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџT_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџTw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
ўм
Ж"
"__inference__traced_restore_193206
file_prefix8
assignvariableop_conv2d_kernel: ,
assignvariableop_1_conv2d_bias: %
assignvariableop_2_variable: <
"assignvariableop_3_conv2d_1_kernel: 0.
 assignvariableop_4_conv2d_1_bias:0'
assignvariableop_5_variable_1: 2
assignvariableop_6_dense_kernel:	А	x+
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
4assignvariableop_30_adadelta_dense_kernel_accum_grad:	А	x@
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
3assignvariableop_45_adadelta_dense_kernel_accum_var:	А	x?
1assignvariableop_46_adadelta_dense_bias_accum_var:x;
1assignvariableop_47_adadelta_variable_accum_var_2: G
5assignvariableop_48_adadelta_dense_1_kernel_accum_var:xTA
3assignvariableop_49_adadelta_dense_1_bias_accum_var:T;
1assignvariableop_50_adadelta_variable_accum_var_3: G
5assignvariableop_51_adadelta_dense_2_kernel_accum_var:TdA
3assignvariableop_52_adadelta_dense_2_bias_accum_var:d;
1assignvariableop_53_adadelta_variable_accum_var_4: 
identity_55ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Ў 
valueЄ BЁ 7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-7/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-9/sharpness/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHп
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Д
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_variableIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_3Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_4Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_5Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_16AssignVariableOp!assignvariableop_16_adadelta_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp"assignvariableop_17_adadelta_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adadelta_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp assignvariableop_19_adadelta_rhoIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adadelta_conv2d_kernel_accum_gradIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adadelta_conv2d_bias_accum_gradIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adadelta_variable_accum_gradIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adadelta_conv2d_1_kernel_accum_gradIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adadelta_conv2d_1_bias_accum_gradIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adadelta_variable_accum_grad_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adadelta_dense_kernel_accum_gradIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adadelta_dense_bias_accum_gradIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adadelta_variable_accum_grad_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adadelta_dense_1_kernel_accum_gradIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adadelta_dense_1_bias_accum_gradIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adadelta_variable_accum_grad_3Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adadelta_dense_2_kernel_accum_gradIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adadelta_dense_2_bias_accum_gradIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adadelta_variable_accum_grad_4Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adadelta_conv2d_kernel_accum_varIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adadelta_conv2d_bias_accum_varIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_41AssignVariableOp/assignvariableop_41_adadelta_variable_accum_varIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adadelta_conv2d_1_kernel_accum_varIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adadelta_conv2d_1_bias_accum_varIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_44AssignVariableOp1assignvariableop_44_adadelta_variable_accum_var_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adadelta_dense_kernel_accum_varIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_46AssignVariableOp1assignvariableop_46_adadelta_dense_bias_accum_varIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_47AssignVariableOp1assignvariableop_47_adadelta_variable_accum_var_2Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adadelta_dense_1_kernel_accum_varIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adadelta_dense_1_bias_accum_varIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_50AssignVariableOp1assignvariableop_50_adadelta_variable_accum_var_3Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adadelta_dense_2_kernel_accum_varIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adadelta_dense_2_bias_accum_varIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_53AssignVariableOp1assignvariableop_53_adadelta_variable_accum_var_4Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ѓ	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: р	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_55Identity_55:output:0*
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
с
В
"spiking_b_relu_4_cond_false_192184%
!spiking_b_relu_4_cond_placeholderA
=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value"
spiking_b_relu_4_cond_identity
spiking_b_relu_4/cond/IdentityIdentity=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџd"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd
Ѕ

ћ
B__inference_conv2d_layer_call_and_return_conditional_losses_192483

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
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
:џџџџџџџџџ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
S
cond_true_192683
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:џџџџџџџџџx"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx
с
В
"spiking_b_relu_3_cond_false_192143%
!spiking_b_relu_3_cond_placeholderA
=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value"
spiking_b_relu_3_cond_identity
spiking_b_relu_3/cond/IdentityIdentity=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџT"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
­
D
(__inference_flatten_layer_call_fn_192623

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџА	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџА	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ0:W S
/
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Ъ
S
cond_true_192518
cond_identity_cast
cond_placeholder
cond_identityg
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:џџџџџџџџџ "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 

Д
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_192531

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ g
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?`
sub_1Subinputssub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ d
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: е
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192519*.
output_shapes
:џџџџџџџџџ *#
then_branchR
cond_true_192518b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ : 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

Ж
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_192763

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџT^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџTL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTd
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: Х
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџT* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192751*&
output_shapes
:џџџџџџџџџT*#
then_branchR
cond_true_192750Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:џџџџџџџџџTe
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџTn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџT: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџT
 
_user_specified_nameinputs
Ц	
є
C__inference_dense_1_layer_call_and_return_conditional_losses_191360

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџTr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџT_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџTw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
Ш	
ѓ
A__inference_dense_layer_call_and_return_conditional_losses_192648

inputs1
matmul_readvariableop_resource:	А	x-
biasadd_readvariableop_resource:x
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А	x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџxw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџА	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџА	
 
_user_specified_nameinputs
ь	
Д
K__inference_softmax__decode_layer_call_and_return_conditional_losses_192849

inputs0
matmul_readvariableop_resource:d

identityЂMatMul/ReadVariableOpJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџdJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0j
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ъ
S
cond_true_191205
cond_identity_cast
cond_placeholder
cond_identityg
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:џџџџџџџџџ "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 
ль
К
!__inference__wrapped_model_191132
conv2d_inputJ
0sequential_conv2d_conv2d_readvariableop_resource: ?
1sequential_conv2d_biasadd_readvariableop_resource: ;
1sequential_spiking_b_relu_readvariableop_resource: L
2sequential_conv2d_1_conv2d_readvariableop_resource: 0A
3sequential_conv2d_1_biasadd_readvariableop_resource:0=
3sequential_spiking_b_relu_1_readvariableop_resource: B
/sequential_dense_matmul_readvariableop_resource:	А	x>
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
identityЂ(sequential/conv2d/BiasAdd/ReadVariableOpЂ'sequential/conv2d/Conv2D/ReadVariableOpЂ*sequential/conv2d_1/BiasAdd/ReadVariableOpЂ)sequential/conv2d_1/Conv2D/ReadVariableOpЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOpЂ)sequential/dense_2/BiasAdd/ReadVariableOpЂ(sequential/dense_2/MatMul/ReadVariableOpЂ0sequential/softmax__decode/MatMul/ReadVariableOpЂ.sequential/spiking_b_relu/Equal/ReadVariableOpЂ(sequential/spiking_b_relu/ReadVariableOpЂ0sequential/spiking_b_relu_1/Equal/ReadVariableOpЂ*sequential/spiking_b_relu_1/ReadVariableOpЂ0sequential/spiking_b_relu_2/Equal/ReadVariableOpЂ*sequential/spiking_b_relu_2/ReadVariableOpЂ0sequential/spiking_b_relu_3/Equal/ReadVariableOpЂ*sequential/spiking_b_relu_3/ReadVariableOpЂ0sequential/spiking_b_relu_4/Equal/ReadVariableOpЂ*sequential/spiking_b_relu_4/ReadVariableOp 
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0У
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ m
(sequential/spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ч
&sequential/spiking_b_relu/GreaterEqualGreaterEqual"sequential/conv2d/BiasAdd:output:01sequential/spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential/spiking_b_relu/CastCast*sequential/spiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ 
(sequential/spiking_b_relu/ReadVariableOpReadVariableOp1sequential_spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0d
sequential/spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ё
sequential/spiking_b_relu/subSub(sequential/spiking_b_relu/sub/x:output:00sequential/spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: d
sequential/spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
sequential/spiking_b_relu/addAddV2!sequential/spiking_b_relu/sub:z:0(sequential/spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
!sequential/spiking_b_relu/truedivRealDiv,sequential/spiking_b_relu/truediv/x:output:0!sequential/spiking_b_relu/add:z:0*
T0*
_output_shapes
: f
!sequential/spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?А
sequential/spiking_b_relu/sub_1Sub"sequential/conv2d/BiasAdd:output:0*sequential/spiking_b_relu/sub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Њ
sequential/spiking_b_relu/mulMul%sequential/spiking_b_relu/truediv:z:0#sequential/spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ f
!sequential/spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Б
sequential/spiking_b_relu/add_1AddV2!sequential/spiking_b_relu/mul:z:0*sequential/spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
1sequential/spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?е
/sequential/spiking_b_relu/clip_by_value/MinimumMinimum#sequential/spiking_b_relu/add_1:z:0:sequential/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ n
)sequential/spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    е
'sequential/spiking_b_relu/clip_by_valueMaximum3sequential/spiking_b_relu/clip_by_value/Minimum:z:02sequential/spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
.sequential/spiking_b_relu/Equal/ReadVariableOpReadVariableOp1sequential_spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
sequential/spiking_b_relu/EqualEqual6sequential/spiking_b_relu/Equal/ReadVariableOp:value:0*sequential/spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: ё
sequential/spiking_b_relu/condStatelessIf#sequential/spiking_b_relu/Equal:z:0"sequential/spiking_b_relu/Cast:y:0+sequential/spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *>
else_branch/R-
+sequential_spiking_b_relu_cond_false_190944*.
output_shapes
:џџџџџџџџџ *=
then_branch.R,
*sequential_spiking_b_relu_cond_true_190943
'sequential/spiking_b_relu/cond/IdentityIdentity'sequential/spiking_b_relu/cond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Ъ
 sequential/max_pooling2d/MaxPoolMaxPool0sequential/spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
Є
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0х
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0*
paddingVALID*
strides

*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Й
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0o
*sequential/spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Э
(sequential/spiking_b_relu_1/GreaterEqualGreaterEqual$sequential/conv2d_1/BiasAdd:output:03sequential/spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0
 sequential/spiking_b_relu_1/CastCast,sequential/spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

0
*sequential/spiking_b_relu_1/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
sequential/spiking_b_relu_1/subSub*sequential/spiking_b_relu_1/sub/x:output:02sequential/spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!sequential/spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
sequential/spiking_b_relu_1/addAddV2#sequential/spiking_b_relu_1/sub:z:0*sequential/spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: j
%sequential/spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Є
#sequential/spiking_b_relu_1/truedivRealDiv.sequential/spiking_b_relu_1/truediv/x:output:0#sequential/spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ж
!sequential/spiking_b_relu_1/sub_1Sub$sequential/conv2d_1/BiasAdd:output:0,sequential/spiking_b_relu_1/sub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0А
sequential/spiking_b_relu_1/mulMul'sequential/spiking_b_relu_1/truediv:z:0%sequential/spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ

0h
#sequential/spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?З
!sequential/spiking_b_relu_1/add_1AddV2#sequential/spiking_b_relu_1/mul:z:0,sequential/spiking_b_relu_1/add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0x
3sequential/spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?л
1sequential/spiking_b_relu_1/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_1/add_1:z:0<sequential/spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0p
+sequential/spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    л
)sequential/spiking_b_relu_1/clip_by_valueMaximum5sequential/spiking_b_relu_1/clip_by_value/Minimum:z:04sequential/spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0
0sequential/spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Г
!sequential/spiking_b_relu_1/EqualEqual8sequential/spiking_b_relu_1/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: §
 sequential/spiking_b_relu_1/condStatelessIf%sequential/spiking_b_relu_1/Equal:z:0$sequential/spiking_b_relu_1/Cast:y:0-sequential/spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ

0* 
_read_only_resource_inputs
 *@
else_branch1R/
-sequential_spiking_b_relu_1_cond_false_190986*.
output_shapes
:џџџџџџџџџ

0*?
then_branch0R.
,sequential_spiking_b_relu_1_cond_true_190985
)sequential/spiking_b_relu_1/cond/IdentityIdentity)sequential/spiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0Ю
"sequential/max_pooling2d_1/MaxPoolMaxPool2sequential/spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџА  Ј
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_1/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџА	
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	А	x*
dtype0Ј
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0Љ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxo
*sequential/spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Т
(sequential/spiking_b_relu_2/GreaterEqualGreaterEqual!sequential/dense/BiasAdd:output:03sequential/spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
 sequential/spiking_b_relu_2/CastCast,sequential/spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџx
*sequential/spiking_b_relu_2/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
sequential/spiking_b_relu_2/subSub*sequential/spiking_b_relu_2/sub/x:output:02sequential/spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!sequential/spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
sequential/spiking_b_relu_2/addAddV2#sequential/spiking_b_relu_2/sub:z:0*sequential/spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: j
%sequential/spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Є
#sequential/spiking_b_relu_2/truedivRealDiv.sequential/spiking_b_relu_2/truediv/x:output:0#sequential/spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ћ
!sequential/spiking_b_relu_2/sub_1Sub!sequential/dense/BiasAdd:output:0,sequential/spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxЈ
sequential/spiking_b_relu_2/mulMul'sequential/spiking_b_relu_2/truediv:z:0%sequential/spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџxh
#sequential/spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Џ
!sequential/spiking_b_relu_2/add_1AddV2#sequential/spiking_b_relu_2/mul:z:0,sequential/spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxx
3sequential/spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?г
1sequential/spiking_b_relu_2/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_2/add_1:z:0<sequential/spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxp
+sequential/spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    г
)sequential/spiking_b_relu_2/clip_by_valueMaximum5sequential/spiking_b_relu_2/clip_by_value/Minimum:z:04sequential/spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
0sequential/spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Г
!sequential/spiking_b_relu_2/EqualEqual8sequential/spiking_b_relu_2/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: э
 sequential/spiking_b_relu_2/condStatelessIf%sequential/spiking_b_relu_2/Equal:z:0$sequential/spiking_b_relu_2/Cast:y:0-sequential/spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџx* 
_read_only_resource_inputs
 *@
else_branch1R/
-sequential_spiking_b_relu_2_cond_false_191030*&
output_shapes
:џџџџџџџџџx*?
then_branch0R.
,sequential_spiking_b_relu_2_cond_true_191029
)sequential/spiking_b_relu_2/cond/IdentityIdentity)sequential/spiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0Л
sequential/dense_1/MatMulMatMul2sequential/spiking_b_relu_2/cond/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџT
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0Џ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџTo
*sequential/spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
(sequential/spiking_b_relu_3/GreaterEqualGreaterEqual#sequential/dense_1/BiasAdd:output:03sequential/spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
 sequential/spiking_b_relu_3/CastCast,sequential/spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџT
*sequential/spiking_b_relu_3/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
sequential/spiking_b_relu_3/subSub*sequential/spiking_b_relu_3/sub/x:output:02sequential/spiking_b_relu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!sequential/spiking_b_relu_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
sequential/spiking_b_relu_3/addAddV2#sequential/spiking_b_relu_3/sub:z:0*sequential/spiking_b_relu_3/add/y:output:0*
T0*
_output_shapes
: j
%sequential/spiking_b_relu_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Є
#sequential/spiking_b_relu_3/truedivRealDiv.sequential/spiking_b_relu_3/truediv/x:output:0#sequential/spiking_b_relu_3/add:z:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu_3/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
!sequential/spiking_b_relu_3/sub_1Sub#sequential/dense_1/BiasAdd:output:0,sequential/spiking_b_relu_3/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTЈ
sequential/spiking_b_relu_3/mulMul'sequential/spiking_b_relu_3/truediv:z:0%sequential/spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџTh
#sequential/spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Џ
!sequential/spiking_b_relu_3/add_1AddV2#sequential/spiking_b_relu_3/mul:z:0,sequential/spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTx
3sequential/spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?г
1sequential/spiking_b_relu_3/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_3/add_1:z:0<sequential/spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTp
+sequential/spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    г
)sequential/spiking_b_relu_3/clip_by_valueMaximum5sequential/spiking_b_relu_3/clip_by_value/Minimum:z:04sequential/spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
0sequential/spiking_b_relu_3/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_3/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Г
!sequential/spiking_b_relu_3/EqualEqual8sequential/spiking_b_relu_3/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_3/Equal/y:output:0*
T0*
_output_shapes
: э
 sequential/spiking_b_relu_3/condStatelessIf%sequential/spiking_b_relu_3/Equal:z:0$sequential/spiking_b_relu_3/Cast:y:0-sequential/spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџT* 
_read_only_resource_inputs
 *@
else_branch1R/
-sequential_spiking_b_relu_3_cond_false_191071*&
output_shapes
:џџџџџџџџџT*?
then_branch0R.
,sequential_spiking_b_relu_3_cond_true_191070
)sequential/spiking_b_relu_3/cond/IdentityIdentity)sequential/spiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype0Л
sequential/dense_2/MatMulMatMul2sequential/spiking_b_relu_3/cond/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Џ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdo
*sequential/spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
(sequential/spiking_b_relu_4/GreaterEqualGreaterEqual#sequential/dense_2/BiasAdd:output:03sequential/spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
 sequential/spiking_b_relu_4/CastCast,sequential/spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd
*sequential/spiking_b_relu_4/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
sequential/spiking_b_relu_4/subSub*sequential/spiking_b_relu_4/sub/x:output:02sequential/spiking_b_relu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!sequential/spiking_b_relu_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
sequential/spiking_b_relu_4/addAddV2#sequential/spiking_b_relu_4/sub:z:0*sequential/spiking_b_relu_4/add/y:output:0*
T0*
_output_shapes
: j
%sequential/spiking_b_relu_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Є
#sequential/spiking_b_relu_4/truedivRealDiv.sequential/spiking_b_relu_4/truediv/x:output:0#sequential/spiking_b_relu_4/add:z:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu_4/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
!sequential/spiking_b_relu_4/sub_1Sub#sequential/dense_2/BiasAdd:output:0,sequential/spiking_b_relu_4/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
sequential/spiking_b_relu_4/mulMul'sequential/spiking_b_relu_4/truediv:z:0%sequential/spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdh
#sequential/spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Џ
!sequential/spiking_b_relu_4/add_1AddV2#sequential/spiking_b_relu_4/mul:z:0,sequential/spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdx
3sequential/spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?г
1sequential/spiking_b_relu_4/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_4/add_1:z:0<sequential/spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdp
+sequential/spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    г
)sequential/spiking_b_relu_4/clip_by_valueMaximum5sequential/spiking_b_relu_4/clip_by_value/Minimum:z:04sequential/spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
0sequential/spiking_b_relu_4/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_4/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Г
!sequential/spiking_b_relu_4/EqualEqual8sequential/spiking_b_relu_4/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_4/Equal/y:output:0*
T0*
_output_shapes
: э
 sequential/spiking_b_relu_4/condStatelessIf%sequential/spiking_b_relu_4/Equal:z:0$sequential/spiking_b_relu_4/Cast:y:0-sequential/spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *@
else_branch1R/
-sequential_spiking_b_relu_4_cond_false_191112*&
output_shapes
:џџџџџџџџџd*?
then_branch0R.
,sequential_spiking_b_relu_4_cond_true_191111
)sequential/spiking_b_relu_4/cond/IdentityIdentity)sequential/spiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
 sequential/softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ж
sequential/softmax__decode/mulMul)sequential/softmax__decode/mul/x:output:02sequential/spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
 sequential/softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?І
sequential/softmax__decode/subSub"sequential/softmax__decode/mul:z:0)sequential/softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdЊ
0sequential/softmax__decode/MatMul/ReadVariableOpReadVariableOp9sequential_softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0Л
!sequential/softmax__decode/MatMulMatMul"sequential/softmax__decode/sub:z:08sequential/softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

"sequential/softmax__decode/SoftmaxSoftmax+sequential/softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
{
IdentityIdentity,sequential/softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp1^sequential/softmax__decode/MatMul/ReadVariableOp/^sequential/spiking_b_relu/Equal/ReadVariableOp)^sequential/spiking_b_relu/ReadVariableOp1^sequential/spiking_b_relu_1/Equal/ReadVariableOp+^sequential/spiking_b_relu_1/ReadVariableOp1^sequential/spiking_b_relu_2/Equal/ReadVariableOp+^sequential/spiking_b_relu_2/ReadVariableOp1^sequential/spiking_b_relu_3/Equal/ReadVariableOp+^sequential/spiking_b_relu_3/ReadVariableOp1^sequential/spiking_b_relu_4/Equal/ReadVariableOp+^sequential/spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 2T
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
:џџџџџџџџџ
&
_user_specified_nameconv2d_input

Ж
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_191464

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџdL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdd
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: Х
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191452*&
output_shapes
:џџџџџџџџџd*#
then_branchR
cond_true_191451Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ю
Ј
!spiking_b_relu_3_cond_true_1923638
4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast%
!spiking_b_relu_3_cond_placeholder"
spiking_b_relu_3_cond_identity
spiking_b_relu_3/cond/IdentityIdentity4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast*
T0*'
_output_shapes
:џџџџџџџџџT"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
Ё
}
1__inference_spiking_b_relu_4_layer_call_fn_192791

inputs
unknown: 
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_191464o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
і
Ј
!spiking_b_relu_1_cond_true_1922788
4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast%
!spiking_b_relu_1_cond_placeholder"
spiking_b_relu_1_cond_identity
spiking_b_relu_1/cond/IdentityIdentity4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast*
T0*/
_output_shapes
:џџџџџџџџџ

0"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0

Д
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ g
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?`
sub_1Subinputssub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ \
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ d
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: е
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191206*.
output_shapes
:џџџџџџџџџ *#
then_branchR
cond_true_191205b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ : 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Е
]
cond_false_191334
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџx"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx
Ю
Ј
!spiking_b_relu_2_cond_true_1921018
4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast%
!spiking_b_relu_2_cond_placeholder"
spiking_b_relu_2_cond_identity
spiking_b_relu_2/cond/IdentityIdentity4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast*
T0*'
_output_shapes
:џџџџџџџџџx"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx
Ј

§
D__inference_conv2d_1_layer_call_and_return_conditional_losses_192560

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ

0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
н
]
cond_false_192596
cond_placeholder
cond_identity_clip_by_value
cond_identityp
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ

0"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0
н
]
cond_false_191206
cond_placeholder
cond_identity_clip_by_value
cond_identityp
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 
Я
щ
-sequential_spiking_b_relu_3_cond_false_1910710
,sequential_spiking_b_relu_3_cond_placeholderW
Ssequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_clip_by_value-
)sequential_spiking_b_relu_3_cond_identityМ
)sequential/spiking_b_relu_3/cond/IdentityIdentitySsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџT"_
)sequential_spiking_b_relu_3_cond_identity2sequential/spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
ѕ
Ј
 spiking_b_relu_cond_false_192016#
spiking_b_relu_cond_placeholder=
9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value 
spiking_b_relu_cond_identity
spiking_b_relu/cond/IdentityIdentity9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ "E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 
у
п
+sequential_spiking_b_relu_cond_false_190944.
*sequential_spiking_b_relu_cond_placeholderS
Osequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_clip_by_value+
'sequential_spiking_b_relu_cond_identityО
'sequential/spiking_b_relu/cond/IdentityIdentityOsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ "[
'sequential_spiking_b_relu_cond_identity0sequential/spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 

Ж
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0g
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

0^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?`
sub_1Subinputssub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0\
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ

0L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0d
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: е
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ

0* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191266*.
output_shapes
:џџџџџџџџџ

0*#
then_branchR
cond_true_191265b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ

0n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ

0: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ

0
 
_user_specified_nameinputs
П

&__inference_dense_layer_call_fn_192638

inputs
unknown:	А	x
	unknown_0:x
identityЂStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_191301o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџА	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџА	
 
_user_specified_nameinputs
н
]
cond_false_191266
cond_placeholder
cond_identity_clip_by_value
cond_identityp
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ

0"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0
ф
п
,sequential_spiking_b_relu_1_cond_true_190985N
Jsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_cast0
,sequential_spiking_b_relu_1_cond_placeholder-
)sequential_spiking_b_relu_1_cond_identityЛ
)sequential/spiking_b_relu_1/cond/IdentityIdentityJsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_cast*
T0*/
_output_shapes
:џџџџџџџџџ

0"_
)sequential_spiking_b_relu_1_cond_identity2sequential/spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ

0:џџџџџџџџџ

0:5 1
/
_output_shapes
:џџџџџџџџџ

0:51
/
_output_shapes
:џџџџџџџџџ

0
а
е
*sequential_spiking_b_relu_cond_true_190943J
Fsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_cast.
*sequential_spiking_b_relu_cond_placeholder+
'sequential_spiking_b_relu_cond_identityЕ
'sequential/spiking_b_relu/cond/IdentityIdentityFsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_cast*
T0*/
_output_shapes
:џџџџџџџџџ "[
'sequential_spiking_b_relu_cond_identity0sequential/spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 
т

spiking_b_relu_cond_true_1920154
0spiking_b_relu_cond_identity_spiking_b_relu_cast#
spiking_b_relu_cond_placeholder 
spiking_b_relu_cond_identity
spiking_b_relu/cond/IdentityIdentity0spiking_b_relu_cond_identity_spiking_b_relu_cast*
T0*/
_output_shapes
:џџџџџџџџџ "E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 
М
п
,sequential_spiking_b_relu_4_cond_true_191111N
Jsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_cast0
,sequential_spiking_b_relu_4_cond_placeholder-
)sequential_spiking_b_relu_4_cond_identityГ
)sequential/spiking_b_relu_4/cond/IdentityIdentityJsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_cast*
T0*'
_output_shapes
:џџџџџџџџџd"_
)sequential_spiking_b_relu_4_cond_identity2sequential/spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџd:џџџџџџџџџd:- )
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd
Я
щ
-sequential_spiking_b_relu_2_cond_false_1910300
,sequential_spiking_b_relu_2_cond_placeholderW
Ssequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_clip_by_value-
)sequential_spiking_b_relu_2_cond_identityМ
)sequential/spiking_b_relu_2/cond/IdentityIdentitySsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџx"_
)sequential_spiking_b_relu_2_cond_identity2sequential/spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџx:џџџџџџџџџx:- )
'
_output_shapes
:џџџџџџџџџx:-)
'
_output_shapes
:џџџџџџџџџx
Р

(__inference_dense_1_layer_call_fn_192705

inputs
unknown:xT
	unknown_0:T
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_191360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџT`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџx: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
Х
_
C__inference_flatten_layer_call_and_return_conditional_losses_192629

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџА  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџА	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџА	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ0:W S
/
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs

Ж
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_191346

inputs!
readvariableop_resource: 
identityЂEqual/ReadVariableOpЂReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџx^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
sub_1Subinputssub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџxL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxd
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: Х
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџx* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191334*&
output_shapes
:џџџџџџџџџx*#
then_branchR
cond_true_191333Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:џџџџџџџџџxe
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџxn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџx: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs
ь	
Д
K__inference_softmax__decode_layer_call_and_return_conditional_losses_191480

inputs0
matmul_readvariableop_resource:d

identityЂMatMul/ReadVariableOpJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџdJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0j
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџd: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
С
}
1__inference_spiking_b_relu_1_layer_call_fn_192569

inputs
unknown: 
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ

0*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ

0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ

0: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ

0
 
_user_specified_nameinputs
Р

(__inference_dense_2_layer_call_fn_192772

inputs
unknown:Td
	unknown_0:d
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_191419o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџT: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџT
 
_user_specified_nameinputs
Ч
Т
F__inference_sequential_layer_call_and_return_conditional_losses_192425

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 0
&spiking_b_relu_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: 06
(conv2d_1_biasadd_readvariableop_resource:02
(spiking_b_relu_1_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	А	x3
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
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂ%softmax__decode/MatMul/ReadVariableOpЂ#spiking_b_relu/Equal/ReadVariableOpЂspiking_b_relu/ReadVariableOpЂ%spiking_b_relu_1/Equal/ReadVariableOpЂspiking_b_relu_1/ReadVariableOpЂ%spiking_b_relu_2/Equal/ReadVariableOpЂspiking_b_relu_2/ReadVariableOpЂ%spiking_b_relu_3/Equal/ReadVariableOpЂspiking_b_relu_3/ReadVariableOpЂ%spiking_b_relu_4/Equal/ReadVariableOpЂspiking_b_relu_4/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ b
spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?І
spiking_b_relu/GreaterEqualGreaterEqualconv2d/BiasAdd:output:0&spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
spiking_b_relu/CastCastspiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ |
spiking_b_relu/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0Y
spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu/subSubspiking_b_relu/sub/x:output:0%spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: Y
spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:s
spiking_b_relu/addAddV2spiking_b_relu/sub:z:0spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: ]
spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
spiking_b_relu/truedivRealDiv!spiking_b_relu/truediv/x:output:0spiking_b_relu/add:z:0*
T0*
_output_shapes
: [
spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu/sub_1Subconv2d/BiasAdd:output:0spiking_b_relu/sub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
spiking_b_relu/mulMulspiking_b_relu/truediv:z:0spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ [
spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu/add_1AddV2spiking_b_relu/mul:z:0spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ k
&spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Д
$spiking_b_relu/clip_by_value/MinimumMinimumspiking_b_relu/add_1:z:0/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ c
spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
spiking_b_relu/clip_by_valueMaximum(spiking_b_relu/clip_by_value/Minimum:z:0'spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
#spiking_b_relu/Equal/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu/EqualEqual+spiking_b_relu/Equal/ReadVariableOp:value:0spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: Џ
spiking_b_relu/condStatelessIfspiking_b_relu/Equal:z:0spiking_b_relu/Cast:y:0 spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *3
else_branch$R"
 spiking_b_relu_cond_false_192237*.
output_shapes
:џџџџџџџџџ *2
then_branch#R!
spiking_b_relu_cond_true_192236
spiking_b_relu/cond/IdentityIdentityspiking_b_relu/cond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Д
max_pooling2d/MaxPoolMaxPool%spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0Ф
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ

0d
spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ќ
spiking_b_relu_1/GreaterEqualGreaterEqualconv2d_1/BiasAdd:output:0(spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0
spiking_b_relu_1/CastCast!spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ

0
spiking_b_relu_1/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_1/subSubspiking_b_relu_1/sub/x:output:0'spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:y
spiking_b_relu_1/addAddV2spiking_b_relu_1/sub:z:0spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_1/truedivRealDiv#spiking_b_relu_1/truediv/x:output:0spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_1/sub_1Subconv2d_1/BiasAdd:output:0!spiking_b_relu_1/sub_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0
spiking_b_relu_1/mulMulspiking_b_relu_1/truediv:z:0spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ

0]
spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_1/add_1AddV2spiking_b_relu_1/mul:z:0!spiking_b_relu_1/add_1/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0m
(spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?К
&spiking_b_relu_1/clip_by_value/MinimumMinimumspiking_b_relu_1/add_1:z:01spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0e
 spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    К
spiking_b_relu_1/clip_by_valueMaximum*spiking_b_relu_1/clip_by_value/Minimum:z:0)spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0
%spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_1/EqualEqual-spiking_b_relu_1/Equal/ReadVariableOp:value:0!spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: Л
spiking_b_relu_1/condStatelessIfspiking_b_relu_1/Equal:z:0spiking_b_relu_1/Cast:y:0"spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ

0* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_1_cond_false_192279*.
output_shapes
:џџџџџџџџџ

0*4
then_branch%R#
!spiking_b_relu_1_cond_true_192278
spiking_b_relu_1/cond/IdentityIdentityspiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ

0И
max_pooling2d_1/MaxPoolMaxPool'spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:џџџџџџџџџ0*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџА  
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџА	
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А	x*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџxd
spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ё
spiking_b_relu_2/GreaterEqualGreaterEqualdense/BiasAdd:output:0(spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
spiking_b_relu_2/CastCast!spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџx
spiking_b_relu_2/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_2/subSubspiking_b_relu_2/sub/x:output:0'spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:y
spiking_b_relu_2/addAddV2spiking_b_relu_2/sub:z:0spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_2/truedivRealDiv#spiking_b_relu_2/truediv/x:output:0spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_2/sub_1Subdense/BiasAdd:output:0!spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
spiking_b_relu_2/mulMulspiking_b_relu_2/truediv:z:0spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџx]
spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_2/add_1AddV2spiking_b_relu_2/mul:z:0!spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxm
(spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?В
&spiking_b_relu_2/clip_by_value/MinimumMinimumspiking_b_relu_2/add_1:z:01spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџxe
 spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    В
spiking_b_relu_2/clip_by_valueMaximum*spiking_b_relu_2/clip_by_value/Minimum:z:0)spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
%spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_2/EqualEqual-spiking_b_relu_2/Equal/ReadVariableOp:value:0!spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: Ћ
spiking_b_relu_2/condStatelessIfspiking_b_relu_2/Equal:z:0spiking_b_relu_2/Cast:y:0"spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџx* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_2_cond_false_192323*&
output_shapes
:џџџџџџџџџx*4
then_branch%R#
!spiking_b_relu_2_cond_true_192322|
spiking_b_relu_2/cond/IdentityIdentityspiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0
dense_1/MatMulMatMul'spiking_b_relu_2/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџT
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџTd
spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѓ
spiking_b_relu_3/GreaterEqualGreaterEqualdense_1/BiasAdd:output:0(spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
spiking_b_relu_3/CastCast!spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџT
spiking_b_relu_3/ReadVariableOpReadVariableOp(spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_3/subSubspiking_b_relu_3/sub/x:output:0'spiking_b_relu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:y
spiking_b_relu_3/addAddV2spiking_b_relu_3/sub:z:0spiking_b_relu_3/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_3/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_3/truedivRealDiv#spiking_b_relu_3/truediv/x:output:0spiking_b_relu_3/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_3/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_3/sub_1Subdense_1/BiasAdd:output:0!spiking_b_relu_3/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
spiking_b_relu_3/mulMulspiking_b_relu_3/truediv:z:0spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџT]
spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_3/add_1AddV2spiking_b_relu_3/mul:z:0!spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTm
(spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?В
&spiking_b_relu_3/clip_by_value/MinimumMinimumspiking_b_relu_3/add_1:z:01spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџTe
 spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    В
spiking_b_relu_3/clip_by_valueMaximum*spiking_b_relu_3/clip_by_value/Minimum:z:0)spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
%spiking_b_relu_3/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_3/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_3/EqualEqual-spiking_b_relu_3/Equal/ReadVariableOp:value:0!spiking_b_relu_3/Equal/y:output:0*
T0*
_output_shapes
: Ћ
spiking_b_relu_3/condStatelessIfspiking_b_relu_3/Equal:z:0spiking_b_relu_3/Cast:y:0"spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџT* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_3_cond_false_192364*&
output_shapes
:џџџџџџџџџT*4
then_branch%R#
!spiking_b_relu_3_cond_true_192363|
spiking_b_relu_3/cond/IdentityIdentityspiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype0
dense_2/MatMulMatMul'spiking_b_relu_3/cond/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdd
spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѓ
spiking_b_relu_4/GreaterEqualGreaterEqualdense_2/BiasAdd:output:0(spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
spiking_b_relu_4/CastCast!spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd
spiking_b_relu_4/ReadVariableOpReadVariableOp(spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_4/subSubspiking_b_relu_4/sub/x:output:0'spiking_b_relu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:y
spiking_b_relu_4/addAddV2spiking_b_relu_4/sub:z:0spiking_b_relu_4/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_4/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_4/truedivRealDiv#spiking_b_relu_4/truediv/x:output:0spiking_b_relu_4/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_4/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_4/sub_1Subdense_2/BiasAdd:output:0!spiking_b_relu_4/sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
spiking_b_relu_4/mulMulspiking_b_relu_4/truediv:z:0spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd]
spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_4/add_1AddV2spiking_b_relu_4/mul:z:0!spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџdm
(spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?В
&spiking_b_relu_4/clip_by_value/MinimumMinimumspiking_b_relu_4/add_1:z:01spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџde
 spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    В
spiking_b_relu_4/clip_by_valueMaximum*spiking_b_relu_4/clip_by_value/Minimum:z:0)spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
%spiking_b_relu_4/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_4/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
spiking_b_relu_4/EqualEqual-spiking_b_relu_4/Equal/ReadVariableOp:value:0!spiking_b_relu_4/Equal/y:output:0*
T0*
_output_shapes
: Ћ
spiking_b_relu_4/condStatelessIfspiking_b_relu_4/Equal:z:0spiking_b_relu_4/Cast:y:0"spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_4_cond_false_192405*&
output_shapes
:џџџџџџџџџd*4
then_branch%R#
!spiking_b_relu_4_cond_true_192404|
spiking_b_relu_4/cond/IdentityIdentityspiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:џџџџџџџџџdZ
softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
softmax__decode/mulMulsoftmax__decode/mul/x:output:0'spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџdZ
softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
softmax__decode/subSubsoftmax__decode/mul:z:0softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
%softmax__decode/MatMul/ReadVariableOpReadVariableOp.softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0
softmax__decode/MatMulMatMulsoftmax__decode/sub:z:0-softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
v
softmax__decode/SoftmaxSoftmax softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
p
IdentityIdentity!softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^softmax__decode/MatMul/ReadVariableOp$^spiking_b_relu/Equal/ReadVariableOp^spiking_b_relu/ReadVariableOp&^spiking_b_relu_1/Equal/ReadVariableOp ^spiking_b_relu_1/ReadVariableOp&^spiking_b_relu_2/Equal/ReadVariableOp ^spiking_b_relu_2/ReadVariableOp&^spiking_b_relu_3/Equal/ReadVariableOp ^spiking_b_relu_3/ReadVariableOp&^spiking_b_relu_4/Equal/ReadVariableOp ^spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 2>
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
:џџџџџџџџџ
 
_user_specified_nameinputs
с
В
"spiking_b_relu_3_cond_false_192364%
!spiking_b_relu_3_cond_placeholderA
=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value"
spiking_b_relu_3_cond_identity
spiking_b_relu_3/cond/IdentityIdentity=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџT"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџT:џџџџџџџџџT:- )
'
_output_shapes
:џџџџџџџџџT:-)
'
_output_shapes
:џџџџџџџџџT
ѕ
Ј
 spiking_b_relu_cond_false_192237#
spiking_b_relu_cond_placeholder=
9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value 
spiking_b_relu_cond_identity
spiking_b_relu/cond/IdentityIdentity9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:џџџџџџџџџ "E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ :џџџџџџџџџ :5 1
/
_output_shapes
:џџџџџџџџџ :51
/
_output_shapes
:џџџџџџџџџ 
Ё
}
1__inference_spiking_b_relu_2_layer_call_fn_192657

inputs
unknown: 
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџx*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_191346o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџx: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџx
 
_user_specified_nameinputs


$__inference_signature_wrapper_192464
conv2d_input!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	А	x
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
identityЂStatefulPartitionedCallї
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
:џџџџџџџџџ
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_191132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:џџџџџџџџџ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameconv2d_input"лL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ф
serving_defaultА
M
conv2d_input=
serving_default_conv2d_input:0џџџџџџџџџC
softmax__decode0
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:ащ
П
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
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Л

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
 	sharpness
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
5	sharpness
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
P	sharpness
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

Wkernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
_	sharpness
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

fkernel
gbias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
n	sharpness
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
И
u_rescaled_key
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
я
|iter
	}decay
~learning_rate
rho
accum_gradз
accum_gradи 
accum_gradй-
accum_gradк.
accum_gradл5
accum_gradмH
accum_gradнI
accum_gradоP
accum_gradпW
accum_gradрX
accum_gradс_
accum_gradтf
accum_gradуg
accum_gradфn
accum_gradх	accum_varц	accum_varч 	accum_varш-	accum_varщ.	accum_varъ5	accum_varыH	accum_varьI	accum_varэP	accum_varюW	accum_varяX	accum_var№_	accum_varёf	accum_varђg	accum_varѓn	accum_varє"
	optimizer

0
1
 2
-3
.4
55
H6
I7
P8
W9
X10
_11
f12
g13
n14
u15"
trackable_list_wrapper

0
1
 2
-3
.4
55
H6
I7
P8
W9
X10
_11
f12
g13
n14"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
њ2ї
+__inference_sequential_layer_call_fn_191520
+__inference_sequential_layer_call_fn_191946
+__inference_sequential_layer_call_fn_191983
+__inference_sequential_layer_call_fn_191785Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
F__inference_sequential_layer_call_and_return_conditional_losses_192204
F__inference_sequential_layer_call_and_return_conditional_losses_192425
F__inference_sequential_layer_call_and_return_conditional_losses_191835
F__inference_sequential_layer_call_and_return_conditional_losses_191885Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
бBЮ
!__inference__wrapped_model_191132conv2d_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
-
serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
б2Ю
'__inference_conv2d_layer_call_fn_192473Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_conv2d_layer_call_and_return_conditional_losses_192483Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
: 2Variable
'
 0"
trackable_list_wrapper
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
й2ж
/__inference_spiking_b_relu_layer_call_fn_192492Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
є2ё
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_192531Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
и2е
.__inference_max_pooling2d_layer_call_fn_192536Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_192541Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
):' 02conv2d_1/kernel
:02conv2d_1/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
г2а
)__inference_conv2d_1_layer_call_fn_192550Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_1_layer_call_and_return_conditional_losses_192560Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
: 2Variable
'
50"
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
л2и
1__inference_spiking_b_relu_1_layer_call_fn_192569Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_192608Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
к2з
0__inference_max_pooling2d_1_layer_call_fn_192613Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_192618Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
в2Я
(__inference_flatten_layer_call_fn_192623Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_192629Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	А	x2dense/kernel
:x2
dense/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Љnon_trainable_variables
Њlayers
Ћmetrics
 Ќlayer_regularization_losses
­layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
а2Э
&__inference_dense_layer_call_fn_192638Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_192648Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
: 2Variable
'
P0"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
л2и
1__inference_spiking_b_relu_2_layer_call_fn_192657Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_192696Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 :xT2dense_1/kernel
:T2dense_1/bias
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
в2Я
(__inference_dense_1_layer_call_fn_192705Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_192715Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
: 2Variable
'
_0"
trackable_list_wrapper
'
_0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
л2и
1__inference_spiking_b_relu_3_layer_call_fn_192724Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_192763Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 :Td2dense_2/kernel
:d2dense_2/bias
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
В
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
в2Я
(__inference_dense_2_layer_call_fn_192772Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_192782Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
: 2Variable
'
n0"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
л2и
1__inference_spiking_b_relu_4_layer_call_fn_192791Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_192830Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:d
2Variable
'
u0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
к2з
0__inference_softmax__decode_layer_call_fn_192837Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_softmax__decode_layer_call_and_return_conditional_losses_192849Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
'
u0"
trackable_list_wrapper

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
Ь0
Э1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
$__inference_signature_wrapper_192464conv2d_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
'
u0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

Юtotal

Яcount
а	variables
б	keras_api"
_tf_keras_metric
c

вtotal

гcount
д
_fn_kwargs
е	variables
ж	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
Ю0
Я1"
trackable_list_wrapper
.
а	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
в0
г1"
trackable_list_wrapper
.
е	variables"
_generic_user_object
9:7 2!Adadelta/conv2d/kernel/accum_grad
+:) 2Adadelta/conv2d/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
;:9 02#Adadelta/conv2d_1/kernel/accum_grad
-:+02!Adadelta/conv2d_1/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
1:/	А	x2 Adadelta/dense/kernel/accum_grad
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
0:.	А	x2Adadelta/dense/kernel/accum_var
):'x2Adadelta/dense/bias/accum_var
#:! 2Adadelta/Variable/accum_var
1:/xT2!Adadelta/dense_1/kernel/accum_var
+:)T2Adadelta/dense_1/bias/accum_var
#:! 2Adadelta/Variable/accum_var
1:/Td2!Adadelta/dense_2/kernel/accum_var
+:)d2Adadelta/dense_2/bias/accum_var
#:! 2Adadelta/Variable/accum_varК
!__inference__wrapped_model_191132 -.5HIPWX_fgnu=Ђ:
3Ђ0
.+
conv2d_inputџџџџџџџџџ
Њ "AЊ>
<
softmax__decode)&
softmax__decodeџџџџџџџџџ
Д
D__inference_conv2d_1_layer_call_and_return_conditional_losses_192560l-.7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ

0
 
)__inference_conv2d_1_layer_call_fn_192550_-.7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ

0В
B__inference_conv2d_layer_call_and_return_conditional_losses_192483l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
'__inference_conv2d_layer_call_fn_192473_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ Ѓ
C__inference_dense_1_layer_call_and_return_conditional_losses_192715\WX/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "%Ђ"

0џџџџџџџџџT
 {
(__inference_dense_1_layer_call_fn_192705OWX/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "џџџџџџџџџTЃ
C__inference_dense_2_layer_call_and_return_conditional_losses_192782\fg/Ђ,
%Ђ"
 
inputsџџџџџџџџџT
Њ "%Ђ"

0џџџџџџџџџd
 {
(__inference_dense_2_layer_call_fn_192772Ofg/Ђ,
%Ђ"
 
inputsџџџџџџџџџT
Њ "џџџџџџџџџdЂ
A__inference_dense_layer_call_and_return_conditional_losses_192648]HI0Ђ-
&Ђ#
!
inputsџџџџџџџџџА	
Њ "%Ђ"

0џџџџџџџџџx
 z
&__inference_dense_layer_call_fn_192638PHI0Ђ-
&Ђ#
!
inputsџџџџџџџџџА	
Њ "џџџџџџџџџxЈ
C__inference_flatten_layer_call_and_return_conditional_losses_192629a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ0
Њ "&Ђ#

0џџџџџџџџџА	
 
(__inference_flatten_layer_call_fn_192623T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ0
Њ "џџџџџџџџџА	ю
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_192618RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ц
0__inference_max_pooling2d_1_layer_call_fn_192613RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџь
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_192541RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
.__inference_max_pooling2d_layer_call_fn_192536RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЫ
F__inference_sequential_layer_call_and_return_conditional_losses_191835 -.5HIPWX_fgnuEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Ы
F__inference_sequential_layer_call_and_return_conditional_losses_191885 -.5HIPWX_fgnuEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Ф
F__inference_sequential_layer_call_and_return_conditional_losses_192204z -.5HIPWX_fgnu?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Ф
F__inference_sequential_layer_call_and_return_conditional_losses_192425z -.5HIPWX_fgnu?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Ђ
+__inference_sequential_layer_call_fn_191520s -.5HIPWX_fgnuEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
Ђ
+__inference_sequential_layer_call_fn_191785s -.5HIPWX_fgnuEЂB
;Ђ8
.+
conv2d_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

+__inference_sequential_layer_call_fn_191946m -.5HIPWX_fgnu?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

+__inference_sequential_layer_call_fn_191983m -.5HIPWX_fgnu?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
Э
$__inference_signature_wrapper_192464Є -.5HIPWX_fgnuMЂJ
Ђ 
CЊ@
>
conv2d_input.+
conv2d_inputџџџџџџџџџ"AЊ>
<
softmax__decode)&
softmax__decodeџџџџџџџџџ
Њ
K__inference_softmax__decode_layer_call_and_return_conditional_losses_192849[u/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ

 
0__inference_softmax__decode_layer_call_fn_192837Nu/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџ
Л
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_192608k57Ђ4
-Ђ*
(%
inputsџџџџџџџџџ

0
Њ "-Ђ*
# 
0џџџџџџџџџ

0
 
1__inference_spiking_b_relu_1_layer_call_fn_192569^57Ђ4
-Ђ*
(%
inputsџџџџџџџџџ

0
Њ " џџџџџџџџџ

0Ћ
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_192696[P/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "%Ђ"

0џџџџџџџџџx
 
1__inference_spiking_b_relu_2_layer_call_fn_192657NP/Ђ,
%Ђ"
 
inputsџџџџџџџџџx
Њ "џџџџџџџџџxЋ
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_192763[_/Ђ,
%Ђ"
 
inputsџџџџџџџџџT
Њ "%Ђ"

0џџџџџџџџџT
 
1__inference_spiking_b_relu_3_layer_call_fn_192724N_/Ђ,
%Ђ"
 
inputsџџџџџџџџџT
Њ "џџџџџџџџџTЋ
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_192830[n/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџd
 
1__inference_spiking_b_relu_4_layer_call_fn_192791Nn/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџdЙ
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_192531k 7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
/__inference_spiking_b_relu_layer_call_fn_192492^ 7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ 