ô
÷È
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
Á
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
executor_typestring ¨
À
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Íî
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
shape:	°	x*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	°	x*
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
¦
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
ª
#Adadelta/conv2d_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*4
shared_name%#Adadelta/conv2d_1/kernel/accum_grad
£
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
shape:	°	x*1
shared_name" Adadelta/dense/kernel/accum_grad

4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense/kernel/accum_grad*
_output_shapes
:	°	x*
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
¤
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
¨
"Adadelta/conv2d_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*3
shared_name$"Adadelta/conv2d_1/kernel/accum_var
¡
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
shape:	°	x*0
shared_name!Adadelta/dense/kernel/accum_var

3Adadelta/dense/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/kernel/accum_var*
_output_shapes
:	°	x*
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
dtype0*Ú}
valueÐ}BÍ} BÆ}
¥
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
¦

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
¦

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
¦

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
¦

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
¦

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
£
u_rescaled_key
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses*
à
|iter
	}decay
~learning_rate
rho
accum_grad×
accum_gradØ 
accum_gradÙ-
accum_gradÚ.
accum_gradÛ5
accum_gradÜH
accum_gradÝI
accum_gradÞP
accum_gradßW
accum_gradàX
accum_gradá_
accum_gradâf
accum_gradãg
accum_gradän
accum_gradå	accum_varæ	accum_varç 	accum_varè-	accum_varé.	accum_varê5	accum_varëH	accum_varìI	accum_varíP	accum_varîW	accum_varïX	accum_varð_	accum_varñf	accum_varòg	accum_varón	accum_varô*
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
µ
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
¡metrics
 ¢layer_regularization_losses
£layer_metrics
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
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
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
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
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
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
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
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
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
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
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
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
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
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
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
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
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
Ì0
Í1*
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

Îtotal

Ïcount
Ð	variables
Ñ	keras_api*
M

Òtotal

Ócount
Ô
_fn_kwargs
Õ	variables
Ö	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Î0
Ï1*

Ð	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ò0
Ó1*

Õ	variables*
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
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
¯
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
:ÿÿÿÿÿÿÿÿÿ
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
§
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
ê
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
"__inference__traced_restore_193206´Ø
¢
S
cond_true_192817
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
á
²
"spiking_b_relu_2_cond_false_192323%
!spiking_b_relu_2_cond_placeholderA
=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value"
spiking_b_relu_2_cond_identity
spiking_b_relu_2/cond/IdentityIdentity=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
¢
S
cond_true_191451
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Î
¨
!spiking_b_relu_4_cond_true_1921838
4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast%
!spiking_b_relu_4_cond_placeholder"
spiking_b_relu_4_cond_identity
spiking_b_relu_4/cond/IdentityIdentity4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
¨

ý
D__inference_conv2d_1_layer_call_and_return_conditional_losses_191233

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Æ	
ô
C__inference_dense_2_layer_call_and_return_conditional_losses_191419

inputs0
matmul_readvariableop_resource:Td-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Td*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
¡

+__inference_sequential_layer_call_fn_191983

inputs!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	°	x
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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

+__inference_sequential_layer_call_fn_191785
conv2d_input!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	°	x
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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
¨
!spiking_b_relu_4_cond_true_1924048
4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast%
!spiking_b_relu_4_cond_placeholder"
spiking_b_relu_4_cond_identity
spiking_b_relu_4/cond/IdentityIdentity4spiking_b_relu_4_cond_identity_spiking_b_relu_4_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
±A
¶
F__inference_sequential_layer_call_and_return_conditional_losses_191885
conv2d_input'
conv2d_191838: 
conv2d_191840: 
spiking_b_relu_191843: )
conv2d_1_191847: 0
conv2d_1_191849:0!
spiking_b_relu_1_191852: 
dense_191857:	°	x
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
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢'softmax__decode/StatefulPartitionedCall¢&spiking_b_relu/StatefulPartitionedCall¢(spiking_b_relu_1/StatefulPartitionedCall¢(spiking_b_relu_2/StatefulPartitionedCall¢(spiking_b_relu_3/StatefulPartitionedCall¢(spiking_b_relu_4/StatefulPartitionedCallö
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_191838conv2d_191840*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
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
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218ó
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278ù
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153Ù
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_191857dense_191859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
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
:ÿÿÿÿÿÿÿÿÿx*#
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
:ÿÿÿÿÿÿÿÿÿT*$
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
:ÿÿÿÿÿÿÿÿÿT*#
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
:ÿÿÿÿÿÿÿÿÿd*$
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
:ÿÿÿÿÿÿÿÿÿd*#
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
í
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
A
°
F__inference_sequential_layer_call_and_return_conditional_losses_191485

inputs'
conv2d_191174: 
conv2d_191176: 
spiking_b_relu_191219: )
conv2d_1_191234: 0
conv2d_1_191236:0!
spiking_b_relu_1_191279: 
dense_191302:	°	x
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
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢'softmax__decode/StatefulPartitionedCall¢&spiking_b_relu/StatefulPartitionedCall¢(spiking_b_relu_1/StatefulPartitionedCall¢(spiking_b_relu_2/StatefulPartitionedCall¢(spiking_b_relu_3/StatefulPartitionedCall¢(spiking_b_relu_4/StatefulPartitionedCallð
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_191174conv2d_191176*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
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
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218ó
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278ù
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153Ù
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_191302dense_191304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
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
:ÿÿÿÿÿÿÿÿÿx*#
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
:ÿÿÿÿÿÿÿÿÿT*$
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
:ÿÿÿÿÿÿÿÿÿT*#
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
:ÿÿÿÿÿÿÿÿÿd*$
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
:ÿÿÿÿÿÿÿÿÿd*#
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
í
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
A
°
F__inference_sequential_layer_call_and_return_conditional_losses_191713

inputs'
conv2d_191666: 
conv2d_191668: 
spiking_b_relu_191671: )
conv2d_1_191675: 0
conv2d_1_191677:0!
spiking_b_relu_1_191680: 
dense_191685:	°	x
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
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢'softmax__decode/StatefulPartitionedCall¢&spiking_b_relu/StatefulPartitionedCall¢(spiking_b_relu_1/StatefulPartitionedCall¢(spiking_b_relu_2/StatefulPartitionedCall¢(spiking_b_relu_3/StatefulPartitionedCall¢(spiking_b_relu_4/StatefulPartitionedCallð
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_191666conv2d_191668*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
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
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218ó
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278ù
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153Ù
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_191685dense_191687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
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
:ÿÿÿÿÿÿÿÿÿx*#
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
:ÿÿÿÿÿÿÿÿÿT*$
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
:ÿÿÿÿÿÿÿÿÿT*#
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
:ÿÿÿÿÿÿÿÿÿd*$
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
:ÿÿÿÿÿÿÿÿÿd*#
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
í
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

0__inference_softmax__decode_layer_call_fn_192837

inputs
unknown:d

identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
³

+__inference_sequential_layer_call_fn_191520
conv2d_input!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	°	x
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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
¼
ß
,sequential_spiking_b_relu_3_cond_true_191070N
Jsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_cast0
,sequential_spiking_b_relu_3_cond_placeholder-
)sequential_spiking_b_relu_3_cond_identity³
)sequential/spiking_b_relu_3/cond/IdentityIdentityJsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"_
)sequential_spiking_b_relu_3_cond_identity2sequential/spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
µ
]
cond_false_191393
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
ö
¨
!spiking_b_relu_1_cond_true_1920578
4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast%
!spiking_b_relu_1_cond_placeholder"
spiking_b_relu_1_cond_identity
spiking_b_relu_1/cond/IdentityIdentity4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
µ
]
cond_false_192684
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

²
"spiking_b_relu_1_cond_false_192279%
!spiking_b_relu_1_cond_placeholderA
=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value"
spiking_b_relu_1_cond_identity£
spiking_b_relu_1/cond/IdentityIdentity=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_192541

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
Â
F__inference_sequential_layer_call_and_return_conditional_losses_192204

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 0
&spiking_b_relu_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: 06
(conv2d_1_biasadd_readvariableop_resource:02
(spiking_b_relu_1_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	°	x3
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
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢%softmax__decode/MatMul/ReadVariableOp¢#spiking_b_relu/Equal/ReadVariableOp¢spiking_b_relu/ReadVariableOp¢%spiking_b_relu_1/Equal/ReadVariableOp¢spiking_b_relu_1/ReadVariableOp¢%spiking_b_relu_2/Equal/ReadVariableOp¢spiking_b_relu_2/ReadVariableOp¢%spiking_b_relu_3/Equal/ReadVariableOp¢spiking_b_relu_3/ReadVariableOp¢%spiking_b_relu_4/Equal/ReadVariableOp¢spiking_b_relu_4/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0§
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ b
spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
spiking_b_relu/GreaterEqualGreaterEqualconv2d/BiasAdd:output:0&spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
spiking_b_relu/CastCastspiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
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
:ÿÿÿÿÿÿÿÿÿ 
spiking_b_relu/mulMulspiking_b_relu/truediv:z:0spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu/add_1AddV2spiking_b_relu/mul:z:0spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
&spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
$spiking_b_relu/clip_by_value/MinimumMinimumspiking_b_relu/add_1:z:0/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
spiking_b_relu/clip_by_valueMaximum(spiking_b_relu/clip_by_value/Minimum:z:0'spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
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
: ¯
spiking_b_relu/condStatelessIfspiking_b_relu/Equal:z:0spiking_b_relu/Cast:y:0 spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *3
else_branch$R"
 spiking_b_relu_cond_false_192016*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ *2
then_branch#R!
spiking_b_relu_cond_true_192015
spiking_b_relu/cond/IdentityIdentityspiking_b_relu/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ´
max_pooling2d/MaxPoolMaxPool%spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0Ä
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0d
spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¬
spiking_b_relu_1/GreaterEqualGreaterEqualconv2d_1/BiasAdd:output:0(spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
spiking_b_relu_1/CastCast!spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0
spiking_b_relu_1/mulMulspiking_b_relu_1/truediv:z:0spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0m
(spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?º
&spiking_b_relu_1/clip_by_value/MinimumMinimumspiking_b_relu_1/add_1:z:01spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0e
 spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    º
spiking_b_relu_1/clip_by_valueMaximum*spiking_b_relu_1/clip_by_value/Minimum:z:0)spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
: »
spiking_b_relu_1/condStatelessIfspiking_b_relu_1/Equal:z:0spiking_b_relu_1/Cast:y:0"spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_1_cond_false_192058*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ

0*4
then_branch%R#
!spiking_b_relu_1_cond_true_192057
spiking_b_relu_1/cond/IdentityIdentityspiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0¸
max_pooling2d_1/MaxPoolMaxPool'spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°  
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	°	x*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¡
spiking_b_relu_2/GreaterEqualGreaterEqualdense/BiasAdd:output:0(spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
spiking_b_relu_2/CastCast!spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
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
:ÿÿÿÿÿÿÿÿÿx
spiking_b_relu_2/mulMulspiking_b_relu_2/truediv:z:0spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx]
spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_2/add_1AddV2spiking_b_relu_2/mul:z:0!spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxm
(spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
&spiking_b_relu_2/clip_by_value/MinimumMinimumspiking_b_relu_2/add_1:z:01spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxe
 spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
spiking_b_relu_2/clip_by_valueMaximum*spiking_b_relu_2/clip_by_value/Minimum:z:0)spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
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
: «
spiking_b_relu_2/condStatelessIfspiking_b_relu_2/Equal:z:0spiking_b_relu_2/Cast:y:0"spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_2_cond_false_192102*&
output_shapes
:ÿÿÿÿÿÿÿÿÿx*4
then_branch%R#
!spiking_b_relu_2_cond_true_192101|
spiking_b_relu_2/cond/IdentityIdentityspiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0
dense_1/MatMulMatMul'spiking_b_relu_2/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTd
spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
spiking_b_relu_3/GreaterEqualGreaterEqualdense_1/BiasAdd:output:0(spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
spiking_b_relu_3/CastCast!spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
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
:ÿÿÿÿÿÿÿÿÿT
spiking_b_relu_3/mulMulspiking_b_relu_3/truediv:z:0spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT]
spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_3/add_1AddV2spiking_b_relu_3/mul:z:0!spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTm
(spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
&spiking_b_relu_3/clip_by_value/MinimumMinimumspiking_b_relu_3/add_1:z:01spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTe
 spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
spiking_b_relu_3/clip_by_valueMaximum*spiking_b_relu_3/clip_by_value/Minimum:z:0)spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
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
: «
spiking_b_relu_3/condStatelessIfspiking_b_relu_3/Equal:z:0spiking_b_relu_3/Cast:y:0"spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_3_cond_false_192143*&
output_shapes
:ÿÿÿÿÿÿÿÿÿT*4
then_branch%R#
!spiking_b_relu_3_cond_true_192142|
spiking_b_relu_3/cond/IdentityIdentityspiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype0
dense_2/MatMulMatMul'spiking_b_relu_3/cond/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
spiking_b_relu_4/GreaterEqualGreaterEqualdense_2/BiasAdd:output:0(spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
spiking_b_relu_4/CastCast!spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
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
:ÿÿÿÿÿÿÿÿÿd
spiking_b_relu_4/mulMulspiking_b_relu_4/truediv:z:0spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_4/add_1AddV2spiking_b_relu_4/mul:z:0!spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
&spiking_b_relu_4/clip_by_value/MinimumMinimumspiking_b_relu_4/add_1:z:01spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
 spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
spiking_b_relu_4/clip_by_valueMaximum*spiking_b_relu_4/clip_by_value/Minimum:z:0)spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
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
: «
spiking_b_relu_4/condStatelessIfspiking_b_relu_4/Equal:z:0spiking_b_relu_4/Cast:y:0"spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_4_cond_false_192184*&
output_shapes
:ÿÿÿÿÿÿÿÿÿd*4
then_branch%R#
!spiking_b_relu_4_cond_true_192183|
spiking_b_relu_4/cond/IdentityIdentityspiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
softmax__decode/mulMulsoftmax__decode/mul/x:output:0'spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
softmax__decode/subSubsoftmax__decode/mul:z:0softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%softmax__decode/MatMul/ReadVariableOpReadVariableOp.softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0
softmax__decode/MatMulMatMulsoftmax__decode/sub:z:0-softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
softmax__decode/SoftmaxSoftmax softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
IdentityIdentity!softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^softmax__decode/MatMul/ReadVariableOp$^spiking_b_relu/Equal/ReadVariableOp^spiking_b_relu/ReadVariableOp&^spiking_b_relu_1/Equal/ReadVariableOp ^spiking_b_relu_1/ReadVariableOp&^spiking_b_relu_2/Equal/ReadVariableOp ^spiking_b_relu_2/ReadVariableOp&^spiking_b_relu_3/Equal/ReadVariableOp ^spiking_b_relu_3/ReadVariableOp&^spiking_b_relu_4/Equal/ReadVariableOp ^spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
¨
!spiking_b_relu_3_cond_true_1921428
4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast%
!spiking_b_relu_3_cond_placeholder"
spiking_b_relu_3_cond_identity
spiking_b_relu_3/cond/IdentityIdentity4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
±A
¶
F__inference_sequential_layer_call_and_return_conditional_losses_191835
conv2d_input'
conv2d_191788: 
conv2d_191790: 
spiking_b_relu_191793: )
conv2d_1_191797: 0
conv2d_1_191799:0!
spiking_b_relu_1_191802: 
dense_191807:	°	x
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
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢'softmax__decode/StatefulPartitionedCall¢&spiking_b_relu/StatefulPartitionedCall¢(spiking_b_relu_1/StatefulPartitionedCall¢(spiking_b_relu_2/StatefulPartitionedCall¢(spiking_b_relu_3/StatefulPartitionedCall¢(spiking_b_relu_4/StatefulPartitionedCallö
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_191788conv2d_191790*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
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
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218ó
max_pooling2d/PartitionedCallPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278ù
max_pooling2d_1/PartitionedCallPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_191153Ù
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_191289þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_191807dense_191809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
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
:ÿÿÿÿÿÿÿÿÿx*#
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
:ÿÿÿÿÿÿÿÿÿT*$
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
:ÿÿÿÿÿÿÿÿÿT*#
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
:ÿÿÿÿÿÿÿÿÿd*$
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
:ÿÿÿÿÿÿÿÿÿd*#
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
í
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall)^spiking_b_relu_3/StatefulPartitionedCall)^spiking_b_relu_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
´
J
.__inference_max_pooling2d_layer_call_fn_192536

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
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
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

'__inference_conv2d_layer_call_fn_192473

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
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
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
{
/__inference_spiking_b_relu_layer_call_fn_192492

inputs
unknown: 
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
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
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¶
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_192696

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
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
:ÿÿÿÿÿÿÿÿÿxT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
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
: Å
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192684*&
output_shapes
:ÿÿÿÿÿÿÿÿÿx*#
then_branchR
cond_true_192683Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxe
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
á
²
"spiking_b_relu_2_cond_false_192102%
!spiking_b_relu_2_cond_placeholderA
=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value"
spiking_b_relu_2_cond_identity
spiking_b_relu_2/cond/IdentityIdentity=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
á
²
"spiking_b_relu_4_cond_false_192405%
!spiking_b_relu_4_cond_placeholderA
=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value"
spiking_b_relu_4_cond_identity
spiking_b_relu_4/cond/IdentityIdentity=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

²
"spiking_b_relu_1_cond_false_192058%
!spiking_b_relu_1_cond_placeholderA
=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value"
spiking_b_relu_1_cond_identity£
spiking_b_relu_1/cond/IdentityIdentity=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
¸
L
0__inference_max_pooling2d_1_layer_call_fn_192613

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
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
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ	
ô
C__inference_dense_2_layer_call_and_return_conditional_losses_192782

inputs0
matmul_readvariableop_resource:Td-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Td*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_191141

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
]
cond_false_191452
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
¡
}
1__inference_spiking_b_relu_3_layer_call_fn_192724

inputs
unknown: 
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*#
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
:ÿÿÿÿÿÿÿÿÿT`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
÷
é
-sequential_spiking_b_relu_1_cond_false_1909860
,sequential_spiking_b_relu_1_cond_placeholderW
Ssequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_clip_by_value-
)sequential_spiking_b_relu_1_cond_identityÄ
)sequential/spiking_b_relu_1/cond/IdentityIdentitySsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"_
)sequential_spiking_b_relu_1_cond_identity2sequential/spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
Ê
S
cond_true_191265
cond_identity_cast
cond_placeholder
cond_identityg
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
Êp
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

identity_1¢MergeV2Checkpointsw
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
dtype0*® 
value¤ B¡ 7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-7/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-9/sharpness/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÜ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¶
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

identity_1Identity_1:output:0*ô
_input_shapesâ
ß: : : : : 0:0: :	°	x:x: :xT:T: :Td:d: :d
: : : : : : : : : : : : 0:0: :	°	x:x: :xT:T: :Td:d: : : : : 0:0: :	°	x:x: :xT:T: :Td:d: : 2(
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
:	°	x: 
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
:	°	x:  
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
:	°	x: /
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
¢
S
cond_true_191392
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Ê
S
cond_true_192595
cond_identity_cast
cond_placeholder
cond_identityg
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
Å
_
C__inference_flatten_layer_call_and_return_conditional_losses_191289

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¥

û
B__inference_conv2d_layer_call_and_return_conditional_losses_191173

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â

spiking_b_relu_cond_true_1922364
0spiking_b_relu_cond_identity_spiking_b_relu_cast#
spiking_b_relu_cond_placeholder 
spiking_b_relu_cond_identity
spiking_b_relu/cond/IdentityIdentity0spiking_b_relu_cond_identity_spiking_b_relu_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
µ
]
cond_false_192751
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Î
¨
!spiking_b_relu_2_cond_true_1923228
4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast%
!spiking_b_relu_2_cond_placeholder"
spiking_b_relu_2_cond_identity
spiking_b_relu_2/cond/IdentityIdentity4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

¶
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_192830

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
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
:ÿÿÿÿÿÿÿÿÿdT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
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
: Å
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192818*&
output_shapes
:ÿÿÿÿÿÿÿÿÿd*#
then_branchR
cond_true_192817Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¡

+__inference_sequential_layer_call_fn_191946

inputs!
unknown: 
	unknown_0: 
	unknown_1: #
	unknown_2: 0
	unknown_3:0
	unknown_4: 
	unknown_5:	°	x
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
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
ß
,sequential_spiking_b_relu_2_cond_true_191029N
Jsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_cast0
,sequential_spiking_b_relu_2_cond_placeholder-
)sequential_spiking_b_relu_2_cond_identity³
)sequential/spiking_b_relu_2/cond/IdentityIdentityJsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"_
)sequential_spiking_b_relu_2_cond_identity2sequential/spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
Ï
é
-sequential_spiking_b_relu_4_cond_false_1911120
,sequential_spiking_b_relu_4_cond_placeholderW
Ssequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_clip_by_value-
)sequential_spiking_b_relu_4_cond_identity¼
)sequential/spiking_b_relu_4/cond/IdentityIdentitySsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"_
)sequential_spiking_b_relu_4_cond_identity2sequential/spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

¶
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_191405

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT^
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
:ÿÿÿÿÿÿÿÿÿTT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTd
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
: Å
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191393*&
output_shapes
:ÿÿÿÿÿÿÿÿÿT*#
then_branchR
cond_true_191392Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTe
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
Ý
]
cond_false_192519
cond_placeholder
cond_identity_clip_by_value
cond_identityp
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
È	
ó
A__inference_dense_layer_call_and_return_conditional_losses_191301

inputs1
matmul_readvariableop_resource:	°	x-
biasadd_readvariableop_resource:x
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	°	x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
ê

)__inference_conv2d_1_layer_call_fn_192550

inputs!
unknown: 0
	unknown_0:0
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢
S
cond_true_192750
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT

¶
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_192608

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0g
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0\
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
: Õ
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192596*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ

0*#
then_branchR
cond_true_192595b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

0: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_192618

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
]
cond_false_192818
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
¢
S
cond_true_191333
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
Æ	
ô
C__inference_dense_1_layer_call_and_return_conditional_losses_192715

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
þÜ
¶"
"__inference__traced_restore_193206
file_prefix8
assignvariableop_conv2d_kernel: ,
assignvariableop_1_conv2d_bias: %
assignvariableop_2_variable: <
"assignvariableop_3_conv2d_1_kernel: 0.
 assignvariableop_4_conv2d_1_bias:0'
assignvariableop_5_variable_1: 2
assignvariableop_6_dense_kernel:	°	x+
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
4assignvariableop_30_adadelta_dense_kernel_accum_grad:	°	x@
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
3assignvariableop_45_adadelta_dense_kernel_accum_var:	°	x?
1assignvariableop_46_adadelta_dense_bias_accum_var:x;
1assignvariableop_47_adadelta_variable_accum_var_2: G
5assignvariableop_48_adadelta_dense_1_kernel_accum_var:xTA
3assignvariableop_49_adadelta_dense_1_bias_accum_var:T;
1assignvariableop_50_adadelta_variable_accum_var_3: G
5assignvariableop_51_adadelta_dense_2_kernel_accum_var:TdA
3assignvariableop_52_adadelta_dense_2_bias_accum_var:d;
1assignvariableop_53_adadelta_variable_accum_var_4: 
identity_55¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*® 
value¤ B¡ 7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-7/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-9/sharpness/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-7/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-9/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHß
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ´
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ò
_output_shapesß
Ü:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
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
:¦
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adadelta_conv2d_kernel_accum_gradIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adadelta_conv2d_bias_accum_gradIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adadelta_variable_accum_gradIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adadelta_conv2d_1_kernel_accum_gradIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adadelta_conv2d_1_bias_accum_gradIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adadelta_variable_accum_grad_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adadelta_dense_kernel_accum_gradIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adadelta_dense_bias_accum_gradIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adadelta_variable_accum_grad_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adadelta_dense_1_kernel_accum_gradIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adadelta_dense_1_bias_accum_gradIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adadelta_variable_accum_grad_3Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adadelta_dense_2_kernel_accum_gradIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adadelta_dense_2_bias_accum_gradIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adadelta_variable_accum_grad_4Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adadelta_conv2d_kernel_accum_varIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:£
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
:§
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adadelta_conv2d_1_kernel_accum_varIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adadelta_conv2d_1_bias_accum_varIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_44AssignVariableOp1assignvariableop_44_adadelta_variable_accum_var_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adadelta_dense_kernel_accum_varIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_46AssignVariableOp1assignvariableop_46_adadelta_dense_bias_accum_varIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_adadelta_variable_accum_var_2Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adadelta_dense_1_kernel_accum_varIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adadelta_dense_1_bias_accum_varIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_50AssignVariableOp1assignvariableop_50_adadelta_variable_accum_var_3Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adadelta_dense_2_kernel_accum_varIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adadelta_dense_2_bias_accum_varIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_adadelta_variable_accum_var_4Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ó	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: à	
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
á
²
"spiking_b_relu_4_cond_false_192184%
!spiking_b_relu_4_cond_placeholderA
=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value"
spiking_b_relu_4_cond_identity
spiking_b_relu_4/cond/IdentityIdentity=spiking_b_relu_4_cond_identity_spiking_b_relu_4_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"I
spiking_b_relu_4_cond_identity'spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
¥

û
B__inference_conv2d_layer_call_and_return_conditional_losses_192483

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
S
cond_true_192683
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
á
²
"spiking_b_relu_3_cond_false_192143%
!spiking_b_relu_3_cond_placeholderA
=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value"
spiking_b_relu_3_cond_identity
spiking_b_relu_3/cond/IdentityIdentity=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
­
D
(__inference_flatten_layer_call_fn_192623

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	* 
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
:ÿÿÿÿÿÿÿÿÿ°	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ê
S
cond_true_192518
cond_identity_cast
cond_placeholder
cond_identityg
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

´
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_192531

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
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
:ÿÿÿÿÿÿÿÿÿ \
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
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
: Õ
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192519*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
then_branchR
cond_true_192518b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¶
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_192763

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT^
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
:ÿÿÿÿÿÿÿÿÿTT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTd
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
: Å
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_192751*&
output_shapes
:ÿÿÿÿÿÿÿÿÿT*#
then_branchR
cond_true_192750Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTe
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
Æ	
ô
C__inference_dense_1_layer_call_and_return_conditional_losses_191360

inputs0
matmul_readvariableop_resource:xT-
biasadd_readvariableop_resource:T
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
È	
ó
A__inference_dense_layer_call_and_return_conditional_losses_192648

inputs1
matmul_readvariableop_resource:	°	x-
biasadd_readvariableop_resource:x
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	°	x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
ì	
´
K__inference_softmax__decode_layer_call_and_return_conditional_losses_192849

inputs0
matmul_readvariableop_resource:d

identity¢MatMul/ReadVariableOpJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0j
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ê
S
cond_true_191205
cond_identity_cast
cond_placeholder
cond_identityg
cond/IdentityIdentitycond_identity_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Ûì
º
!__inference__wrapped_model_191132
conv2d_inputJ
0sequential_conv2d_conv2d_readvariableop_resource: ?
1sequential_conv2d_biasadd_readvariableop_resource: ;
1sequential_spiking_b_relu_readvariableop_resource: L
2sequential_conv2d_1_conv2d_readvariableop_resource: 0A
3sequential_conv2d_1_biasadd_readvariableop_resource:0=
3sequential_spiking_b_relu_1_readvariableop_resource: B
/sequential_dense_matmul_readvariableop_resource:	°	x>
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
identity¢(sequential/conv2d/BiasAdd/ReadVariableOp¢'sequential/conv2d/Conv2D/ReadVariableOp¢*sequential/conv2d_1/BiasAdd/ReadVariableOp¢)sequential/conv2d_1/Conv2D/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢)sequential/dense_2/BiasAdd/ReadVariableOp¢(sequential/dense_2/MatMul/ReadVariableOp¢0sequential/softmax__decode/MatMul/ReadVariableOp¢.sequential/spiking_b_relu/Equal/ReadVariableOp¢(sequential/spiking_b_relu/ReadVariableOp¢0sequential/spiking_b_relu_1/Equal/ReadVariableOp¢*sequential/spiking_b_relu_1/ReadVariableOp¢0sequential/spiking_b_relu_2/Equal/ReadVariableOp¢*sequential/spiking_b_relu_2/ReadVariableOp¢0sequential/spiking_b_relu_3/Equal/ReadVariableOp¢*sequential/spiking_b_relu_3/ReadVariableOp¢0sequential/spiking_b_relu_4/Equal/ReadVariableOp¢*sequential/spiking_b_relu_4/ReadVariableOp 
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ã
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
(sequential/spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ç
&sequential/spiking_b_relu/GreaterEqualGreaterEqual"sequential/conv2d/BiasAdd:output:01sequential/spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential/spiking_b_relu/CastCast*sequential/spiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(sequential/spiking_b_relu/ReadVariableOpReadVariableOp1sequential_spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0d
sequential/spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
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
 *   ?°
sequential/spiking_b_relu/sub_1Sub"sequential/conv2d/BiasAdd:output:0*sequential/spiking_b_relu/sub_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
sequential/spiking_b_relu/mulMul%sequential/spiking_b_relu/truediv:z:0#sequential/spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
!sequential/spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?±
sequential/spiking_b_relu/add_1AddV2!sequential/spiking_b_relu/mul:z:0*sequential/spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
1sequential/spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Õ
/sequential/spiking_b_relu/clip_by_value/MinimumMinimum#sequential/spiking_b_relu/add_1:z:0:sequential/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
)sequential/spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Õ
'sequential/spiking_b_relu/clip_by_valueMaximum3sequential/spiking_b_relu/clip_by_value/Minimum:z:02sequential/spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
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
: ñ
sequential/spiking_b_relu/condStatelessIf#sequential/spiking_b_relu/Equal:z:0"sequential/spiking_b_relu/Cast:y:0+sequential/spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *>
else_branch/R-
+sequential_spiking_b_relu_cond_false_190944*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ *=
then_branch.R,
*sequential_spiking_b_relu_cond_true_190943
'sequential/spiking_b_relu/cond/IdentityIdentity'sequential/spiking_b_relu/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ê
 sequential/max_pooling2d/MaxPoolMaxPool0sequential/spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
¤
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0å
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0*
paddingVALID*
strides

*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0¹
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0o
*sequential/spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Í
(sequential/spiking_b_relu_1/GreaterEqualGreaterEqual$sequential/conv2d_1/BiasAdd:output:03sequential/spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
 sequential/spiking_b_relu_1/CastCast,sequential/spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
 *  ?§
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
 *  ?¤
#sequential/spiking_b_relu_1/truedivRealDiv.sequential/spiking_b_relu_1/truediv/x:output:0#sequential/spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
!sequential/spiking_b_relu_1/sub_1Sub$sequential/conv2d_1/BiasAdd:output:0,sequential/spiking_b_relu_1/sub_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0°
sequential/spiking_b_relu_1/mulMul'sequential/spiking_b_relu_1/truediv:z:0%sequential/spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0h
#sequential/spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?·
!sequential/spiking_b_relu_1/add_1AddV2#sequential/spiking_b_relu_1/mul:z:0,sequential/spiking_b_relu_1/add_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0x
3sequential/spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Û
1sequential/spiking_b_relu_1/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_1/add_1:z:0<sequential/spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0p
+sequential/spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Û
)sequential/spiking_b_relu_1/clip_by_valueMaximum5sequential/spiking_b_relu_1/clip_by_value/Minimum:z:04sequential/spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
 *  ?³
!sequential/spiking_b_relu_1/EqualEqual8sequential/spiking_b_relu_1/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: ý
 sequential/spiking_b_relu_1/condStatelessIf%sequential/spiking_b_relu_1/Equal:z:0$sequential/spiking_b_relu_1/Cast:y:0-sequential/spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0* 
_read_only_resource_inputs
 *@
else_branch1R/
-sequential_spiking_b_relu_1_cond_false_190986*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ

0*?
then_branch0R.
,sequential_spiking_b_relu_1_cond_true_190985
)sequential/spiking_b_relu_1/cond/IdentityIdentity)sequential/spiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0Î
"sequential/max_pooling2d_1/MaxPoolMaxPool2sequential/spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
ksize
*
paddingVALID*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°  ¨
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_1/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	°	x*
dtype0¨
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxo
*sequential/spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Â
(sequential/spiking_b_relu_2/GreaterEqualGreaterEqual!sequential/dense/BiasAdd:output:03sequential/spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 sequential/spiking_b_relu_2/CastCast,sequential/spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
*sequential/spiking_b_relu_2/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
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
 *  ?¤
#sequential/spiking_b_relu_2/truedivRealDiv.sequential/spiking_b_relu_2/truediv/x:output:0#sequential/spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?«
!sequential/spiking_b_relu_2/sub_1Sub!sequential/dense/BiasAdd:output:0,sequential/spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¨
sequential/spiking_b_relu_2/mulMul'sequential/spiking_b_relu_2/truediv:z:0%sequential/spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxh
#sequential/spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
!sequential/spiking_b_relu_2/add_1AddV2#sequential/spiking_b_relu_2/mul:z:0,sequential/spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
3sequential/spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ó
1sequential/spiking_b_relu_2/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_2/add_1:z:0<sequential/spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxp
+sequential/spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ó
)sequential/spiking_b_relu_2/clip_by_valueMaximum5sequential/spiking_b_relu_2/clip_by_value/Minimum:z:04sequential/spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
0sequential/spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!sequential/spiking_b_relu_2/EqualEqual8sequential/spiking_b_relu_2/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: í
 sequential/spiking_b_relu_2/condStatelessIf%sequential/spiking_b_relu_2/Equal:z:0$sequential/spiking_b_relu_2/Cast:y:0-sequential/spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *@
else_branch1R/
-sequential_spiking_b_relu_2_cond_false_191030*&
output_shapes
:ÿÿÿÿÿÿÿÿÿx*?
then_branch0R.
,sequential_spiking_b_relu_2_cond_true_191029
)sequential/spiking_b_relu_2/cond/IdentityIdentity)sequential/spiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0»
sequential/dense_1/MatMulMatMul2sequential/spiking_b_relu_2/cond/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0¯
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTo
*sequential/spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ä
(sequential/spiking_b_relu_3/GreaterEqualGreaterEqual#sequential/dense_1/BiasAdd:output:03sequential/spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 sequential/spiking_b_relu_3/CastCast,sequential/spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
*sequential/spiking_b_relu_3/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
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
 *  ?¤
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
:ÿÿÿÿÿÿÿÿÿT¨
sequential/spiking_b_relu_3/mulMul'sequential/spiking_b_relu_3/truediv:z:0%sequential/spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTh
#sequential/spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
!sequential/spiking_b_relu_3/add_1AddV2#sequential/spiking_b_relu_3/mul:z:0,sequential/spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTx
3sequential/spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ó
1sequential/spiking_b_relu_3/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_3/add_1:z:0<sequential/spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTp
+sequential/spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ó
)sequential/spiking_b_relu_3/clip_by_valueMaximum5sequential/spiking_b_relu_3/clip_by_value/Minimum:z:04sequential/spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
0sequential/spiking_b_relu_3/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_3/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!sequential/spiking_b_relu_3/EqualEqual8sequential/spiking_b_relu_3/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_3/Equal/y:output:0*
T0*
_output_shapes
: í
 sequential/spiking_b_relu_3/condStatelessIf%sequential/spiking_b_relu_3/Equal:z:0$sequential/spiking_b_relu_3/Cast:y:0-sequential/spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT* 
_read_only_resource_inputs
 *@
else_branch1R/
-sequential_spiking_b_relu_3_cond_false_191071*&
output_shapes
:ÿÿÿÿÿÿÿÿÿT*?
then_branch0R.
,sequential_spiking_b_relu_3_cond_true_191070
)sequential/spiking_b_relu_3/cond/IdentityIdentity)sequential/spiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype0»
sequential/dense_2/MatMulMatMul2sequential/spiking_b_relu_3/cond/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0¯
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
*sequential/spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ä
(sequential/spiking_b_relu_4/GreaterEqualGreaterEqual#sequential/dense_2/BiasAdd:output:03sequential/spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 sequential/spiking_b_relu_4/CastCast,sequential/spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*sequential/spiking_b_relu_4/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
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
 *  ?¤
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
:ÿÿÿÿÿÿÿÿÿd¨
sequential/spiking_b_relu_4/mulMul'sequential/spiking_b_relu_4/truediv:z:0%sequential/spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
#sequential/spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
!sequential/spiking_b_relu_4/add_1AddV2#sequential/spiking_b_relu_4/mul:z:0,sequential/spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
3sequential/spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ó
1sequential/spiking_b_relu_4/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_4/add_1:z:0<sequential/spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
+sequential/spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ó
)sequential/spiking_b_relu_4/clip_by_valueMaximum5sequential/spiking_b_relu_4/clip_by_value/Minimum:z:04sequential/spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
0sequential/spiking_b_relu_4/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_4_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_4/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!sequential/spiking_b_relu_4/EqualEqual8sequential/spiking_b_relu_4/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_4/Equal/y:output:0*
T0*
_output_shapes
: í
 sequential/spiking_b_relu_4/condStatelessIf%sequential/spiking_b_relu_4/Equal:z:0$sequential/spiking_b_relu_4/Cast:y:0-sequential/spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *@
else_branch1R/
-sequential_spiking_b_relu_4_cond_false_191112*&
output_shapes
:ÿÿÿÿÿÿÿÿÿd*?
then_branch0R.
,sequential_spiking_b_relu_4_cond_true_191111
)sequential/spiking_b_relu_4/cond/IdentityIdentity)sequential/spiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
 sequential/softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @¶
sequential/softmax__decode/mulMul)sequential/softmax__decode/mul/x:output:02sequential/spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
 sequential/softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¦
sequential/softmax__decode/subSub"sequential/softmax__decode/mul:z:0)sequential/softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
0sequential/softmax__decode/MatMul/ReadVariableOpReadVariableOp9sequential_softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0»
!sequential/softmax__decode/MatMulMatMul"sequential/softmax__decode/sub:z:08sequential/softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"sequential/softmax__decode/SoftmaxSoftmax+sequential/softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
IdentityIdentity,sequential/softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp1^sequential/softmax__decode/MatMul/ReadVariableOp/^sequential/spiking_b_relu/Equal/ReadVariableOp)^sequential/spiking_b_relu/ReadVariableOp1^sequential/spiking_b_relu_1/Equal/ReadVariableOp+^sequential/spiking_b_relu_1/ReadVariableOp1^sequential/spiking_b_relu_2/Equal/ReadVariableOp+^sequential/spiking_b_relu_2/ReadVariableOp1^sequential/spiking_b_relu_3/Equal/ReadVariableOp+^sequential/spiking_b_relu_3/ReadVariableOp1^sequential/spiking_b_relu_4/Equal/ReadVariableOp+^sequential/spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2T
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
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input

¶
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_191464

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
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
:ÿÿÿÿÿÿÿÿÿdT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
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
: Å
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191452*&
output_shapes
:ÿÿÿÿÿÿÿÿÿd*#
then_branchR
cond_true_191451Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Î
¨
!spiking_b_relu_3_cond_true_1923638
4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast%
!spiking_b_relu_3_cond_placeholder"
spiking_b_relu_3_cond_identity
spiking_b_relu_3/cond/IdentityIdentity4spiking_b_relu_3_cond_identity_spiking_b_relu_3_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
¡
}
1__inference_spiking_b_relu_4_layer_call_fn_192791

inputs
unknown: 
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*#
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
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ö
¨
!spiking_b_relu_1_cond_true_1922788
4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast%
!spiking_b_relu_1_cond_placeholder"
spiking_b_relu_1_cond_identity
spiking_b_relu_1/cond/IdentityIdentity4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0

´
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_191218

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
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
:ÿÿÿÿÿÿÿÿÿ \
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?c
add_1AddV2mul:z:0add_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
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
: Õ
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191206*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
then_branchR
cond_true_191205b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
µ
]
cond_false_191334
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
Î
¨
!spiking_b_relu_2_cond_true_1921018
4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast%
!spiking_b_relu_2_cond_placeholder"
spiking_b_relu_2_cond_identity
spiking_b_relu_2/cond/IdentityIdentity4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
¨

ý
D__inference_conv2d_1_layer_call_and_return_conditional_losses_192560

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
]
cond_false_192596
cond_placeholder
cond_identity_clip_by_value
cond_identityp
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
Ý
]
cond_false_191206
cond_placeholder
cond_identity_clip_by_value
cond_identityp
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Ï
é
-sequential_spiking_b_relu_3_cond_false_1910710
,sequential_spiking_b_relu_3_cond_placeholderW
Ssequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_clip_by_value-
)sequential_spiking_b_relu_3_cond_identity¼
)sequential/spiking_b_relu_3/cond/IdentityIdentitySsequential_spiking_b_relu_3_cond_identity_sequential_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"_
)sequential_spiking_b_relu_3_cond_identity2sequential/spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
õ
¨
 spiking_b_relu_cond_false_192016#
spiking_b_relu_cond_placeholder=
9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value 
spiking_b_relu_cond_identity
spiking_b_relu/cond/IdentityIdentity9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
ã
ß
+sequential_spiking_b_relu_cond_false_190944.
*sequential_spiking_b_relu_cond_placeholderS
Osequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_clip_by_value+
'sequential_spiking_b_relu_cond_identity¾
'sequential/spiking_b_relu/cond/IdentityIdentityOsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "[
'sequential_spiking_b_relu_cond_identity0sequential/spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

¶
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_191278

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0g
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0\
mulMultruediv:z:0	sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
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
: Õ
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191266*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ

0*#
then_branchR
cond_true_191265b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

0: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
 
_user_specified_nameinputs
¿

&__inference_dense_layer_call_fn_192638

inputs
unknown:	°	x
	unknown_0:x
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
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
:ÿÿÿÿÿÿÿÿÿx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ°	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
Ý
]
cond_false_191266
cond_placeholder
cond_identity_clip_by_value
cond_identityp
cond/IdentityIdentitycond_identity_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
ä
ß
,sequential_spiking_b_relu_1_cond_true_190985N
Jsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_cast0
,sequential_spiking_b_relu_1_cond_placeholder-
)sequential_spiking_b_relu_1_cond_identity»
)sequential/spiking_b_relu_1/cond/IdentityIdentityJsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0"_
)sequential_spiking_b_relu_1_cond_identity2sequential/spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ

0:ÿÿÿÿÿÿÿÿÿ

0:5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
Ð
Õ
*sequential_spiking_b_relu_cond_true_190943J
Fsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_cast.
*sequential_spiking_b_relu_cond_placeholder+
'sequential_spiking_b_relu_cond_identityµ
'sequential/spiking_b_relu/cond/IdentityIdentityFsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "[
'sequential_spiking_b_relu_cond_identity0sequential/spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
â

spiking_b_relu_cond_true_1920154
0spiking_b_relu_cond_identity_spiking_b_relu_cast#
spiking_b_relu_cond_placeholder 
spiking_b_relu_cond_identity
spiking_b_relu/cond/IdentityIdentity0spiking_b_relu_cond_identity_spiking_b_relu_cast*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
¼
ß
,sequential_spiking_b_relu_4_cond_true_191111N
Jsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_cast0
,sequential_spiking_b_relu_4_cond_placeholder-
)sequential_spiking_b_relu_4_cond_identity³
)sequential/spiking_b_relu_4/cond/IdentityIdentityJsequential_spiking_b_relu_4_cond_identity_sequential_spiking_b_relu_4_cast*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"_
)sequential_spiking_b_relu_4_cond_identity2sequential/spiking_b_relu_4/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ï
é
-sequential_spiking_b_relu_2_cond_false_1910300
,sequential_spiking_b_relu_2_cond_placeholderW
Ssequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_clip_by_value-
)sequential_spiking_b_relu_2_cond_identity¼
)sequential/spiking_b_relu_2/cond/IdentityIdentitySsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx"_
)sequential_spiking_b_relu_2_cond_identity2sequential/spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿx:ÿÿÿÿÿÿÿÿÿx:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
À

(__inference_dense_1_layer_call_fn_192705

inputs
unknown:xT
	unknown_0:T
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
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
:ÿÿÿÿÿÿÿÿÿT`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
Å
_
C__inference_flatten_layer_call_and_return_conditional_losses_192629

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs

¶
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_191346

inputs!
readvariableop_resource: 
identity¢Equal/ReadVariableOp¢ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
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
:ÿÿÿÿÿÿÿÿÿxT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
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
: Å
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_191334*&
output_shapes
:ÿÿÿÿÿÿÿÿÿx*#
then_branchR
cond_true_191333Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxe
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
ì	
´
K__inference_softmax__decode_layer_call_and_return_conditional_losses_191480

inputs0
matmul_readvariableop_resource:d

identity¢MatMul/ReadVariableOpJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0j
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Á
}
1__inference_spiking_b_relu_1_layer_call_fn_192569

inputs
unknown: 
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

0: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
 
_user_specified_nameinputs
À

(__inference_dense_2_layer_call_fn_192772

inputs
unknown:Td
	unknown_0:d
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
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
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿT: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
Ç
Â
F__inference_sequential_layer_call_and_return_conditional_losses_192425

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 0
&spiking_b_relu_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: 06
(conv2d_1_biasadd_readvariableop_resource:02
(spiking_b_relu_1_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	°	x3
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
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢%softmax__decode/MatMul/ReadVariableOp¢#spiking_b_relu/Equal/ReadVariableOp¢spiking_b_relu/ReadVariableOp¢%spiking_b_relu_1/Equal/ReadVariableOp¢spiking_b_relu_1/ReadVariableOp¢%spiking_b_relu_2/Equal/ReadVariableOp¢spiking_b_relu_2/ReadVariableOp¢%spiking_b_relu_3/Equal/ReadVariableOp¢spiking_b_relu_3/ReadVariableOp¢%spiking_b_relu_4/Equal/ReadVariableOp¢spiking_b_relu_4/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0§
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
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
:ÿÿÿÿÿÿÿÿÿ b
spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
spiking_b_relu/GreaterEqualGreaterEqualconv2d/BiasAdd:output:0&spiking_b_relu/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
spiking_b_relu/CastCastspiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
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
:ÿÿÿÿÿÿÿÿÿ 
spiking_b_relu/mulMulspiking_b_relu/truediv:z:0spiking_b_relu/sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu/add_1AddV2spiking_b_relu/mul:z:0spiking_b_relu/add_1/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
&spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?´
$spiking_b_relu/clip_by_value/MinimumMinimumspiking_b_relu/add_1:z:0/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
spiking_b_relu/clip_by_valueMaximum(spiking_b_relu/clip_by_value/Minimum:z:0'spiking_b_relu/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
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
: ¯
spiking_b_relu/condStatelessIfspiking_b_relu/Equal:z:0spiking_b_relu/Cast:y:0 spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *3
else_branch$R"
 spiking_b_relu_cond_false_192237*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ *2
then_branch#R!
spiking_b_relu_cond_true_192236
spiking_b_relu/cond/IdentityIdentityspiking_b_relu/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ´
max_pooling2d/MaxPoolMaxPool%spiking_b_relu/cond/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0Ä
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0d
spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¬
spiking_b_relu_1/GreaterEqualGreaterEqualconv2d_1/BiasAdd:output:0(spiking_b_relu_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0
spiking_b_relu_1/CastCast!spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0
spiking_b_relu_1/mulMulspiking_b_relu_1/truediv:z:0spiking_b_relu_1/sub_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ

0m
(spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?º
&spiking_b_relu_1/clip_by_value/MinimumMinimumspiking_b_relu_1/add_1:z:01spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0e
 spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    º
spiking_b_relu_1/clip_by_valueMaximum*spiking_b_relu_1/clip_by_value/Minimum:z:0)spiking_b_relu_1/clip_by_value/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
: »
spiking_b_relu_1/condStatelessIfspiking_b_relu_1/Equal:z:0spiking_b_relu_1/Cast:y:0"spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_1_cond_false_192279*.
output_shapes
:ÿÿÿÿÿÿÿÿÿ

0*4
then_branch%R#
!spiking_b_relu_1_cond_true_192278
spiking_b_relu_1/cond/IdentityIdentityspiking_b_relu_1/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0¸
max_pooling2d_1/MaxPoolMaxPool'spiking_b_relu_1/cond/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ°  
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	°	x*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¡
spiking_b_relu_2/GreaterEqualGreaterEqualdense/BiasAdd:output:0(spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
spiking_b_relu_2/CastCast!spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
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
:ÿÿÿÿÿÿÿÿÿx
spiking_b_relu_2/mulMulspiking_b_relu_2/truediv:z:0spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx]
spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_2/add_1AddV2spiking_b_relu_2/mul:z:0!spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxm
(spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
&spiking_b_relu_2/clip_by_value/MinimumMinimumspiking_b_relu_2/add_1:z:01spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxe
 spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
spiking_b_relu_2/clip_by_valueMaximum*spiking_b_relu_2/clip_by_value/Minimum:z:0)spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
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
: «
spiking_b_relu_2/condStatelessIfspiking_b_relu_2/Equal:z:0spiking_b_relu_2/Cast:y:0"spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_2_cond_false_192323*&
output_shapes
:ÿÿÿÿÿÿÿÿÿx*4
then_branch%R#
!spiking_b_relu_2_cond_true_192322|
spiking_b_relu_2/cond/IdentityIdentityspiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype0
dense_1/MatMulMatMul'spiking_b_relu_2/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTd
spiking_b_relu_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
spiking_b_relu_3/GreaterEqualGreaterEqualdense_1/BiasAdd:output:0(spiking_b_relu_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
spiking_b_relu_3/CastCast!spiking_b_relu_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
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
:ÿÿÿÿÿÿÿÿÿT
spiking_b_relu_3/mulMulspiking_b_relu_3/truediv:z:0spiking_b_relu_3/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT]
spiking_b_relu_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_3/add_1AddV2spiking_b_relu_3/mul:z:0!spiking_b_relu_3/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTm
(spiking_b_relu_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
&spiking_b_relu_3/clip_by_value/MinimumMinimumspiking_b_relu_3/add_1:z:01spiking_b_relu_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿTe
 spiking_b_relu_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
spiking_b_relu_3/clip_by_valueMaximum*spiking_b_relu_3/clip_by_value/Minimum:z:0)spiking_b_relu_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
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
: «
spiking_b_relu_3/condStatelessIfspiking_b_relu_3/Equal:z:0spiking_b_relu_3/Cast:y:0"spiking_b_relu_3/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_3_cond_false_192364*&
output_shapes
:ÿÿÿÿÿÿÿÿÿT*4
then_branch%R#
!spiking_b_relu_3_cond_true_192363|
spiking_b_relu_3/cond/IdentityIdentityspiking_b_relu_3/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:Td*
dtype0
dense_2/MatMulMatMul'spiking_b_relu_3/cond/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
spiking_b_relu_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
spiking_b_relu_4/GreaterEqualGreaterEqualdense_2/BiasAdd:output:0(spiking_b_relu_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
spiking_b_relu_4/CastCast!spiking_b_relu_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
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
:ÿÿÿÿÿÿÿÿÿd
spiking_b_relu_4/mulMulspiking_b_relu_4/truediv:z:0spiking_b_relu_4/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
spiking_b_relu_4/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
spiking_b_relu_4/add_1AddV2spiking_b_relu_4/mul:z:0!spiking_b_relu_4/add_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(spiking_b_relu_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
&spiking_b_relu_4/clip_by_value/MinimumMinimumspiking_b_relu_4/add_1:z:01spiking_b_relu_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
 spiking_b_relu_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
spiking_b_relu_4/clip_by_valueMaximum*spiking_b_relu_4/clip_by_value/Minimum:z:0)spiking_b_relu_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
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
: «
spiking_b_relu_4/condStatelessIfspiking_b_relu_4/Equal:z:0spiking_b_relu_4/Cast:y:0"spiking_b_relu_4/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *5
else_branch&R$
"spiking_b_relu_4_cond_false_192405*&
output_shapes
:ÿÿÿÿÿÿÿÿÿd*4
then_branch%R#
!spiking_b_relu_4_cond_true_192404|
spiking_b_relu_4/cond/IdentityIdentityspiking_b_relu_4/cond:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
softmax__decode/mulMulsoftmax__decode/mul/x:output:0'spiking_b_relu_4/cond/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
softmax__decode/subSubsoftmax__decode/mul:z:0softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%softmax__decode/MatMul/ReadVariableOpReadVariableOp.softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0
softmax__decode/MatMulMatMulsoftmax__decode/sub:z:0-softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
softmax__decode/SoftmaxSoftmax softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
IdentityIdentity!softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^softmax__decode/MatMul/ReadVariableOp$^spiking_b_relu/Equal/ReadVariableOp^spiking_b_relu/ReadVariableOp&^spiking_b_relu_1/Equal/ReadVariableOp ^spiking_b_relu_1/ReadVariableOp&^spiking_b_relu_2/Equal/ReadVariableOp ^spiking_b_relu_2/ReadVariableOp&^spiking_b_relu_3/Equal/ReadVariableOp ^spiking_b_relu_3/ReadVariableOp&^spiking_b_relu_4/Equal/ReadVariableOp ^spiking_b_relu_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
²
"spiking_b_relu_3_cond_false_192364%
!spiking_b_relu_3_cond_placeholderA
=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value"
spiking_b_relu_3_cond_identity
spiking_b_relu_3/cond/IdentityIdentity=spiking_b_relu_3_cond_identity_spiking_b_relu_3_clip_by_value*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT"I
spiking_b_relu_3_cond_identity'spiking_b_relu_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿT:ÿÿÿÿÿÿÿÿÿT:- )
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
õ
¨
 spiking_b_relu_cond_false_192237#
spiking_b_relu_cond_placeholder=
9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value 
spiking_b_relu_cond_identity
spiking_b_relu/cond/IdentityIdentity9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :5 1
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
¡
}
1__inference_spiking_b_relu_2_layer_call_fn_192657

inputs
unknown: 
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*#
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
:ÿÿÿÿÿÿÿÿÿx`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿx: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
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
	unknown_5:	°	x
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
identity¢StatefulPartitionedCall÷
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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
serving_default°
M
conv2d_input=
serving_default_conv2d_input:0ÿÿÿÿÿÿÿÿÿC
softmax__decode0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:Ðé
¿
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
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
´
 	sharpness
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
»

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
´
5	sharpness
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
´
P	sharpness
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Wkernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
´
_	sharpness
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
»

fkernel
gbias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
´
n	sharpness
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
¸
u_rescaled_key
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
ï
|iter
	}decay
~learning_rate
rho
accum_grad×
accum_gradØ 
accum_gradÙ-
accum_gradÚ.
accum_gradÛ5
accum_gradÜH
accum_gradÝI
accum_gradÞP
accum_gradßW
accum_gradàX
accum_gradá_
accum_gradâf
accum_gradãg
accum_gradän
accum_gradå	accum_varæ	accum_varç 	accum_varè-	accum_varé.	accum_varê5	accum_varëH	accum_varìI	accum_varíP	accum_varîW	accum_varïX	accum_varð_	accum_varñf	accum_varòg	accum_varón	accum_varô"
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
Ï
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
ú2÷
+__inference_sequential_layer_call_fn_191520
+__inference_sequential_layer_call_fn_191946
+__inference_sequential_layer_call_fn_191983
+__inference_sequential_layer_call_fn_191785À
·²³
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
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_192204
F__inference_sequential_layer_call_and_return_conditional_losses_192425
F__inference_sequential_layer_call_and_return_conditional_losses_191835
F__inference_sequential_layer_call_and_return_conditional_losses_191885À
·²³
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
kwonlydefaultsª 
annotationsª *
 
ÑBÎ
!__inference__wrapped_model_191132conv2d_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
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
Ñ2Î
'__inference_conv2d_layer_call_fn_192473¢
²
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
annotationsª *
 
ì2é
B__inference_conv2d_layer_call_and_return_conditional_losses_192483¢
²
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
annotationsª *
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
²
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
Ù2Ö
/__inference_spiking_b_relu_layer_call_fn_192492¢
²
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
annotationsª *
 
ô2ñ
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_192531¢
²
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
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
Ø2Õ
.__inference_max_pooling2d_layer_call_fn_192536¢
²
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
annotationsª *
 
ó2ð
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_192541¢
²
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
annotationsª *
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
²
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
Ó2Ð
)__inference_conv2d_1_layer_call_fn_192550¢
²
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
annotationsª *
 
î2ë
D__inference_conv2d_1_layer_call_and_return_conditional_losses_192560¢
²
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
annotationsª *
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
²
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
Û2Ø
1__inference_spiking_b_relu_1_layer_call_fn_192569¢
²
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
annotationsª *
 
ö2ó
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_192608¢
²
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_1_layer_call_fn_192613¢
²
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
annotationsª *
 
õ2ò
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_192618¢
²
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_flatten_layer_call_fn_192623¢
²
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
annotationsª *
 
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_192629¢
²
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
annotationsª *
 
:	°	x2dense/kernel
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
²
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_layer_call_fn_192638¢
²
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
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_192648¢
²
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
annotationsª *
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
²
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_spiking_b_relu_2_layer_call_fn_192657¢
²
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
annotationsª *
 
ö2ó
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_192696¢
²
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
annotationsª *
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
²
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_1_layer_call_fn_192705¢
²
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
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_192715¢
²
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
annotationsª *
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
²
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_spiking_b_relu_3_layer_call_fn_192724¢
²
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
annotationsª *
 
ö2ó
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_192763¢
²
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
annotationsª *
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
²
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_2_layer_call_fn_192772¢
²
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
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_192782¢
²
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
annotationsª *
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
²
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_spiking_b_relu_4_layer_call_fn_192791¢
²
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
annotationsª *
 
ö2ó
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_192830¢
²
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
annotationsª *
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
²
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_softmax__decode_layer_call_fn_192837¢
²
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
annotationsª *
 
õ2ò
K__inference_softmax__decode_layer_call_and_return_conditional_losses_192849¢
²
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
annotationsª *
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
Ì0
Í1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÐBÍ
$__inference_signature_wrapper_192464conv2d_input"
²
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
annotationsª *
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

Îtotal

Ïcount
Ð	variables
Ñ	keras_api"
_tf_keras_metric
c

Òtotal

Ócount
Ô
_fn_kwargs
Õ	variables
Ö	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
Î0
Ï1"
trackable_list_wrapper
.
Ð	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ò0
Ó1"
trackable_list_wrapper
.
Õ	variables"
_generic_user_object
9:7 2!Adadelta/conv2d/kernel/accum_grad
+:) 2Adadelta/conv2d/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
;:9 02#Adadelta/conv2d_1/kernel/accum_grad
-:+02!Adadelta/conv2d_1/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
1:/	°	x2 Adadelta/dense/kernel/accum_grad
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
0:.	°	x2Adadelta/dense/kernel/accum_var
):'x2Adadelta/dense/bias/accum_var
#:! 2Adadelta/Variable/accum_var
1:/xT2!Adadelta/dense_1/kernel/accum_var
+:)T2Adadelta/dense_1/bias/accum_var
#:! 2Adadelta/Variable/accum_var
1:/Td2!Adadelta/dense_2/kernel/accum_var
+:)d2Adadelta/dense_2/bias/accum_var
#:! 2Adadelta/Variable/accum_varº
!__inference__wrapped_model_191132 -.5HIPWX_fgnu=¢:
3¢0
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
ª "Aª>
<
softmax__decode)&
softmax__decodeÿÿÿÿÿÿÿÿÿ
´
D__inference_conv2d_1_layer_call_and_return_conditional_losses_192560l-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

0
 
)__inference_conv2d_1_layer_call_fn_192550_-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ

0²
B__inference_conv2d_layer_call_and_return_conditional_losses_192483l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
'__inference_conv2d_layer_call_fn_192473_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ £
C__inference_dense_1_layer_call_and_return_conditional_losses_192715\WX/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿx
ª "%¢"

0ÿÿÿÿÿÿÿÿÿT
 {
(__inference_dense_1_layer_call_fn_192705OWX/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿx
ª "ÿÿÿÿÿÿÿÿÿT£
C__inference_dense_2_layer_call_and_return_conditional_losses_192782\fg/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 {
(__inference_dense_2_layer_call_fn_192772Ofg/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "ÿÿÿÿÿÿÿÿÿd¢
A__inference_dense_layer_call_and_return_conditional_losses_192648]HI0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ°	
ª "%¢"

0ÿÿÿÿÿÿÿÿÿx
 z
&__inference_dense_layer_call_fn_192638PHI0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ°	
ª "ÿÿÿÿÿÿÿÿÿx¨
C__inference_flatten_layer_call_and_return_conditional_losses_192629a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ°	
 
(__inference_flatten_layer_call_fn_192623T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0
ª "ÿÿÿÿÿÿÿÿÿ°	î
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_192618R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_1_layer_call_fn_192613R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_192541R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_layer_call_fn_192536R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
F__inference_sequential_layer_call_and_return_conditional_losses_191835 -.5HIPWX_fgnuE¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ë
F__inference_sequential_layer_call_and_return_conditional_losses_191885 -.5HIPWX_fgnuE¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ä
F__inference_sequential_layer_call_and_return_conditional_losses_192204z -.5HIPWX_fgnu?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ä
F__inference_sequential_layer_call_and_return_conditional_losses_192425z -.5HIPWX_fgnu?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¢
+__inference_sequential_layer_call_fn_191520s -.5HIPWX_fgnuE¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¢
+__inference_sequential_layer_call_fn_191785s -.5HIPWX_fgnuE¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

+__inference_sequential_layer_call_fn_191946m -.5HIPWX_fgnu?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

+__inference_sequential_layer_call_fn_191983m -.5HIPWX_fgnu?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Í
$__inference_signature_wrapper_192464¤ -.5HIPWX_fgnuM¢J
¢ 
Cª@
>
conv2d_input.+
conv2d_inputÿÿÿÿÿÿÿÿÿ"Aª>
<
softmax__decode)&
softmax__decodeÿÿÿÿÿÿÿÿÿ
ª
K__inference_softmax__decode_layer_call_and_return_conditional_losses_192849[u/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
0__inference_softmax__decode_layer_call_fn_192837Nu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ
»
L__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_192608k57¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

0
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

0
 
1__inference_spiking_b_relu_1_layer_call_fn_192569^57¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

0
ª " ÿÿÿÿÿÿÿÿÿ

0«
L__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_192696[P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿx
ª "%¢"

0ÿÿÿÿÿÿÿÿÿx
 
1__inference_spiking_b_relu_2_layer_call_fn_192657NP/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿx
ª "ÿÿÿÿÿÿÿÿÿx«
L__inference_spiking_b_relu_3_layer_call_and_return_conditional_losses_192763[_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "%¢"

0ÿÿÿÿÿÿÿÿÿT
 
1__inference_spiking_b_relu_3_layer_call_fn_192724N_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "ÿÿÿÿÿÿÿÿÿT«
L__inference_spiking_b_relu_4_layer_call_and_return_conditional_losses_192830[n/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
1__inference_spiking_b_relu_4_layer_call_fn_192791Nn/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿd¹
J__inference_spiking_b_relu_layer_call_and_return_conditional_losses_192531k 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
/__inference_spiking_b_relu_layer_call_fn_192492^ 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ 