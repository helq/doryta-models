��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
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
delete_old_dirsbool(�
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
2	�
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
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02unknown8��

v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
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
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
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
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@d*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@d*
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
p

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*
shared_name
Variable_3
i
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
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
�
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
�
 Adadelta/dense/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*1
shared_name" Adadelta/dense/kernel/accum_grad
�
4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense/kernel/accum_grad* 
_output_shapes
:
��*
dtype0
�
Adadelta/dense/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name Adadelta/dense/bias/accum_grad
�
2Adadelta/dense/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_grad*
_output_shapes	
:�*
dtype0
�
Adadelta/Variable/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdadelta/Variable/accum_grad
�
0Adadelta/Variable/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad*
_output_shapes
: *
dtype0
�
"Adadelta/dense_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*3
shared_name$"Adadelta/dense_1/kernel/accum_grad
�
6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_1/kernel/accum_grad*
_output_shapes
:	�@*
dtype0
�
 Adadelta/dense_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adadelta/dense_1/bias/accum_grad
�
4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_1/bias/accum_grad*
_output_shapes
:@*
dtype0
�
Adadelta/Variable/accum_grad_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_1
�
2Adadelta/Variable/accum_grad_1/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_1*
_output_shapes
: *
dtype0
�
"Adadelta/dense_2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@d*3
shared_name$"Adadelta/dense_2/kernel/accum_grad
�
6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_2/kernel/accum_grad*
_output_shapes

:@d*
dtype0
�
 Adadelta/dense_2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*1
shared_name" Adadelta/dense_2/bias/accum_grad
�
4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_2/bias/accum_grad*
_output_shapes
:d*
dtype0
�
Adadelta/Variable/accum_grad_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adadelta/Variable/accum_grad_2
�
2Adadelta/Variable/accum_grad_2/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_grad_2*
_output_shapes
: *
dtype0
�
Adadelta/dense/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!Adadelta/dense/kernel/accum_var
�
3Adadelta/dense/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/kernel/accum_var* 
_output_shapes
:
��*
dtype0
�
Adadelta/dense/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nameAdadelta/dense/bias/accum_var
�
1Adadelta/dense/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_var*
_output_shapes	
:�*
dtype0
�
Adadelta/Variable/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdadelta/Variable/accum_var
�
/Adadelta/Variable/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var*
_output_shapes
: *
dtype0
�
!Adadelta/dense_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*2
shared_name#!Adadelta/dense_1/kernel/accum_var
�
5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_1/kernel/accum_var*
_output_shapes
:	�@*
dtype0
�
Adadelta/dense_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adadelta/dense_1/bias/accum_var
�
3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_1/bias/accum_var*
_output_shapes
:@*
dtype0
�
Adadelta/Variable/accum_var_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_1
�
1Adadelta/Variable/accum_var_1/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_1*
_output_shapes
: *
dtype0
�
!Adadelta/dense_2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@d*2
shared_name#!Adadelta/dense_2/kernel/accum_var
�
5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_2/kernel/accum_var*
_output_shapes

:@d*
dtype0
�
Adadelta/dense_2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adadelta/dense_2/bias/accum_var
�
3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_2/bias/accum_var*
_output_shapes
:d*
dtype0
�
Adadelta/Variable/accum_var_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdadelta/Variable/accum_var_2
�
1Adadelta/Variable/accum_var_2/Read/ReadVariableOpReadVariableOpAdadelta/Variable/accum_var_2*
_output_shapes
: *
dtype0

NoOpNoOp
�K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�K
value�KB�J B�J
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
	sharpness
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
�
(	sharpness
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
�

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
�
7	sharpness
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
�
>_rescaled_key
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
�
Eiter
	Fdecay
Glearning_rate
Hrho
accum_grad}
accum_grad~
accum_grad 
accum_grad�!
accum_grad�(
accum_grad�/
accum_grad�0
accum_grad�7
accum_grad�	accum_var�	accum_var�	accum_var� 	accum_var�!	accum_var�(	accum_var�/	accum_var�0	accum_var�7	accum_var�*
J
0
1
2
 3
!4
(5
/6
07
78
>9*
C
0
1
2
 3
!4
(5
/6
07
78*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Nserving_default* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
[U
VARIABLE_VALUEVariable9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUE
Variable_19layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUE*

(0*

(0*
* 
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUE
Variable_29layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUE*

70*

70*
* 
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUE
Variable_3=layer_with_weights-6/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUE*

>0*
* 
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
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

>0*
5
0
1
2
3
4
5
6*

r0
s1*
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

>0*
* 
* 
* 
* 
8
	ttotal
	ucount
v	variables
w	keras_api*
H
	xtotal
	ycount
z
_fn_kwargs
{	variables
|	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

t0
u1*

v	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

x0
y1*

{	variables*
��
VARIABLE_VALUE Adadelta/dense/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/dense/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/Variable/accum_grad^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adadelta/dense_1/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adadelta/dense_1/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/Variable/accum_grad_1^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adadelta/dense_2/kernel/accum_grad[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adadelta/dense_2/bias/accum_gradYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/Variable/accum_grad_2^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/dense/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/dense/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/Variable/accum_var]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adadelta/dense_1/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/dense_1/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/Variable/accum_var_1]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adadelta/dense_2/kernel/accum_varZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/dense_2/bias/accum_varXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdadelta/Variable/accum_var_2]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_dense_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense/kernel
dense/biasVariabledense_1/kerneldense_1/bias
Variable_1dense_2/kerneldense_2/bias
Variable_2
Variable_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_51304
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpVariable/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpVariable_1/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOp2Adadelta/dense/bias/accum_grad/Read/ReadVariableOp0Adadelta/Variable/accum_grad/Read/ReadVariableOp6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOp2Adadelta/Variable/accum_grad_1/Read/ReadVariableOp6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOp2Adadelta/Variable/accum_grad_2/Read/ReadVariableOp3Adadelta/dense/kernel/accum_var/Read/ReadVariableOp1Adadelta/dense/bias/accum_var/Read/ReadVariableOp/Adadelta/Variable/accum_var/Read/ReadVariableOp5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOp1Adadelta/Variable/accum_var_1/Read/ReadVariableOp5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOp1Adadelta/Variable/accum_var_2/Read/ReadVariableOpConst*1
Tin*
(2&	*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_51655
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasVariabledense_1/kerneldense_1/bias
Variable_1dense_2/kerneldense_2/bias
Variable_2
Variable_3Adadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcounttotal_1count_1 Adadelta/dense/kernel/accum_gradAdadelta/dense/bias/accum_gradAdadelta/Variable/accum_grad"Adadelta/dense_1/kernel/accum_grad Adadelta/dense_1/bias/accum_gradAdadelta/Variable/accum_grad_1"Adadelta/dense_2/kernel/accum_grad Adadelta/dense_2/bias/accum_gradAdadelta/Variable/accum_grad_2Adadelta/dense/kernel/accum_varAdadelta/dense/bias/accum_varAdadelta/Variable/accum_var!Adadelta/dense_1/kernel/accum_varAdadelta/dense_1/bias/accum_varAdadelta/Variable/accum_var_1!Adadelta/dense_2/kernel/accum_varAdadelta/dense_2/bias/accum_varAdadelta/Variable/accum_var_2*0
Tin)
'2%*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_51773��	
�
\
cond_false_50655
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:���������d"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�

�
*__inference_sequential_layer_call_fn_50879
dense_input
unknown:
��
	unknown_0:	�
	unknown_1: 
	unknown_2:	�@
	unknown_3:@
	unknown_4: 
	unknown_5:@d
	unknown_6:d
	unknown_7: 
	unknown_8:d

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_50831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�

�
*__inference_sequential_layer_call_fn_51007

inputs
unknown:
��
	unknown_0:	�
	unknown_1: 
	unknown_2:	�@
	unknown_3:@
	unknown_4: 
	unknown_5:@d
	unknown_6:d
	unknown_7: 
	unknown_8:d

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_50831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dense_2_layer_call_fn_51447

inputs
unknown:@d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_50622o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
E__inference_sequential_layer_call_and_return_conditional_losses_50831

inputs
dense_50803:
��
dense_50805:	�
spiking_b_relu_50808:  
dense_1_50811:	�@
dense_1_50813:@ 
spiking_b_relu_1_50816: 
dense_2_50819:@d
dense_2_50821:d 
spiking_b_relu_2_50824: '
softmax__decode_50827:d

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50803dense_50805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_50504�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_50808*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_50549�
dense_1/StatefulPartitionedCallStatefulPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0dense_1_50811dense_1_50813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_50563�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_1_50816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_50608�
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0dense_2_50819dense_2_50821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_50622�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_2_50824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_50667�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_50827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_50683
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
Ď
�

 __inference__wrapped_model_50487
dense_inputC
/sequential_dense_matmul_readvariableop_resource:
��?
0sequential_dense_biasadd_readvariableop_resource:	�;
1sequential_spiking_b_relu_readvariableop_resource: D
1sequential_dense_1_matmul_readvariableop_resource:	�@@
2sequential_dense_1_biasadd_readvariableop_resource:@=
3sequential_spiking_b_relu_1_readvariableop_resource: C
1sequential_dense_2_matmul_readvariableop_resource:@d@
2sequential_dense_2_biasadd_readvariableop_resource:d=
3sequential_spiking_b_relu_2_readvariableop_resource: K
9sequential_softmax__decode_matmul_readvariableop_resource:d

identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�)sequential/dense_2/BiasAdd/ReadVariableOp�(sequential/dense_2/MatMul/ReadVariableOp�0sequential/softmax__decode/MatMul/ReadVariableOp�.sequential/spiking_b_relu/Equal/ReadVariableOp�(sequential/spiking_b_relu/ReadVariableOp�0sequential/spiking_b_relu_1/Equal/ReadVariableOp�*sequential/spiking_b_relu_1/ReadVariableOp�0sequential/spiking_b_relu_2/Equal/ReadVariableOp�*sequential/spiking_b_relu_2/ReadVariableOp�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
(sequential/spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
&sequential/spiking_b_relu/GreaterEqualGreaterEqual!sequential/dense/BiasAdd:output:01sequential/spiking_b_relu/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
sequential/spiking_b_relu/CastCast*sequential/spiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
(sequential/spiking_b_relu/ReadVariableOpReadVariableOp1sequential_spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0d
sequential/spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/spiking_b_relu/subSub(sequential/spiking_b_relu/sub/x:output:00sequential/spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: d
sequential/spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
sequential/spiking_b_relu/addAddV2!sequential/spiking_b_relu/sub:z:0(sequential/spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
!sequential/spiking_b_relu/truedivRealDiv,sequential/spiking_b_relu/truediv/x:output:0!sequential/spiking_b_relu/add:z:0*
T0*
_output_shapes
: f
!sequential/spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
sequential/spiking_b_relu/sub_1Sub!sequential/dense/BiasAdd:output:0*sequential/spiking_b_relu/sub_1/y:output:0*
T0*(
_output_shapes
:�����������
sequential/spiking_b_relu/mulMul%sequential/spiking_b_relu/truediv:z:0#sequential/spiking_b_relu/sub_1:z:0*
T0*(
_output_shapes
:����������f
!sequential/spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
sequential/spiking_b_relu/add_1AddV2!sequential/spiking_b_relu/mul:z:0*sequential/spiking_b_relu/add_1/y:output:0*
T0*(
_output_shapes
:����������v
1sequential/spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
/sequential/spiking_b_relu/clip_by_value/MinimumMinimum#sequential/spiking_b_relu/add_1:z:0:sequential/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:����������n
)sequential/spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential/spiking_b_relu/clip_by_valueMaximum3sequential/spiking_b_relu/clip_by_value/Minimum:z:02sequential/spiking_b_relu/clip_by_value/y:output:0*
T0*(
_output_shapes
:�����������
.sequential/spiking_b_relu/Equal/ReadVariableOpReadVariableOp1sequential_spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/spiking_b_relu/EqualEqual6sequential/spiking_b_relu/Equal/ReadVariableOp:value:0*sequential/spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: �
sequential/spiking_b_relu/condStatelessIf#sequential/spiking_b_relu/Equal:z:0"sequential/spiking_b_relu/Cast:y:0+sequential/spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *=
else_branch.R,
*sequential_spiking_b_relu_cond_false_50385*'
output_shapes
:����������*<
then_branch-R+
)sequential_spiking_b_relu_cond_true_50384�
'sequential/spiking_b_relu/cond/IdentityIdentity'sequential/spiking_b_relu/cond:output:0*
T0*(
_output_shapes
:�����������
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential/dense_1/MatMulMatMul0sequential/spiking_b_relu/cond/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@o
*sequential/spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
(sequential/spiking_b_relu_1/GreaterEqualGreaterEqual#sequential/dense_1/BiasAdd:output:03sequential/spiking_b_relu_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
 sequential/spiking_b_relu_1/CastCast,sequential/spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
*sequential/spiking_b_relu_1/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/spiking_b_relu_1/subSub*sequential/spiking_b_relu_1/sub/x:output:02sequential/spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!sequential/spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
sequential/spiking_b_relu_1/addAddV2#sequential/spiking_b_relu_1/sub:z:0*sequential/spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: j
%sequential/spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sequential/spiking_b_relu_1/truedivRealDiv.sequential/spiking_b_relu_1/truediv/x:output:0#sequential/spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
!sequential/spiking_b_relu_1/sub_1Sub#sequential/dense_1/BiasAdd:output:0,sequential/spiking_b_relu_1/sub_1/y:output:0*
T0*'
_output_shapes
:���������@�
sequential/spiking_b_relu_1/mulMul'sequential/spiking_b_relu_1/truediv:z:0%sequential/spiking_b_relu_1/sub_1:z:0*
T0*'
_output_shapes
:���������@h
#sequential/spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
!sequential/spiking_b_relu_1/add_1AddV2#sequential/spiking_b_relu_1/mul:z:0,sequential/spiking_b_relu_1/add_1/y:output:0*
T0*'
_output_shapes
:���������@x
3sequential/spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
1sequential/spiking_b_relu_1/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_1/add_1:z:0<sequential/spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@p
+sequential/spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)sequential/spiking_b_relu_1/clip_by_valueMaximum5sequential/spiking_b_relu_1/clip_by_value/Minimum:z:04sequential/spiking_b_relu_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
0sequential/spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
!sequential/spiking_b_relu_1/EqualEqual8sequential/spiking_b_relu_1/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: �
 sequential/spiking_b_relu_1/condStatelessIf%sequential/spiking_b_relu_1/Equal:z:0$sequential/spiking_b_relu_1/Cast:y:0-sequential/spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *?
else_branch0R.
,sequential_spiking_b_relu_1_cond_false_50426*&
output_shapes
:���������@*>
then_branch/R-
+sequential_spiking_b_relu_1_cond_true_50425�
)sequential/spiking_b_relu_1/cond/IdentityIdentity)sequential/spiking_b_relu_1/cond:output:0*
T0*'
_output_shapes
:���������@�
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:@d*
dtype0�
sequential/dense_2/MatMulMatMul2sequential/spiking_b_relu_1/cond/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������do
*sequential/spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
(sequential/spiking_b_relu_2/GreaterEqualGreaterEqual#sequential/dense_2/BiasAdd:output:03sequential/spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d�
 sequential/spiking_b_relu_2/CastCast,sequential/spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d�
*sequential/spiking_b_relu_2/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0f
!sequential/spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/spiking_b_relu_2/subSub*sequential/spiking_b_relu_2/sub/x:output:02sequential/spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!sequential/spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
sequential/spiking_b_relu_2/addAddV2#sequential/spiking_b_relu_2/sub:z:0*sequential/spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: j
%sequential/spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sequential/spiking_b_relu_2/truedivRealDiv.sequential/spiking_b_relu_2/truediv/x:output:0#sequential/spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: h
#sequential/spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
!sequential/spiking_b_relu_2/sub_1Sub#sequential/dense_2/BiasAdd:output:0,sequential/spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:���������d�
sequential/spiking_b_relu_2/mulMul'sequential/spiking_b_relu_2/truediv:z:0%sequential/spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:���������dh
#sequential/spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
!sequential/spiking_b_relu_2/add_1AddV2#sequential/spiking_b_relu_2/mul:z:0,sequential/spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:���������dx
3sequential/spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
1sequential/spiking_b_relu_2/clip_by_value/MinimumMinimum%sequential/spiking_b_relu_2/add_1:z:0<sequential/spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������dp
+sequential/spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)sequential/spiking_b_relu_2/clip_by_valueMaximum5sequential/spiking_b_relu_2/clip_by_value/Minimum:z:04sequential/spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d�
0sequential/spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp3sequential_spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0h
#sequential/spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
!sequential/spiking_b_relu_2/EqualEqual8sequential/spiking_b_relu_2/Equal/ReadVariableOp:value:0,sequential/spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: �
 sequential/spiking_b_relu_2/condStatelessIf%sequential/spiking_b_relu_2/Equal:z:0$sequential/spiking_b_relu_2/Cast:y:0-sequential/spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *?
else_branch0R.
,sequential_spiking_b_relu_2_cond_false_50467*&
output_shapes
:���������d*>
then_branch/R-
+sequential_spiking_b_relu_2_cond_true_50466�
)sequential/spiking_b_relu_2/cond/IdentityIdentity)sequential/spiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:���������de
 sequential/softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
sequential/softmax__decode/mulMul)sequential/softmax__decode/mul/x:output:02sequential/spiking_b_relu_2/cond/Identity:output:0*
T0*'
_output_shapes
:���������de
 sequential/softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential/softmax__decode/subSub"sequential/softmax__decode/mul:z:0)sequential/softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:���������d�
0sequential/softmax__decode/MatMul/ReadVariableOpReadVariableOp9sequential_softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0�
!sequential/softmax__decode/MatMulMatMul"sequential/softmax__decode/sub:z:08sequential/softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"sequential/softmax__decode/SoftmaxSoftmax+sequential/softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:���������
{
IdentityIdentity,sequential/softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp1^sequential/softmax__decode/MatMul/ReadVariableOp/^sequential/spiking_b_relu/Equal/ReadVariableOp)^sequential/spiking_b_relu/ReadVariableOp1^sequential/spiking_b_relu_1/Equal/ReadVariableOp+^sequential/spiking_b_relu_1/ReadVariableOp1^sequential/spiking_b_relu_2/Equal/ReadVariableOp+^sequential/spiking_b_relu_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2R
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
*sequential/spiking_b_relu_2/ReadVariableOp*sequential/spiking_b_relu_2/ReadVariableOp:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
�
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_50549

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*(
_output_shapes
:����������`
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
sub_1Subinputssub_1/y:output:0*
T0*(
_output_shapes
:����������U
mulMultruediv:z:0	sub_1:z:0*
T0*(
_output_shapes
:����������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?\
add_1AddV2mul:z:0add_1/y:output:0*
T0*(
_output_shapes
:����������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:����������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:����������d
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: �
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_50537*'
output_shapes
:����������*"
then_branchR
cond_true_50536[
cond/IdentityIdentitycond:output:0*
T0*(
_output_shapes
:����������f
IdentityIdentitycond/Identity:output:0^NoOp*
T0*(
_output_shapes
:����������n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
R
cond_true_50595
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:���������@"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�
�
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_51438

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
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
:���������@T
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@d
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: �
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_51426*&
output_shapes
:���������@*"
then_branchR
cond_true_51425Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:���������@e
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������@n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������@: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,sequential_spiking_b_relu_2_cond_false_504670
,sequential_spiking_b_relu_2_cond_placeholderW
Ssequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_clip_by_value-
)sequential_spiking_b_relu_2_cond_identity�
)sequential/spiking_b_relu_2/cond/IdentityIdentitySsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:���������d"_
)sequential_spiking_b_relu_2_cond_identity2sequential/spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�

�
*__inference_sequential_layer_call_fn_50982

inputs
unknown:
��
	unknown_0:	�
	unknown_1: 
	unknown_2:	�@
	unknown_3:@
	unknown_4: 
	unknown_5:@d
	unknown_6:d
	unknown_7: 
	unknown_8:d

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_50688o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
 spiking_b_relu_1_cond_true_510808
4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast%
!spiking_b_relu_1_cond_placeholder"
spiking_b_relu_1_cond_identity�
spiking_b_relu_1/cond/IdentityIdentity4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast*
T0*'
_output_shapes
:���������@"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�
�
!spiking_b_relu_1_cond_false_51216%
!spiking_b_relu_1_cond_placeholderA
=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value"
spiking_b_relu_1_cond_identity�
spiking_b_relu_1/cond/IdentityIdentity=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value*
T0*'
_output_shapes
:���������@"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�
�
!spiking_b_relu_1_cond_false_51081%
!spiking_b_relu_1_cond_placeholderA
=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value"
spiking_b_relu_1_cond_identity�
spiking_b_relu_1/cond/IdentityIdentity=spiking_b_relu_1_cond_identity_spiking_b_relu_1_clip_by_value*
T0*'
_output_shapes
:���������@"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�
�
spiking_b_relu_cond_true_511744
0spiking_b_relu_cond_identity_spiking_b_relu_cast#
spiking_b_relu_cond_placeholder 
spiking_b_relu_cond_identity�
spiking_b_relu/cond/IdentityIdentity0spiking_b_relu_cond_identity_spiking_b_relu_cast*
T0*(
_output_shapes
:����������"E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�
|
0__inference_spiking_b_relu_1_layer_call_fn_51399

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_50608o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������@: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
\
cond_false_51493
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:���������d"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�

�
*__inference_sequential_layer_call_fn_50711
dense_input
unknown:
��
	unknown_0:	�
	unknown_1: 
	unknown_2:	�@
	unknown_3:@
	unknown_4: 
	unknown_5:@d
	unknown_6:d
	unknown_7: 
	unknown_8:d

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_50688o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
R
cond_true_51358
cond_identity_cast
cond_placeholder
cond_identity`
cond/IdentityIdentitycond_identity_cast*
T0*(
_output_shapes
:����������"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�
z
.__inference_spiking_b_relu_layer_call_fn_51332

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_50549p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
|
0__inference_spiking_b_relu_2_layer_call_fn_51466

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_50667o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
/__inference_softmax__decode_layer_call_fn_51512

inputs
unknown:d

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_50683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
 spiking_b_relu_1_cond_true_512158
4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast%
!spiking_b_relu_1_cond_placeholder"
spiking_b_relu_1_cond_identity�
spiking_b_relu_1/cond/IdentityIdentity4spiking_b_relu_1_cond_identity_spiking_b_relu_1_cast*
T0*'
_output_shapes
:���������@"I
spiking_b_relu_1_cond_identity'spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�$
�
E__inference_sequential_layer_call_and_return_conditional_losses_50688

inputs
dense_50505:
��
dense_50507:	�
spiking_b_relu_50550:  
dense_1_50564:	�@
dense_1_50566:@ 
spiking_b_relu_1_50609: 
dense_2_50623:@d
dense_2_50625:d 
spiking_b_relu_2_50668: '
softmax__decode_50684:d

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50505dense_50507*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_50504�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_50550*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_50549�
dense_1/StatefulPartitionedCallStatefulPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0dense_1_50564dense_1_50566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_50563�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_1_50609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_50608�
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0dense_2_50623dense_2_50625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_50622�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_2_50668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_50667�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_50684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_50683
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_50622

inputs0
matmul_readvariableop_resource:@d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
R
cond_true_51492
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:���������d"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�
\
cond_false_51359
cond_placeholder
cond_identity_clip_by_value
cond_identityi
cond/IdentityIdentitycond_identity_clip_by_value*
T0*(
_output_shapes
:����������"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�
R
cond_true_51425
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:���������@"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�
\
cond_false_50596
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:���������@"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�
�
 spiking_b_relu_2_cond_true_512568
4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast%
!spiking_b_relu_2_cond_placeholder"
spiking_b_relu_2_cond_identity�
spiking_b_relu_2/cond/IdentityIdentity4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast*
T0*'
_output_shapes
:���������d"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�%
�
E__inference_sequential_layer_call_and_return_conditional_losses_50941
dense_input
dense_50913:
��
dense_50915:	�
spiking_b_relu_50918:  
dense_1_50921:	�@
dense_1_50923:@ 
spiking_b_relu_1_50926: 
dense_2_50929:@d
dense_2_50931:d 
spiking_b_relu_2_50934: '
softmax__decode_50937:d

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_50913dense_50915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_50504�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_50918*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_50549�
dense_1/StatefulPartitionedCallStatefulPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0dense_1_50921dense_1_50923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_50563�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_1_50926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_50608�
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0dense_2_50929dense_2_50931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_50622�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_2_50934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_50667�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_50937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_50683
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
�
*sequential_spiking_b_relu_cond_false_50385.
*sequential_spiking_b_relu_cond_placeholderS
Osequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_clip_by_value+
'sequential_spiking_b_relu_cond_identity�
'sequential/spiking_b_relu/cond/IdentityIdentityOsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_clip_by_value*
T0*(
_output_shapes
:����������"[
'sequential_spiking_b_relu_cond_identity0sequential/spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�	
�
J__inference_softmax__decode_layer_call_and_return_conditional_losses_51524

inputs0
matmul_readvariableop_resource:d

identity��MatMul/ReadVariableOpJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������dJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:���������dt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0j
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_50563

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
J__inference_softmax__decode_layer_call_and_return_conditional_losses_50683

inputs0
matmul_readvariableop_resource:d

identity��MatMul/ReadVariableOpJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
mulMulmul/x:output:0inputs*
T0*'
_output_shapes
:���������dJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:���������dt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0j
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
spiking_b_relu_cond_false_51175#
spiking_b_relu_cond_placeholder=
9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value 
spiking_b_relu_cond_identity�
spiking_b_relu/cond/IdentityIdentity9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value*
T0*(
_output_shapes
:����������"E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�
\
cond_false_50537
cond_placeholder
cond_identity_clip_by_value
cond_identityi
cond/IdentityIdentitycond_identity_clip_by_value*
T0*(
_output_shapes
:����������"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�
�
)sequential_spiking_b_relu_cond_true_50384J
Fsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_cast.
*sequential_spiking_b_relu_cond_placeholder+
'sequential_spiking_b_relu_cond_identity�
'sequential/spiking_b_relu/cond/IdentityIdentityFsequential_spiking_b_relu_cond_identity_sequential_spiking_b_relu_cast*
T0*(
_output_shapes
:����������"[
'sequential_spiking_b_relu_cond_identity0sequential/spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�	
�
#__inference_signature_wrapper_51304
dense_input
unknown:
��
	unknown_0:	�
	unknown_1: 
	unknown_2:	�@
	unknown_3:@
	unknown_4: 
	unknown_5:@d
	unknown_6:d
	unknown_7: 
	unknown_8:d

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_50487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
�
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_50608

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
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
:���������@T
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:���������@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@d
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: �
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_50596*&
output_shapes
:���������@*"
then_branchR
cond_true_50595Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:���������@e
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������@n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������@: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_51323

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�N
�
__inference__traced_save_51655
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop'
#savev2_variable_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop)
%savev2_variable_1_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop?
;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_dense_bias_accum_grad_read_readvariableop;
7savev2_adadelta_variable_accum_grad_read_readvariableopA
=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableop=
9savev2_adadelta_variable_accum_grad_1_read_readvariableopA
=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableop=
9savev2_adadelta_variable_accum_grad_2_read_readvariableop>
:savev2_adadelta_dense_kernel_accum_var_read_readvariableop<
8savev2_adadelta_dense_bias_accum_var_read_readvariableop:
6savev2_adadelta_variable_accum_var_read_readvariableop@
<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop<
8savev2_adadelta_variable_accum_var_1_read_readvariableop@
<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_2_bias_accum_var_read_readvariableop<
8savev2_adadelta_variable_accum_var_2_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-6/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop#savev2_variable_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop%savev2_variable_1_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop9savev2_adadelta_dense_bias_accum_grad_read_readvariableop7savev2_adadelta_variable_accum_grad_read_readvariableop=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_1_read_readvariableop=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableop9savev2_adadelta_variable_accum_grad_2_read_readvariableop:savev2_adadelta_dense_kernel_accum_var_read_readvariableop8savev2_adadelta_dense_bias_accum_var_read_readvariableop6savev2_adadelta_variable_accum_var_read_readvariableop<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_1_read_readvariableop<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_2_bias_accum_var_read_readvariableop8savev2_adadelta_variable_accum_var_2_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�: :	�@:@: :@d:d: :d
: : : : : : : : :
��:�: :	�@:@: :@d:d: :
��:�: :	�@:@: :@d:d: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:

_output_shapes
: :$ 

_output_shapes

:@d: 

_output_shapes
:d:	

_output_shapes
: :$
 

_output_shapes

:d
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:

_output_shapes
: :$ 

_output_shapes

:@d: 

_output_shapes
:d:

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :%!

_output_shapes
:	�@:  

_output_shapes
:@:!

_output_shapes
: :$" 

_output_shapes

:@d: #

_output_shapes
:d:$

_output_shapes
: :%

_output_shapes
: 
�
�
spiking_b_relu_cond_true_510394
0spiking_b_relu_cond_identity_spiking_b_relu_cast#
spiking_b_relu_cond_placeholder 
spiking_b_relu_cond_identity�
spiking_b_relu/cond/IdentityIdentity0spiking_b_relu_cond_identity_spiking_b_relu_cast*
T0*(
_output_shapes
:����������"E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�
R
cond_true_50654
cond_identity_cast
cond_placeholder
cond_identity_
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:���������d"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�%
�
E__inference_sequential_layer_call_and_return_conditional_losses_50910
dense_input
dense_50882:
��
dense_50884:	�
spiking_b_relu_50887:  
dense_1_50890:	�@
dense_1_50892:@ 
spiking_b_relu_1_50895: 
dense_2_50898:@d
dense_2_50900:d 
spiking_b_relu_2_50903: '
softmax__decode_50906:d

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_50882dense_50884*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_50504�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_50887*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_50549�
dense_1/StatefulPartitionedCallStatefulPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0dense_1_50890dense_1_50892*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_50563�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_1_50895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_50608�
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0dense_2_50898dense_2_50900*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_50622�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_2_50903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_50667�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_50906*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_softmax__decode_layer_call_and_return_conditional_losses_50683
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
�
+sequential_spiking_b_relu_1_cond_true_50425N
Jsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_cast0
,sequential_spiking_b_relu_1_cond_placeholder-
)sequential_spiking_b_relu_1_cond_identity�
)sequential/spiking_b_relu_1/cond/IdentityIdentityJsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_cast*
T0*'
_output_shapes
:���������@"_
)sequential_spiking_b_relu_1_cond_identity2sequential/spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�w
�
E__inference_sequential_layer_call_and_return_conditional_losses_51277

inputs8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�0
&spiking_b_relu_readvariableop_resource: 9
&dense_1_matmul_readvariableop_resource:	�@5
'dense_1_biasadd_readvariableop_resource:@2
(spiking_b_relu_1_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:@d5
'dense_2_biasadd_readvariableop_resource:d2
(spiking_b_relu_2_readvariableop_resource: @
.softmax__decode_matmul_readvariableop_resource:d

identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�%softmax__decode/MatMul/ReadVariableOp�#spiking_b_relu/Equal/ReadVariableOp�spiking_b_relu/ReadVariableOp�%spiking_b_relu_1/Equal/ReadVariableOp�spiking_b_relu_1/ReadVariableOp�%spiking_b_relu_2/Equal/ReadVariableOp�spiking_b_relu_2/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b
spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu/GreaterEqualGreaterEqualdense/BiasAdd:output:0&spiking_b_relu/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������~
spiking_b_relu/CastCastspiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������|
spiking_b_relu/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0Y
spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu/subSubspiking_b_relu/sub/x:output:0%spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: Y
spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:s
spiking_b_relu/addAddV2spiking_b_relu/sub:z:0spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: ]
spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
spiking_b_relu/truedivRealDiv!spiking_b_relu/truediv/x:output:0spiking_b_relu/add:z:0*
T0*
_output_shapes
: [
spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu/sub_1Subdense/BiasAdd:output:0spiking_b_relu/sub_1/y:output:0*
T0*(
_output_shapes
:�����������
spiking_b_relu/mulMulspiking_b_relu/truediv:z:0spiking_b_relu/sub_1:z:0*
T0*(
_output_shapes
:����������[
spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu/add_1AddV2spiking_b_relu/mul:z:0spiking_b_relu/add_1/y:output:0*
T0*(
_output_shapes
:����������k
&spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$spiking_b_relu/clip_by_value/MinimumMinimumspiking_b_relu/add_1:z:0/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:����������c
spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
spiking_b_relu/clip_by_valueMaximum(spiking_b_relu/clip_by_value/Minimum:z:0'spiking_b_relu/clip_by_value/y:output:0*
T0*(
_output_shapes
:�����������
#spiking_b_relu/Equal/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu/EqualEqual+spiking_b_relu/Equal/ReadVariableOp:value:0spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: �
spiking_b_relu/condStatelessIfspiking_b_relu/Equal:z:0spiking_b_relu/Cast:y:0 spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
else_branch#R!
spiking_b_relu_cond_false_51175*'
output_shapes
:����������*1
then_branch"R 
spiking_b_relu_cond_true_51174y
spiking_b_relu/cond/IdentityIdentityspiking_b_relu/cond:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_1/MatMulMatMul%spiking_b_relu/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_1/GreaterEqualGreaterEqualdense_1/BiasAdd:output:0(spiking_b_relu_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
spiking_b_relu_1/CastCast!spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
spiking_b_relu_1/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_1/subSubspiking_b_relu_1/sub/x:output:0'spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:y
spiking_b_relu_1/addAddV2spiking_b_relu_1/sub:z:0spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_1/truedivRealDiv#spiking_b_relu_1/truediv/x:output:0spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_1/sub_1Subdense_1/BiasAdd:output:0!spiking_b_relu_1/sub_1/y:output:0*
T0*'
_output_shapes
:���������@�
spiking_b_relu_1/mulMulspiking_b_relu_1/truediv:z:0spiking_b_relu_1/sub_1:z:0*
T0*'
_output_shapes
:���������@]
spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_1/add_1AddV2spiking_b_relu_1/mul:z:0!spiking_b_relu_1/add_1/y:output:0*
T0*'
_output_shapes
:���������@m
(spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&spiking_b_relu_1/clip_by_value/MinimumMinimumspiking_b_relu_1/add_1:z:01spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@e
 spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
spiking_b_relu_1/clip_by_valueMaximum*spiking_b_relu_1/clip_by_value/Minimum:z:0)spiking_b_relu_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
%spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_1/EqualEqual-spiking_b_relu_1/Equal/ReadVariableOp:value:0!spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: �
spiking_b_relu_1/condStatelessIfspiking_b_relu_1/Equal:z:0spiking_b_relu_1/Cast:y:0"spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_1_cond_false_51216*&
output_shapes
:���������@*3
then_branch$R"
 spiking_b_relu_1_cond_true_51215|
spiking_b_relu_1/cond/IdentityIdentityspiking_b_relu_1/cond:output:0*
T0*'
_output_shapes
:���������@�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@d*
dtype0�
dense_2/MatMulMatMul'spiking_b_relu_1/cond/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_2/GreaterEqualGreaterEqualdense_2/BiasAdd:output:0(spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d�
spiking_b_relu_2/CastCast!spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d�
spiking_b_relu_2/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_2/subSubspiking_b_relu_2/sub/x:output:0'spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:y
spiking_b_relu_2/addAddV2spiking_b_relu_2/sub:z:0spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_2/truedivRealDiv#spiking_b_relu_2/truediv/x:output:0spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_2/sub_1Subdense_2/BiasAdd:output:0!spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:���������d�
spiking_b_relu_2/mulMulspiking_b_relu_2/truediv:z:0spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:���������d]
spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_2/add_1AddV2spiking_b_relu_2/mul:z:0!spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:���������dm
(spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&spiking_b_relu_2/clip_by_value/MinimumMinimumspiking_b_relu_2/add_1:z:01spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������de
 spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
spiking_b_relu_2/clip_by_valueMaximum*spiking_b_relu_2/clip_by_value/Minimum:z:0)spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d�
%spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_2/EqualEqual-spiking_b_relu_2/Equal/ReadVariableOp:value:0!spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: �
spiking_b_relu_2/condStatelessIfspiking_b_relu_2/Equal:z:0spiking_b_relu_2/Cast:y:0"spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_2_cond_false_51257*&
output_shapes
:���������d*3
then_branch$R"
 spiking_b_relu_2_cond_true_51256|
spiking_b_relu_2/cond/IdentityIdentityspiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:���������dZ
softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
softmax__decode/mulMulsoftmax__decode/mul/x:output:0'spiking_b_relu_2/cond/Identity:output:0*
T0*'
_output_shapes
:���������dZ
softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
softmax__decode/subSubsoftmax__decode/mul:z:0softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:���������d�
%softmax__decode/MatMul/ReadVariableOpReadVariableOp.softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0�
softmax__decode/MatMulMatMulsoftmax__decode/sub:z:0-softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
v
softmax__decode/SoftmaxSoftmax softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:���������
p
IdentityIdentity!softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^softmax__decode/MatMul/ReadVariableOp$^spiking_b_relu/Equal/ReadVariableOp^spiking_b_relu/ReadVariableOp&^spiking_b_relu_1/Equal/ReadVariableOp ^spiking_b_relu_1/ReadVariableOp&^spiking_b_relu_2/Equal/ReadVariableOp ^spiking_b_relu_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2<
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
spiking_b_relu_2/ReadVariableOpspiking_b_relu_2/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_51505

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
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
:���������dT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������dL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:���������d\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������dd
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: �
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_51493*&
output_shapes
:���������d*"
then_branchR
cond_true_51492Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:���������de
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������dn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
ۓ
�
!__inference__traced_restore_51773
file_prefix1
assignvariableop_dense_kernel:
��,
assignvariableop_1_dense_bias:	�%
assignvariableop_2_variable: 4
!assignvariableop_3_dense_1_kernel:	�@-
assignvariableop_4_dense_1_bias:@'
assignvariableop_5_variable_1: 3
!assignvariableop_6_dense_2_kernel:@d-
assignvariableop_7_dense_2_bias:d'
assignvariableop_8_variable_2: /
assignvariableop_9_variable_3:d
+
!assignvariableop_10_adadelta_iter:	 ,
"assignvariableop_11_adadelta_decay: 4
*assignvariableop_12_adadelta_learning_rate: *
 assignvariableop_13_adadelta_rho: #
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: H
4assignvariableop_18_adadelta_dense_kernel_accum_grad:
��A
2assignvariableop_19_adadelta_dense_bias_accum_grad:	�:
0assignvariableop_20_adadelta_variable_accum_grad: I
6assignvariableop_21_adadelta_dense_1_kernel_accum_grad:	�@B
4assignvariableop_22_adadelta_dense_1_bias_accum_grad:@<
2assignvariableop_23_adadelta_variable_accum_grad_1: H
6assignvariableop_24_adadelta_dense_2_kernel_accum_grad:@dB
4assignvariableop_25_adadelta_dense_2_bias_accum_grad:d<
2assignvariableop_26_adadelta_variable_accum_grad_2: G
3assignvariableop_27_adadelta_dense_kernel_accum_var:
��@
1assignvariableop_28_adadelta_dense_bias_accum_var:	�9
/assignvariableop_29_adadelta_variable_accum_var: H
5assignvariableop_30_adadelta_dense_1_kernel_accum_var:	�@A
3assignvariableop_31_adadelta_dense_1_bias_accum_var:@;
1assignvariableop_32_adadelta_variable_accum_var_1: G
5assignvariableop_33_adadelta_dense_2_kernel_accum_var:@dA
3assignvariableop_34_adadelta_dense_2_bias_accum_var:d;
1assignvariableop_35_adadelta_variable_accum_var_2: 
identity_37��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-6/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-1/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-3/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-5/sharpness/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variableIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_3Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_adadelta_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_adadelta_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adadelta_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_adadelta_rhoIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adadelta_dense_kernel_accum_gradIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adadelta_dense_bias_accum_gradIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adadelta_variable_accum_gradIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adadelta_dense_1_kernel_accum_gradIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adadelta_dense_1_bias_accum_gradIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adadelta_variable_accum_grad_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adadelta_dense_2_kernel_accum_gradIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adadelta_dense_2_bias_accum_gradIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adadelta_variable_accum_grad_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adadelta_dense_kernel_accum_varIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adadelta_dense_bias_accum_varIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adadelta_variable_accum_varIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adadelta_dense_1_kernel_accum_varIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adadelta_dense_1_bias_accum_varIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adadelta_variable_accum_var_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp5assignvariableop_33_adadelta_dense_2_kernel_accum_varIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adadelta_dense_2_bias_accum_varIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adadelta_variable_accum_var_2Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
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
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_51457

inputs0
matmul_readvariableop_resource:@d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_51390

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_50504

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_51380

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_50563o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,sequential_spiking_b_relu_1_cond_false_504260
,sequential_spiking_b_relu_1_cond_placeholderW
Ssequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_clip_by_value-
)sequential_spiking_b_relu_1_cond_identity�
)sequential/spiking_b_relu_1/cond/IdentityIdentitySsequential_spiking_b_relu_1_cond_identity_sequential_spiking_b_relu_1_clip_by_value*
T0*'
_output_shapes
:���������@"_
)sequential_spiking_b_relu_1_cond_identity2sequential/spiking_b_relu_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�
�
 spiking_b_relu_2_cond_true_511218
4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast%
!spiking_b_relu_2_cond_placeholder"
spiking_b_relu_2_cond_identity�
spiking_b_relu_2/cond/IdentityIdentity4spiking_b_relu_2_cond_identity_spiking_b_relu_2_cast*
T0*'
_output_shapes
:���������d"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�w
�
E__inference_sequential_layer_call_and_return_conditional_losses_51142

inputs8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�0
&spiking_b_relu_readvariableop_resource: 9
&dense_1_matmul_readvariableop_resource:	�@5
'dense_1_biasadd_readvariableop_resource:@2
(spiking_b_relu_1_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:@d5
'dense_2_biasadd_readvariableop_resource:d2
(spiking_b_relu_2_readvariableop_resource: @
.softmax__decode_matmul_readvariableop_resource:d

identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�%softmax__decode/MatMul/ReadVariableOp�#spiking_b_relu/Equal/ReadVariableOp�spiking_b_relu/ReadVariableOp�%spiking_b_relu_1/Equal/ReadVariableOp�spiking_b_relu_1/ReadVariableOp�%spiking_b_relu_2/Equal/ReadVariableOp�spiking_b_relu_2/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b
spiking_b_relu/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu/GreaterEqualGreaterEqualdense/BiasAdd:output:0&spiking_b_relu/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������~
spiking_b_relu/CastCastspiking_b_relu/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������|
spiking_b_relu/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0Y
spiking_b_relu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu/subSubspiking_b_relu/sub/x:output:0%spiking_b_relu/ReadVariableOp:value:0*
T0*
_output_shapes
: Y
spiking_b_relu/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:s
spiking_b_relu/addAddV2spiking_b_relu/sub:z:0spiking_b_relu/add/y:output:0*
T0*
_output_shapes
: ]
spiking_b_relu/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
spiking_b_relu/truedivRealDiv!spiking_b_relu/truediv/x:output:0spiking_b_relu/add:z:0*
T0*
_output_shapes
: [
spiking_b_relu/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu/sub_1Subdense/BiasAdd:output:0spiking_b_relu/sub_1/y:output:0*
T0*(
_output_shapes
:�����������
spiking_b_relu/mulMulspiking_b_relu/truediv:z:0spiking_b_relu/sub_1:z:0*
T0*(
_output_shapes
:����������[
spiking_b_relu/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu/add_1AddV2spiking_b_relu/mul:z:0spiking_b_relu/add_1/y:output:0*
T0*(
_output_shapes
:����������k
&spiking_b_relu/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$spiking_b_relu/clip_by_value/MinimumMinimumspiking_b_relu/add_1:z:0/spiking_b_relu/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:����������c
spiking_b_relu/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
spiking_b_relu/clip_by_valueMaximum(spiking_b_relu/clip_by_value/Minimum:z:0'spiking_b_relu/clip_by_value/y:output:0*
T0*(
_output_shapes
:�����������
#spiking_b_relu/Equal/ReadVariableOpReadVariableOp&spiking_b_relu_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu/EqualEqual+spiking_b_relu/Equal/ReadVariableOp:value:0spiking_b_relu/Equal/y:output:0*
T0*
_output_shapes
: �
spiking_b_relu/condStatelessIfspiking_b_relu/Equal:z:0spiking_b_relu/Cast:y:0 spiking_b_relu/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
else_branch#R!
spiking_b_relu_cond_false_51040*'
output_shapes
:����������*1
then_branch"R 
spiking_b_relu_cond_true_51039y
spiking_b_relu/cond/IdentityIdentityspiking_b_relu/cond:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_1/MatMulMatMul%spiking_b_relu/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
spiking_b_relu_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_1/GreaterEqualGreaterEqualdense_1/BiasAdd:output:0(spiking_b_relu_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
spiking_b_relu_1/CastCast!spiking_b_relu_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
spiking_b_relu_1/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_1/subSubspiking_b_relu_1/sub/x:output:0'spiking_b_relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:y
spiking_b_relu_1/addAddV2spiking_b_relu_1/sub:z:0spiking_b_relu_1/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_1/truedivRealDiv#spiking_b_relu_1/truediv/x:output:0spiking_b_relu_1/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_1/sub_1Subdense_1/BiasAdd:output:0!spiking_b_relu_1/sub_1/y:output:0*
T0*'
_output_shapes
:���������@�
spiking_b_relu_1/mulMulspiking_b_relu_1/truediv:z:0spiking_b_relu_1/sub_1:z:0*
T0*'
_output_shapes
:���������@]
spiking_b_relu_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_1/add_1AddV2spiking_b_relu_1/mul:z:0!spiking_b_relu_1/add_1/y:output:0*
T0*'
_output_shapes
:���������@m
(spiking_b_relu_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&spiking_b_relu_1/clip_by_value/MinimumMinimumspiking_b_relu_1/add_1:z:01spiking_b_relu_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@e
 spiking_b_relu_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
spiking_b_relu_1/clip_by_valueMaximum*spiking_b_relu_1/clip_by_value/Minimum:z:0)spiking_b_relu_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@�
%spiking_b_relu_1/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_1_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_1/EqualEqual-spiking_b_relu_1/Equal/ReadVariableOp:value:0!spiking_b_relu_1/Equal/y:output:0*
T0*
_output_shapes
: �
spiking_b_relu_1/condStatelessIfspiking_b_relu_1/Equal:z:0spiking_b_relu_1/Cast:y:0"spiking_b_relu_1/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_1_cond_false_51081*&
output_shapes
:���������@*3
then_branch$R"
 spiking_b_relu_1_cond_true_51080|
spiking_b_relu_1/cond/IdentityIdentityspiking_b_relu_1/cond:output:0*
T0*'
_output_shapes
:���������@�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@d*
dtype0�
dense_2/MatMulMatMul'spiking_b_relu_1/cond/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
spiking_b_relu_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_2/GreaterEqualGreaterEqualdense_2/BiasAdd:output:0(spiking_b_relu_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d�
spiking_b_relu_2/CastCast!spiking_b_relu_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d�
spiking_b_relu_2/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0[
spiking_b_relu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_2/subSubspiking_b_relu_2/sub/x:output:0'spiking_b_relu_2/ReadVariableOp:value:0*
T0*
_output_shapes
: [
spiking_b_relu_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:y
spiking_b_relu_2/addAddV2spiking_b_relu_2/sub:z:0spiking_b_relu_2/add/y:output:0*
T0*
_output_shapes
: _
spiking_b_relu_2/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_2/truedivRealDiv#spiking_b_relu_2/truediv/x:output:0spiking_b_relu_2/add:z:0*
T0*
_output_shapes
: ]
spiking_b_relu_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_2/sub_1Subdense_2/BiasAdd:output:0!spiking_b_relu_2/sub_1/y:output:0*
T0*'
_output_shapes
:���������d�
spiking_b_relu_2/mulMulspiking_b_relu_2/truediv:z:0spiking_b_relu_2/sub_1:z:0*
T0*'
_output_shapes
:���������d]
spiking_b_relu_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
spiking_b_relu_2/add_1AddV2spiking_b_relu_2/mul:z:0!spiking_b_relu_2/add_1/y:output:0*
T0*'
_output_shapes
:���������dm
(spiking_b_relu_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&spiking_b_relu_2/clip_by_value/MinimumMinimumspiking_b_relu_2/add_1:z:01spiking_b_relu_2/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������de
 spiking_b_relu_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
spiking_b_relu_2/clip_by_valueMaximum*spiking_b_relu_2/clip_by_value/Minimum:z:0)spiking_b_relu_2/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d�
%spiking_b_relu_2/Equal/ReadVariableOpReadVariableOp(spiking_b_relu_2_readvariableop_resource*
_output_shapes
: *
dtype0]
spiking_b_relu_2/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
spiking_b_relu_2/EqualEqual-spiking_b_relu_2/Equal/ReadVariableOp:value:0!spiking_b_relu_2/Equal/y:output:0*
T0*
_output_shapes
: �
spiking_b_relu_2/condStatelessIfspiking_b_relu_2/Equal:z:0spiking_b_relu_2/Cast:y:0"spiking_b_relu_2/clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *4
else_branch%R#
!spiking_b_relu_2_cond_false_51122*&
output_shapes
:���������d*3
then_branch$R"
 spiking_b_relu_2_cond_true_51121|
spiking_b_relu_2/cond/IdentityIdentityspiking_b_relu_2/cond:output:0*
T0*'
_output_shapes
:���������dZ
softmax__decode/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
softmax__decode/mulMulsoftmax__decode/mul/x:output:0'spiking_b_relu_2/cond/Identity:output:0*
T0*'
_output_shapes
:���������dZ
softmax__decode/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
softmax__decode/subSubsoftmax__decode/mul:z:0softmax__decode/sub/y:output:0*
T0*'
_output_shapes
:���������d�
%softmax__decode/MatMul/ReadVariableOpReadVariableOp.softmax__decode_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0�
softmax__decode/MatMulMatMulsoftmax__decode/sub:z:0-softmax__decode/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
v
softmax__decode/SoftmaxSoftmax softmax__decode/MatMul:product:0*
T0*'
_output_shapes
:���������
p
IdentityIdentity!softmax__decode/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^softmax__decode/MatMul/ReadVariableOp$^spiking_b_relu/Equal/ReadVariableOp^spiking_b_relu/ReadVariableOp&^spiking_b_relu_1/Equal/ReadVariableOp ^spiking_b_relu_1/ReadVariableOp&^spiking_b_relu_2/Equal/ReadVariableOp ^spiking_b_relu_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2<
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
spiking_b_relu_2/ReadVariableOpspiking_b_relu_2/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+sequential_spiking_b_relu_2_cond_true_50466N
Jsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_cast0
,sequential_spiking_b_relu_2_cond_placeholder-
)sequential_spiking_b_relu_2_cond_identity�
)sequential/spiking_b_relu_2/cond/IdentityIdentityJsequential_spiking_b_relu_2_cond_identity_sequential_spiking_b_relu_2_cast*
T0*'
_output_shapes
:���������d"_
)sequential_spiking_b_relu_2_cond_identity2sequential/spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�
�
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_50667

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?o
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d_
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
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
:���������dT
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������dL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:���������d\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������dd
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: �
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_50655*&
output_shapes
:���������d*"
then_branchR
cond_true_50654Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:���������de
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������dn
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
spiking_b_relu_cond_false_51040#
spiking_b_relu_cond_placeholder=
9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value 
spiking_b_relu_cond_identity�
spiking_b_relu/cond/IdentityIdentity9spiking_b_relu_cond_identity_spiking_b_relu_clip_by_value*
T0*(
_output_shapes
:����������"E
spiking_b_relu_cond_identity%spiking_b_relu/cond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�
�
!spiking_b_relu_2_cond_false_51122%
!spiking_b_relu_2_cond_placeholderA
=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value"
spiking_b_relu_2_cond_identity�
spiking_b_relu_2/cond/IdentityIdentity=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:���������d"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�
\
cond_false_51426
cond_placeholder
cond_identity_clip_by_value
cond_identityh
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:���������@"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:- )
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@
�
R
cond_true_50536
cond_identity_cast
cond_placeholder
cond_identity`
cond/IdentityIdentitycond_identity_cast*
T0*(
_output_shapes
:����������"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:. *
(
_output_shapes
:����������:.*
(
_output_shapes
:����������
�
�
!spiking_b_relu_2_cond_false_51257%
!spiking_b_relu_2_cond_placeholderA
=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value"
spiking_b_relu_2_cond_identity�
spiking_b_relu_2/cond/IdentityIdentity=spiking_b_relu_2_cond_identity_spiking_b_relu_2_clip_by_value*
T0*'
_output_shapes
:���������d"I
spiking_b_relu_2_cond_identity'spiking_b_relu_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������d:���������d:- )
'
_output_shapes
:���������d:-)
'
_output_shapes
:���������d
�
�
%__inference_dense_layer_call_fn_51313

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_50504p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_51371

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpS
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*(
_output_shapes
:����������`
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?S
subSubsub/x:output:0ReadVariableOp:value:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:F
addAddV2sub:z:0add/y:output:0*
T0*
_output_shapes
: N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
truedivRealDivtruediv/x:output:0add:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
sub_1Subinputssub_1/y:output:0*
T0*(
_output_shapes
:����������U
mulMultruediv:z:0	sub_1:z:0*
T0*(
_output_shapes
:����������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?\
add_1AddV2mul:z:0add_1/y:output:0*
T0*(
_output_shapes
:����������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:����������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:����������d
Equal/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: �
condStatelessIf	Equal:z:0Cast:y:0clip_by_value:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:����������* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_51359*'
output_shapes
:����������*"
then_branchR
cond_true_51358[
cond/IdentityIdentitycond:output:0*
T0*(
_output_shapes
:����������f
IdentityIdentitycond/Identity:output:0^NoOp*
T0*(
_output_shapes
:����������n
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2,
Equal/ReadVariableOpEqual/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
dense_input5
serving_default_dense_input:0����������C
softmax__decode0
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	sharpness
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(	sharpness
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
�
7	sharpness
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
�
>_rescaled_key
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Eiter
	Fdecay
Glearning_rate
Hrho
accum_grad}
accum_grad~
accum_grad 
accum_grad�!
accum_grad�(
accum_grad�/
accum_grad�0
accum_grad�7
accum_grad�	accum_var�	accum_var�	accum_var� 	accum_var�!	accum_var�(	accum_var�/	accum_var�0	accum_var�7	accum_var�"
	optimizer
f
0
1
2
 3
!4
(5
/6
07
78
>9"
trackable_list_wrapper
_
0
1
2
 3
!4
(5
/6
07
78"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_sequential_layer_call_fn_50711
*__inference_sequential_layer_call_fn_50982
*__inference_sequential_layer_call_fn_51007
*__inference_sequential_layer_call_fn_50879�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_sequential_layer_call_and_return_conditional_losses_51142
E__inference_sequential_layer_call_and_return_conditional_losses_51277
E__inference_sequential_layer_call_and_return_conditional_losses_50910
E__inference_sequential_layer_call_and_return_conditional_losses_50941�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_50487dense_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Nserving_default"
signature_map
 :
��2dense/kernel
:�2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
%__inference_dense_layer_call_fn_51313�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_51323�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
: 2Variable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
.__inference_spiking_b_relu_layer_call_fn_51332�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_51371�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!:	�@2dense_1/kernel
:@2dense_1/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_dense_1_layer_call_fn_51380�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_1_layer_call_and_return_conditional_losses_51390�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
: 2Variable
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�2�
0__inference_spiking_b_relu_1_layer_call_fn_51399�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_51438�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 :@d2dense_2/kernel
:d2dense_2/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_dense_2_layer_call_fn_51447�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_2_layer_call_and_return_conditional_losses_51457�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
: 2Variable
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�2�
0__inference_spiking_b_relu_2_layer_call_fn_51466�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_51505�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:d
2Variable
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_softmax__decode_layer_call_fn_51512�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_softmax__decode_layer_call_and_return_conditional_losses_51524�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
'
>0"
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
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_signature_wrapper_51304dense_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	ttotal
	ucount
v	variables
w	keras_api"
_tf_keras_metric
^
	xtotal
	ycount
z
_fn_kwargs
{	variables
|	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
t0
u1"
trackable_list_wrapper
-
v	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
2:0
��2 Adadelta/dense/kernel/accum_grad
+:)�2Adadelta/dense/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
3:1	�@2"Adadelta/dense_1/kernel/accum_grad
,:*@2 Adadelta/dense_1/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
2:0@d2"Adadelta/dense_2/kernel/accum_grad
,:*d2 Adadelta/dense_2/bias/accum_grad
$:" 2Adadelta/Variable/accum_grad
1:/
��2Adadelta/dense/kernel/accum_var
*:(�2Adadelta/dense/bias/accum_var
#:! 2Adadelta/Variable/accum_var
2:0	�@2!Adadelta/dense_1/kernel/accum_var
+:)@2Adadelta/dense_1/bias/accum_var
#:! 2Adadelta/Variable/accum_var
1:/@d2!Adadelta/dense_2/kernel/accum_var
+:)d2Adadelta/dense_2/bias/accum_var
#:! 2Adadelta/Variable/accum_var�
 __inference__wrapped_model_50487�
 !(/07>5�2
+�(
&�#
dense_input����������
� "A�>
<
softmax__decode)�&
softmax__decode���������
�
B__inference_dense_1_layer_call_and_return_conditional_losses_51390] !0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� {
'__inference_dense_1_layer_call_fn_51380P !0�-
&�#
!�
inputs����������
� "����������@�
B__inference_dense_2_layer_call_and_return_conditional_losses_51457\/0/�,
%�"
 �
inputs���������@
� "%�"
�
0���������d
� z
'__inference_dense_2_layer_call_fn_51447O/0/�,
%�"
 �
inputs���������@
� "����������d�
@__inference_dense_layer_call_and_return_conditional_losses_51323^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� z
%__inference_dense_layer_call_fn_51313Q0�-
&�#
!�
inputs����������
� "������������
E__inference_sequential_layer_call_and_return_conditional_losses_50910r
 !(/07>=�:
3�0
&�#
dense_input����������
p 

 
� "%�"
�
0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_50941r
 !(/07>=�:
3�0
&�#
dense_input����������
p

 
� "%�"
�
0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_51142m
 !(/07>8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������

� �
E__inference_sequential_layer_call_and_return_conditional_losses_51277m
 !(/07>8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������

� �
*__inference_sequential_layer_call_fn_50711e
 !(/07>=�:
3�0
&�#
dense_input����������
p 

 
� "����������
�
*__inference_sequential_layer_call_fn_50879e
 !(/07>=�:
3�0
&�#
dense_input����������
p

 
� "����������
�
*__inference_sequential_layer_call_fn_50982`
 !(/07>8�5
.�+
!�
inputs����������
p 

 
� "����������
�
*__inference_sequential_layer_call_fn_51007`
 !(/07>8�5
.�+
!�
inputs����������
p

 
� "����������
�
#__inference_signature_wrapper_51304�
 !(/07>D�A
� 
:�7
5
dense_input&�#
dense_input����������"A�>
<
softmax__decode)�&
softmax__decode���������
�
J__inference_softmax__decode_layer_call_and_return_conditional_losses_51524[>/�,
%�"
 �
inputs���������d
� "%�"
�
0���������

� �
/__inference_softmax__decode_layer_call_fn_51512N>/�,
%�"
 �
inputs���������d
� "����������
�
K__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_51438[(/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� �
0__inference_spiking_b_relu_1_layer_call_fn_51399N(/�,
%�"
 �
inputs���������@
� "����������@�
K__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_51505[7/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� �
0__inference_spiking_b_relu_2_layer_call_fn_51466N7/�,
%�"
 �
inputs���������d
� "����������d�
I__inference_spiking_b_relu_layer_call_and_return_conditional_losses_51371]0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
.__inference_spiking_b_relu_layer_call_fn_51332P0�-
&�#
!�
inputs����������
� "�����������