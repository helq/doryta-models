�
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
executor_typestring �
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
 �"serve*2.6.02unknown8�
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

NoOpNoOp
�%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�$
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
	
signatures
#
_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
�
	sharpness
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
 	variables
!trainable_variables
"	keras_api
�
#	sharpness
#$_self_saveable_object_factories
%regularization_losses
&	variables
'trainable_variables
(	keras_api
�

)kernel
*bias
#+_self_saveable_object_factories
,regularization_losses
-	variables
.trainable_variables
/	keras_api
�
0	sharpness
#1_self_saveable_object_factories
2regularization_losses
3	variables
4trainable_variables
5	keras_api
�
6_rescaled_key
#7_self_saveable_object_factories
8regularization_losses
9	variables
:trainable_variables
;	keras_api
 
 
 
 
F
0
1
2
3
4
#5
)6
*7
08
69
?
0
1
2
3
4
#5
)6
*7
08
�

<layers
regularization_losses
=layer_metrics
	variables
>metrics
?layer_regularization_losses
@non_trainable_variables
trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
�

Alayers
Blayer_regularization_losses
regularization_losses
Clayer_metrics
	variables
Dmetrics
Enon_trainable_variables
trainable_variables
WU
VARIABLE_VALUEVariable9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
�

Flayers
Glayer_regularization_losses
regularization_losses
Hlayer_metrics
	variables
Imetrics
Jnon_trainable_variables
trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
�

Klayers
Llayer_regularization_losses
regularization_losses
Mlayer_metrics
 	variables
Nmetrics
Onon_trainable_variables
!trainable_variables
YW
VARIABLE_VALUE
Variable_19layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUE
 
 

#0

#0
�

Players
Qlayer_regularization_losses
%regularization_losses
Rlayer_metrics
&	variables
Smetrics
Tnon_trainable_variables
'trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

)0
*1

)0
*1
�

Ulayers
Vlayer_regularization_losses
,regularization_losses
Wlayer_metrics
-	variables
Xmetrics
Ynon_trainable_variables
.trainable_variables
YW
VARIABLE_VALUE
Variable_29layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUE
 
 

00

00
�

Zlayers
[layer_regularization_losses
2regularization_losses
\layer_metrics
3	variables
]metrics
^non_trainable_variables
4trainable_variables
][
VARIABLE_VALUE
Variable_3=layer_with_weights-6/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUE
 
 

60
 
�

_layers
`layer_regularization_losses
8regularization_losses
alayer_metrics
9	variables
bmetrics
cnon_trainable_variables
:trainable_variables
1
0
1
2
3
4
5
6
 

d0
e1
 

60
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
60
4
	ftotal
	gcount
h	variables
i	keras_api
D
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

h	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

m	variables
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
GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_6528
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpVariable/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpVariable_1/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8� *&
f!R
__inference__traced_save_6768
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasVariabledense_1/kerneldense_1/bias
Variable_1dense_2/kerneldense_2/bias
Variable_2
Variable_3totalcounttotal_1count_1*
Tin
2*
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
GPU 2J 8� *)
f$R"
 __inference__traced_restore_6820��
�
s
'__inference_restored_function_body_6197

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_3482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
H__inference_softmax__decode_layer_call_and_return_conditional_losses_291

inputs0
matmul_readvariableop_resource:d

identity��MatMul/ReadVariableOpS
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
:���������d2
mulS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/y\
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:���������d2
sub�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:���������
2	
Softmaxf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOpl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
s
'__inference_restored_function_body_6181

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_8912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
b
__inference_cond_true_22262_862
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:���������@2
cond/Identity"'
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
�
�
-__inference_softmax__decode_layer_call_fn_297

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
GPU 2J 8� *Q
fLRJ
H__inference_softmax__decode_layer_call_and_return_conditional_losses_2912
StatefulPartitionedCallh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
b
__inference_cond_true_22329_245
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:���������d2
cond/Identity"'
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

�
?__inference_dense_layer_call_and_return_conditional_losses_6229

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
�

�
)__inference_sequential_layer_call_fn_6439
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
GPU 2J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_63912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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

�
?__inference_dense_layer_call_and_return_conditional_losses_6665

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
�%
�
D__inference_sequential_layer_call_and_return_conditional_losses_6280

inputs

dense_6230:
��

dense_6232:	�
spiking_b_relu_6235: 
dense_1_6249:	�@
dense_1_6251:@
spiking_b_relu_1_6254: 
dense_2_6268:@d
dense_2_6270:d
spiking_b_relu_2_6273: &
softmax__decode_6276:d

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_6230
dense_6232*
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
GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62292
dense/StatefulPartitionedCall�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_6235*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61652(
&spiking_b_relu/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0dense_1_6249dense_1_6251*
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
GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62482!
dense_1/StatefulPartitionedCall�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_1_6254*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61812*
(spiking_b_relu_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0dense_2_6268dense_2_6270*
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
GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_62672!
dense_2/StatefulPartitionedCall�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_2_6273*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61972*
(spiking_b_relu_2/StatefulPartitionedCall�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_6276*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_62072)
'softmax__decode/StatefulPartitionedCall�
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
�&
�
D__inference_sequential_layer_call_and_return_conditional_losses_6470
dense_input

dense_6442:
��

dense_6444:	�
spiking_b_relu_6447: 
dense_1_6450:	�@
dense_1_6452:@
spiking_b_relu_1_6455: 
dense_2_6458:@d
dense_2_6460:d
spiking_b_relu_2_6463: &
softmax__decode_6466:d

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input
dense_6442
dense_6444*
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
GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62292
dense/StatefulPartitionedCall�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_6447*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61652(
&spiking_b_relu/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0dense_1_6450dense_1_6452*
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
GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62482!
dense_1/StatefulPartitionedCall�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_1_6455*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61812*
(spiking_b_relu_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0dense_2_6458dense_2_6460*
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
GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_62672!
dense_2/StatefulPartitionedCall�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_2_6463*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61972*
(spiking_b_relu_2/StatefulPartitionedCall�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_6466*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_62072)
'softmax__decode/StatefulPartitionedCall�
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
�

�
A__inference_dense_1_layer_call_and_return_conditional_losses_6684

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
�8
�
__inference__wrapped_model_6212
dense_inputC
/sequential_dense_matmul_readvariableop_resource:
��?
0sequential_dense_biasadd_readvariableop_resource:	�(
sequential_spiking_b_relu_6166: D
1sequential_dense_1_matmul_readvariableop_resource:	�@@
2sequential_dense_1_biasadd_readvariableop_resource:@*
 sequential_spiking_b_relu_1_6182: C
1sequential_dense_2_matmul_readvariableop_resource:@d@
2sequential_dense_2_biasadd_readvariableop_resource:d*
 sequential_spiking_b_relu_2_6198: 1
sequential_softmax__decode_6208:d

identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�)sequential/dense_2/BiasAdd/ReadVariableOp�(sequential/dense_2/MatMul/ReadVariableOp�2sequential/softmax__decode/StatefulPartitionedCall�1sequential/spiking_b_relu/StatefulPartitionedCall�3sequential/spiking_b_relu_1/StatefulPartitionedCall�3sequential/spiking_b_relu_2/StatefulPartitionedCall�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02(
&sequential/dense/MatMul/ReadVariableOp�
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/MatMul�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/BiasAdd�
1sequential/spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall!sequential/dense/BiasAdd:output:0sequential_spiking_b_relu_6166*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_616523
1sequential/spiking_b_relu/StatefulPartitionedCall�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp�
sequential/dense_1/MatMulMatMul:sequential/spiking_b_relu/StatefulPartitionedCall:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
sequential/dense_1/MatMul�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
sequential/dense_1/BiasAdd�
3sequential/spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall#sequential/dense_1/BiasAdd:output:0 sequential_spiking_b_relu_1_6182*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_618125
3sequential/spiking_b_relu_1/StatefulPartitionedCall�
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:@d*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp�
sequential/dense_2/MatMulMatMul<sequential/spiking_b_relu_1/StatefulPartitionedCall:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential/dense_2/MatMul�
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp�
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential/dense_2/BiasAdd�
3sequential/spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall#sequential/dense_2/BiasAdd:output:0 sequential_spiking_b_relu_2_6198*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_619725
3sequential/spiking_b_relu_2/StatefulPartitionedCall�
2sequential/softmax__decode/StatefulPartitionedCallStatefulPartitionedCall<sequential/spiking_b_relu_2/StatefulPartitionedCall:output:0sequential_softmax__decode_6208*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_620724
2sequential/softmax__decode/StatefulPartitionedCall�
IdentityIdentity;sequential/softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp3^sequential/softmax__decode/StatefulPartitionedCall2^sequential/spiking_b_relu/StatefulPartitionedCall4^sequential/spiking_b_relu_1/StatefulPartitionedCall4^sequential/spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2h
2sequential/softmax__decode/StatefulPartitionedCall2sequential/softmax__decode/StatefulPartitionedCall2f
1sequential/spiking_b_relu/StatefulPartitionedCall1sequential/spiking_b_relu/StatefulPartitionedCall2j
3sequential/spiking_b_relu_1/StatefulPartitionedCall3sequential/spiking_b_relu_1/StatefulPartitionedCall2j
3sequential/spiking_b_relu_2/StatefulPartitionedCall3sequential/spiking_b_relu_2/StatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
a
__inference_cond_true_21432_78
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:���������@2
cond/Identity"'
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
b
__inference_cond_true_21491_145
cond_identity_cast
cond_placeholder
cond_identityp
cond/IdentityIdentitycond_identity_cast*
T0*'
_output_shapes
:���������d2
cond/Identity"'
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
"__inference_signature_wrapper_6528
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
GPU 2J 8� *(
f#R!
__inference__wrapped_model_62122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
l
 __inference_cond_false_22330_319
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:���������d2
cond/Identity"'
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

�
G__inference_softmax__decode_layer_call_and_return_conditional_losses_90

inputs0
matmul_readvariableop_resource:d

identity��MatMul/ReadVariableOpS
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
:���������d2
mulS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/y\
subSubmul:z:0sub/y:output:0*
T0*'
_output_shapes
:���������d2
sub�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulsub:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:���������
2	
Softmaxf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOpl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
G__inference_spiking_b_relu_layer_call_and_return_conditional_losses_279

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y�
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
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
 *  �?2
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
 *o�:2
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
 *  �?2
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
sub_1/yb
sub_1Subinputssub_1/y:output:0*
T0*(
_output_shapes
:����������2
sub_1\
mulMultruediv:z:0	sub_1:z:0*
T0*(
_output_shapes
:����������2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/ye
add_1AddV2mul:z:0add_1/y:output:0*
T0*(
_output_shapes
:����������2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:����������2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:����������2
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
 *  �?2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal�
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
 *3
else_branch$R"
 __inference_cond_false_22196_250*'
output_shapes
:����������*2
then_branch#R!
__inference_cond_true_22195_1052
condl
cond/IdentityIdentitycond:output:0*
T0*(
_output_shapes
:����������2
cond/Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOpr
IdentityIdentitycond/Identity:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity"
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
�
�
&__inference_dense_2_layer_call_fn_6693

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
GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_62672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�&
�
__inference__traced_save_6768
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
%savev2_variable_3_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-6/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop#savev2_variable_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop%savev2_variable_1_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*c
_input_shapesR
P: :
��:�: :	�@:@: :@d:d: :d
: : : : : 2(
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
: 
�
�
I__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_174

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpe
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
:���������d2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
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
 *  �?2
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
 *o�:2
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
 *  �?2
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
:���������d2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������d2
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
:���������d2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������d2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d2
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
 *  �?2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal�
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
 *2
else_branch#R!
__inference_cond_false_21492_95*&
output_shapes
:���������d*2
then_branch#R!
__inference_cond_true_21491_1452
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:���������d2
cond/Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOpq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity"
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
�
�
I__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_731

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpe
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
:���������@2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
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
 *  �?2
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
 *o�:2
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
 *  �?2
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
:���������@2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������@2
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
:���������@2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@2
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
 *  �?2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal�
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
 *3
else_branch$R"
 __inference_cond_false_21433_702*&
output_shapes
:���������@*1
then_branch"R 
__inference_cond_true_21432_782
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:���������@2
cond/Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOpq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity"
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
�/
�
D__inference_sequential_layer_call_and_return_conditional_losses_6612

inputs8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�
spiking_b_relu_6587: 9
&dense_1_matmul_readvariableop_resource:	�@5
'dense_1_biasadd_readvariableop_resource:@
spiking_b_relu_1_6596: 8
&dense_2_matmul_readvariableop_resource:@d5
'dense_2_biasadd_readvariableop_resource:d
spiking_b_relu_2_6605: &
softmax__decode_6608:d

identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAdd�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCalldense/BiasAdd:output:0spiking_b_relu_6587*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61652(
&spiking_b_relu/StatefulPartitionedCall�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMul/spiking_b_relu/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_1/BiasAdd�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCalldense_1/BiasAdd:output:0spiking_b_relu_1_6596*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61812*
(spiking_b_relu_1/StatefulPartitionedCall�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@d*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul1spiking_b_relu_1/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_2/BiasAdd�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCalldense_2/BiasAdd:output:0spiking_b_relu_2_6605*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61972*
(spiking_b_relu_2/StatefulPartitionedCall�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_6608*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_62072)
'softmax__decode/StatefulPartitionedCall�
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
s
'__inference_restored_function_body_6165

inputs
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_spiking_b_relu_layer_call_and_return_conditional_losses_2792
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
 __inference__traced_restore_6820
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
#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: 
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/sharpness/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-5/sharpness/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-6/_rescaled_key/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variableIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_3Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14f
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_15�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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

�
)__inference_sequential_layer_call_fn_6553

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
GPU 2J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_62802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
l
 __inference_cond_false_21374_495
cond_placeholder
cond_identity_clip_by_value
cond_identityz
cond/IdentityIdentitycond_identity_clip_by_value*
T0*(
_output_shapes
:����������2
cond/Identity"'
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
�&
�
D__inference_sequential_layer_call_and_return_conditional_losses_6501
dense_input

dense_6473:
��

dense_6475:	�
spiking_b_relu_6478: 
dense_1_6481:	�@
dense_1_6483:@
spiking_b_relu_1_6486: 
dense_2_6489:@d
dense_2_6491:d
spiking_b_relu_2_6494: &
softmax__decode_6497:d

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input
dense_6473
dense_6475*
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
GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62292
dense/StatefulPartitionedCall�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_6478*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61652(
&spiking_b_relu/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0dense_1_6481dense_1_6483*
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
GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62482!
dense_1/StatefulPartitionedCall�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_1_6486*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61812*
(spiking_b_relu_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0dense_2_6489dense_2_6491*
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
GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_62672!
dense_2/StatefulPartitionedCall�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_2_6494*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61972*
(spiking_b_relu_2/StatefulPartitionedCall�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_6497*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_62072)
'softmax__decode/StatefulPartitionedCall�
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
�
l
 __inference_cond_false_22196_250
cond_placeholder
cond_identity_clip_by_value
cond_identityz
cond/IdentityIdentitycond_identity_clip_by_value*
T0*(
_output_shapes
:����������2
cond/Identity"'
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
�

�
A__inference_dense_1_layer_call_and_return_conditional_losses_6248

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
�
k
__inference_cond_false_21492_95
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:���������d2
cond/Identity"'
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
D__inference_sequential_layer_call_and_return_conditional_losses_6391

inputs

dense_6363:
��

dense_6365:	�
spiking_b_relu_6368: 
dense_1_6371:	�@
dense_1_6373:@
spiking_b_relu_1_6376: 
dense_2_6379:@d
dense_2_6381:d
spiking_b_relu_2_6384: &
softmax__decode_6387:d

identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_6363
dense_6365*
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
GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62292
dense/StatefulPartitionedCall�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0spiking_b_relu_6368*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61652(
&spiking_b_relu/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall/spiking_b_relu/StatefulPartitionedCall:output:0dense_1_6371dense_1_6373*
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
GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62482!
dense_1/StatefulPartitionedCall�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0spiking_b_relu_1_6376*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61812*
(spiking_b_relu_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_1/StatefulPartitionedCall:output:0dense_2_6379dense_2_6381*
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
GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_62672!
dense_2/StatefulPartitionedCall�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0spiking_b_relu_2_6384*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61972*
(spiking_b_relu_2/StatefulPartitionedCall�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_6387*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_62072)
'softmax__decode/StatefulPartitionedCall�
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
�
b
__inference_cond_true_21373_500
cond_identity_cast
cond_placeholder
cond_identityq
cond/IdentityIdentitycond_identity_cast*
T0*(
_output_shapes
:����������2
cond/Identity"'
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
�
&__inference_dense_1_layer_call_fn_6674

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
GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
x
,__inference_spiking_b_relu_layer_call_fn_535

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
GPU 2J 8� *P
fKRI
G__inference_spiking_b_relu_layer_call_and_return_conditional_losses_5292
StatefulPartitionedCallh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_348

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpe
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
:���������d2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
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
 *  �?2
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
 *o�:2
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
 *  �?2
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
:���������d2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������d2
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
:���������d2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������d2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d2
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
 *  �?2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal�
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
 *3
else_branch$R"
 __inference_cond_false_22330_319*&
output_shapes
:���������d*2
then_branch#R!
__inference_cond_true_22329_2452
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:���������d2
cond/Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOpq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity"
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
�
l
 __inference_cond_false_22263_140
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:���������@2
cond/Identity"'
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
�

�
)__inference_sequential_layer_call_fn_6578

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
GPU 2J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_63912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
$__inference_dense_layer_call_fn_6655

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
GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62292
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_891

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpe
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
:���������@2
GreaterEqualg
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
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
 *  �?2
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
 *o�:2
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
 *  �?2
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
:���������@2
sub_1[
mulMultruediv:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������@2
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
:���������@2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������@2
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
 *  �?2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal�
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
 *3
else_branch$R"
 __inference_cond_false_22263_140*&
output_shapes
:���������@*2
then_branch#R!
__inference_cond_true_22262_8622
condk
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:���������@2
cond/Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOpq
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity"
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
�
{
'__inference_restored_function_body_6207

inputs
unknown:d

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_softmax__decode_layer_call_and_return_conditional_losses_902
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
A__inference_dense_2_layer_call_and_return_conditional_losses_6267

inputs0
matmul_readvariableop_resource:@d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
b
__inference_cond_true_22195_105
cond_identity_cast
cond_placeholder
cond_identityq
cond/IdentityIdentitycond_identity_cast*
T0*(
_output_shapes
:����������2
cond/Identity"'
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
�

�
A__inference_dense_2_layer_call_and_return_conditional_losses_6703

inputs0
matmul_readvariableop_resource:@d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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

�
)__inference_sequential_layer_call_fn_6303
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
GPU 2J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_62802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
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
l
 __inference_cond_false_21433_702
cond_placeholder
cond_identity_clip_by_value
cond_identityy
cond/IdentityIdentitycond_identity_clip_by_value*
T0*'
_output_shapes
:���������@2
cond/Identity"'
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
�
z
.__inference_spiking_b_relu_1_layer_call_fn_737

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
GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_7312
StatefulPartitionedCallh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������@: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�/
�
D__inference_sequential_layer_call_and_return_conditional_losses_6646

inputs8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�
spiking_b_relu_6621: 9
&dense_1_matmul_readvariableop_resource:	�@5
'dense_1_biasadd_readvariableop_resource:@
spiking_b_relu_1_6630: 8
&dense_2_matmul_readvariableop_resource:@d5
'dense_2_biasadd_readvariableop_resource:d
spiking_b_relu_2_6639: &
softmax__decode_6642:d

identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�'softmax__decode/StatefulPartitionedCall�&spiking_b_relu/StatefulPartitionedCall�(spiking_b_relu_1/StatefulPartitionedCall�(spiking_b_relu_2/StatefulPartitionedCall�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAdd�
&spiking_b_relu/StatefulPartitionedCallStatefulPartitionedCalldense/BiasAdd:output:0spiking_b_relu_6621*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61652(
&spiking_b_relu/StatefulPartitionedCall�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMul/spiking_b_relu/StatefulPartitionedCall:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_1/BiasAdd�
(spiking_b_relu_1/StatefulPartitionedCallStatefulPartitionedCalldense_1/BiasAdd:output:0spiking_b_relu_1_6630*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61812*
(spiking_b_relu_1/StatefulPartitionedCall�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@d*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul1spiking_b_relu_1/StatefulPartitionedCall:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_2/BiasAdd�
(spiking_b_relu_2/StatefulPartitionedCallStatefulPartitionedCalldense_2/BiasAdd:output:0spiking_b_relu_2_6639*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_61972*
(spiking_b_relu_2/StatefulPartitionedCall�
'softmax__decode/StatefulPartitionedCallStatefulPartitionedCall1spiking_b_relu_2/StatefulPartitionedCall:output:0softmax__decode_6642*
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
GPU 2J 8� *0
f+R)
'__inference_restored_function_body_62072)
'softmax__decode/StatefulPartitionedCall�
IdentityIdentity0softmax__decode/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^softmax__decode/StatefulPartitionedCall'^spiking_b_relu/StatefulPartitionedCall)^spiking_b_relu_1/StatefulPartitionedCall)^spiking_b_relu_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'softmax__decode/StatefulPartitionedCall'softmax__decode/StatefulPartitionedCall2P
&spiking_b_relu/StatefulPartitionedCall&spiking_b_relu/StatefulPartitionedCall2T
(spiking_b_relu_1/StatefulPartitionedCall(spiking_b_relu_1/StatefulPartitionedCall2T
(spiking_b_relu_2/StatefulPartitionedCall(spiking_b_relu_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_spiking_b_relu_layer_call_and_return_conditional_losses_529

inputs!
readvariableop_resource: 
identity��Equal/ReadVariableOp�ReadVariableOpe
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
GreaterEqual/y�
GreaterEqualGreaterEqualinputsGreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
GreaterEqualh
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
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
 *  �?2
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
 *o�:2
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
 *  �?2
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
sub_1/yb
sub_1Subinputssub_1/y:output:0*
T0*(
_output_shapes
:����������2
sub_1\
mulMultruediv:z:0	sub_1:z:0*
T0*(
_output_shapes
:����������2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
add_1/ye
add_1AddV2mul:z:0add_1/y:output:0*
T0*(
_output_shapes
:����������2
add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:����������2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:����������2
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
 *  �?2	
Equal/yh
EqualEqualEqual/ReadVariableOp:value:0Equal/y:output:0*
T0*
_output_shapes
: 2
Equal�
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
 *3
else_branch$R"
 __inference_cond_false_21374_495*'
output_shapes
:����������*2
then_branch#R!
__inference_cond_true_21373_5002
condl
cond/IdentityIdentitycond:output:0*
T0*(
_output_shapes
:����������2
cond/Identityv
NoOpNoOp^Equal/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOpr
IdentityIdentitycond/Identity:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity"
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
�
z
.__inference_spiking_b_relu_2_layer_call_fn_180

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
GPU 2J 8� *R
fMRK
I__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_1742
StatefulPartitionedCallh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs"�L
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
tensorflow/serving/predict:�
�
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
	
signatures
#
_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_default_save_signature"
_tf_keras_sequential
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	sharpness
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
 	variables
!trainable_variables
"	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
�
#	sharpness
#$_self_saveable_object_factories
%regularization_losses
&	variables
'trainable_variables
(	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
�

)kernel
*bias
#+_self_saveable_object_factories
,regularization_losses
-	variables
.trainable_variables
/	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0	sharpness
#1_self_saveable_object_factories
2regularization_losses
3	variables
4trainable_variables
5	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
�
6_rescaled_key
#7_self_saveable_object_factories
8regularization_losses
9	variables
:trainable_variables
;	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
#5
)6
*7
08
69"
trackable_list_wrapper
_
0
1
2
3
4
#5
)6
*7
08"
trackable_list_wrapper
�

<layers
regularization_losses
=layer_metrics
	variables
>metrics
?layer_regularization_losses
@non_trainable_variables
trainable_variables
o__call__
q_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 :
��2dense/kernel
:�2
dense/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

Alayers
Blayer_regularization_losses
regularization_losses
Clayer_metrics
	variables
Dmetrics
Enon_trainable_variables
trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�

Flayers
Glayer_regularization_losses
regularization_losses
Hlayer_metrics
	variables
Imetrics
Jnon_trainable_variables
trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
!:	�@2dense_1/kernel
:@2dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

Klayers
Llayer_regularization_losses
regularization_losses
Mlayer_metrics
 	variables
Nmetrics
Onon_trainable_variables
!trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
�

Players
Qlayer_regularization_losses
%regularization_losses
Rlayer_metrics
&	variables
Smetrics
Tnon_trainable_variables
'trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 :@d2dense_2/kernel
:d2dense_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�

Ulayers
Vlayer_regularization_losses
,regularization_losses
Wlayer_metrics
-	variables
Xmetrics
Ynon_trainable_variables
.trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
: 2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
00"
trackable_list_wrapper
'
00"
trackable_list_wrapper
�

Zlayers
[layer_regularization_losses
2regularization_losses
\layer_metrics
3	variables
]metrics
^non_trainable_variables
4trainable_variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
:d
2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
�

_layers
`layer_regularization_losses
8regularization_losses
alayer_metrics
9	variables
bmetrics
cnon_trainable_variables
:trainable_variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
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
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
'
60"
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
'
60"
trackable_list_wrapper
N
	ftotal
	gcount
h	variables
i	keras_api"
_tf_keras_metric
^
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
�2�
)__inference_sequential_layer_call_fn_6303
)__inference_sequential_layer_call_fn_6553
)__inference_sequential_layer_call_fn_6578
)__inference_sequential_layer_call_fn_6439�
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
D__inference_sequential_layer_call_and_return_conditional_losses_6612
D__inference_sequential_layer_call_and_return_conditional_losses_6646
D__inference_sequential_layer_call_and_return_conditional_losses_6470
D__inference_sequential_layer_call_and_return_conditional_losses_6501�
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
__inference__wrapped_model_6212dense_input"�
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
�2�
$__inference_dense_layer_call_fn_6655�
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
?__inference_dense_layer_call_and_return_conditional_losses_6665�
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
,__inference_spiking_b_relu_layer_call_fn_535�
���
FullArgSpec
args�

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
annotations� *
 
�2�
G__inference_spiking_b_relu_layer_call_and_return_conditional_losses_279�
���
FullArgSpec
args�

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
annotations� *
 
�2�
&__inference_dense_1_layer_call_fn_6674�
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
A__inference_dense_1_layer_call_and_return_conditional_losses_6684�
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
.__inference_spiking_b_relu_1_layer_call_fn_737�
���
FullArgSpec
args�

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
annotations� *
 
�2�
I__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_891�
���
FullArgSpec
args�

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
annotations� *
 
�2�
&__inference_dense_2_layer_call_fn_6693�
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
A__inference_dense_2_layer_call_and_return_conditional_losses_6703�
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
.__inference_spiking_b_relu_2_layer_call_fn_180�
���
FullArgSpec
args�

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
annotations� *
 
�2�
I__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_348�
���
FullArgSpec
args�

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
annotations� *
 
�2�
-__inference_softmax__decode_layer_call_fn_297�
���
FullArgSpec
args�

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
annotations� *
 
�2�
G__inference_softmax__decode_layer_call_and_return_conditional_losses_90�
���
FullArgSpec
args�

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
annotations� *
 
�B�
"__inference_signature_wrapper_6528dense_input"�
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
 �
__inference__wrapped_model_6212�
#)*065�2
+�(
&�#
dense_input����������
� "A�>
<
softmax__decode)�&
softmax__decode���������
�
A__inference_dense_1_layer_call_and_return_conditional_losses_6684]0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� z
&__inference_dense_1_layer_call_fn_6674P0�-
&�#
!�
inputs����������
� "����������@�
A__inference_dense_2_layer_call_and_return_conditional_losses_6703\)*/�,
%�"
 �
inputs���������@
� "%�"
�
0���������d
� y
&__inference_dense_2_layer_call_fn_6693O)*/�,
%�"
 �
inputs���������@
� "����������d�
?__inference_dense_layer_call_and_return_conditional_losses_6665^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� y
$__inference_dense_layer_call_fn_6655Q0�-
&�#
!�
inputs����������
� "������������
D__inference_sequential_layer_call_and_return_conditional_losses_6470r
#)*06=�:
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
D__inference_sequential_layer_call_and_return_conditional_losses_6501r
#)*06=�:
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
D__inference_sequential_layer_call_and_return_conditional_losses_6612m
#)*068�5
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
D__inference_sequential_layer_call_and_return_conditional_losses_6646m
#)*068�5
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
)__inference_sequential_layer_call_fn_6303e
#)*06=�:
3�0
&�#
dense_input����������
p 

 
� "����������
�
)__inference_sequential_layer_call_fn_6439e
#)*06=�:
3�0
&�#
dense_input����������
p

 
� "����������
�
)__inference_sequential_layer_call_fn_6553`
#)*068�5
.�+
!�
inputs����������
p 

 
� "����������
�
)__inference_sequential_layer_call_fn_6578`
#)*068�5
.�+
!�
inputs����������
p

 
� "����������
�
"__inference_signature_wrapper_6528�
#)*06D�A
� 
:�7
5
dense_input&�#
dense_input����������"A�>
<
softmax__decode)�&
softmax__decode���������
�
G__inference_softmax__decode_layer_call_and_return_conditional_losses_90[6/�,
%�"
 �
inputs���������d
� "%�"
�
0���������

� 
-__inference_softmax__decode_layer_call_fn_297N6/�,
%�"
 �
inputs���������d
� "����������
�
I__inference_spiking_b_relu_1_layer_call_and_return_conditional_losses_891[#/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� �
.__inference_spiking_b_relu_1_layer_call_fn_737N#/�,
%�"
 �
inputs���������@
� "����������@�
I__inference_spiking_b_relu_2_layer_call_and_return_conditional_losses_348[0/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� �
.__inference_spiking_b_relu_2_layer_call_fn_180N0/�,
%�"
 �
inputs���������d
� "����������d�
G__inference_spiking_b_relu_layer_call_and_return_conditional_losses_279]0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
,__inference_spiking_b_relu_layer_call_fn_535P0�-
&�#
!�
inputs����������
� "�����������