??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02v2.3.0-rc2-23-gb36436b0878??
{
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? * 
shared_namedense_39/kernel
t
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes
:	? *
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
: *
dtype0
z
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_40/kernel
s
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes

:  *
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes
: *
dtype0
z
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_41/kernel
s
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes

: *
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
h


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
 
*

0
1
2
3
4
5
*

0
1
2
3
4
5
 
?
trainable_variables

layers
metrics
	variables
layer_regularization_losses
layer_metrics
 non_trainable_variables
regularization_losses
 
[Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1


0
1
 
?
trainable_variables

!layers
"metrics
	variables
#layer_regularization_losses
$layer_metrics
%non_trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

&layers
'metrics
	variables
(layer_regularization_losses
)layer_metrics
*non_trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_41/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

+layers
,metrics
	variables
-layer_regularization_losses
.layer_metrics
/non_trainable_variables
regularization_losses

0
1
2
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
?
serving_default_dense_39_inputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_39_inputdense_39/kerneldense_39/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_120555
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOpConst*
Tin

2*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_120740
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_39/kerneldense_39/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/bias*
Tin
	2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_120768??
?
?
.__inference_sequential_17_layer_call_fn_120639

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_1205212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_40_layer_call_and_return_conditional_losses_120400

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_dense_41_layer_call_and_return_conditional_losses_120427

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_120358
dense_39_input9
5sequential_17_dense_39_matmul_readvariableop_resource:
6sequential_17_dense_39_biasadd_readvariableop_resource9
5sequential_17_dense_40_matmul_readvariableop_resource:
6sequential_17_dense_40_biasadd_readvariableop_resource9
5sequential_17_dense_41_matmul_readvariableop_resource:
6sequential_17_dense_41_biasadd_readvariableop_resource
identity??
,sequential_17/dense_39/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_39_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02.
,sequential_17/dense_39/MatMul/ReadVariableOp?
sequential_17/dense_39/MatMulMatMuldense_39_input4sequential_17/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_17/dense_39/MatMul?
-sequential_17/dense_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_17/dense_39/BiasAdd/ReadVariableOp?
sequential_17/dense_39/BiasAddBiasAdd'sequential_17/dense_39/MatMul:product:05sequential_17/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_17/dense_39/BiasAdd?
sequential_17/dense_39/ReluRelu'sequential_17/dense_39/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_17/dense_39/Relu?
,sequential_17/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_40_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02.
,sequential_17/dense_40/MatMul/ReadVariableOp?
sequential_17/dense_40/MatMulMatMul)sequential_17/dense_39/Relu:activations:04sequential_17/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_17/dense_40/MatMul?
-sequential_17/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_17/dense_40/BiasAdd/ReadVariableOp?
sequential_17/dense_40/BiasAddBiasAdd'sequential_17/dense_40/MatMul:product:05sequential_17/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_17/dense_40/BiasAdd?
sequential_17/dense_40/ReluRelu'sequential_17/dense_40/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_17/dense_40/Relu?
,sequential_17/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_41_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_17/dense_41/MatMul/ReadVariableOp?
sequential_17/dense_41/MatMulMatMul)sequential_17/dense_40/Relu:activations:04sequential_17/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_41/MatMul?
-sequential_17/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_17/dense_41/BiasAdd/ReadVariableOp?
sequential_17/dense_41/BiasAddBiasAdd'sequential_17/dense_41/MatMul:product:05sequential_17/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_17/dense_41/BiasAdd?
sequential_17/dense_41/SoftmaxSoftmax'sequential_17/dense_41/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_17/dense_41/Softmax|
IdentityIdentity(sequential_17/dense_41/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::::X T
(
_output_shapes
:??????????
(
_user_specified_namedense_39_input
?
~
)__inference_dense_40_layer_call_fn_120679

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_1204002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_dense_39_layer_call_and_return_conditional_losses_120650

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_40_layer_call_and_return_conditional_losses_120670

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference__traced_save_120740
file_prefix.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_cc55ef8931d34dc98ea5cff72a91665f/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
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

identity_1Identity_1:output:0*H
_input_shapes7
5: :	? : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	? : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?
?
"__inference__traced_restore_120768
file_prefix$
 assignvariableop_dense_39_kernel$
 assignvariableop_1_dense_39_bias&
"assignvariableop_2_dense_40_kernel$
 assignvariableop_3_dense_40_bias&
"assignvariableop_4_dense_41_kernel$
 assignvariableop_5_dense_41_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_39_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_39_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_40_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_40_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_41_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_41_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
~
)__inference_dense_39_layer_call_fn_120659

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_1203732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_39_layer_call_and_return_conditional_losses_120373

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_41_layer_call_fn_120699

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_1204272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_dense_41_layer_call_and_return_conditional_losses_120690

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120605

inputs+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'dense_41_matmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identity??
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_39/MatMul/ReadVariableOp?
dense_39/MatMulMatMulinputs&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_39/MatMul?
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_39/BiasAdd/ReadVariableOp?
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_39/BiasAdds
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_39/Relu?
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_40/MatMul/ReadVariableOp?
dense_40/MatMulMatMuldense_39/Relu:activations:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_40/MatMul?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_40/BiasAdds
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_40/Relu?
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_41/MatMul/ReadVariableOp?
dense_41/MatMulMatMuldense_40/Relu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_41/MatMul?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_41/BiasAdd|
dense_41/SoftmaxSoftmaxdense_41/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_41/Softmaxn
IdentityIdentitydense_41/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120521

inputs
dense_39_120505
dense_39_120507
dense_40_120510
dense_40_120512
dense_41_120515
dense_41_120517
identity?? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCallinputsdense_39_120505dense_39_120507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_1203732"
 dense_39/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_120510dense_40_120512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_1204002"
 dense_40/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_120515dense_41_120517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_1204272"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120463
dense_39_input
dense_39_120447
dense_39_120449
dense_40_120452
dense_40_120454
dense_41_120457
dense_41_120459
identity?? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCalldense_39_inputdense_39_120447dense_39_120449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_1203732"
 dense_39/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_120452dense_40_120454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_1204002"
 dense_40/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_120457dense_41_120459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_1204272"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_39_input
?
?
$__inference_signature_wrapper_120555
dense_39_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_39_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1203582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_39_input
?
?
.__inference_sequential_17_layer_call_fn_120536
dense_39_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_39_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_1205212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_39_input
?
?
.__inference_sequential_17_layer_call_fn_120500
dense_39_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_39_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_1204852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_39_input
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120444
dense_39_input
dense_39_120384
dense_39_120386
dense_40_120411
dense_40_120413
dense_41_120438
dense_41_120440
identity?? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCalldense_39_inputdense_39_120384dense_39_120386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_1203732"
 dense_39/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_120411dense_40_120413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_1204002"
 dense_40/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_120438dense_41_120440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_1204272"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_39_input
?
?
.__inference_sequential_17_layer_call_fn_120622

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_1204852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120485

inputs
dense_39_120469
dense_39_120471
dense_40_120474
dense_40_120476
dense_41_120479
dense_41_120481
identity?? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCallinputsdense_39_120469dense_39_120471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_1203732"
 dense_39/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_120474dense_40_120476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_1204002"
 dense_40/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_120479dense_41_120481*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_1204272"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120580

inputs+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'dense_41_matmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identity??
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype02 
dense_39/MatMul/ReadVariableOp?
dense_39/MatMulMatMulinputs&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_39/MatMul?
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_39/BiasAdd/ReadVariableOp?
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_39/BiasAdds
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_39/Relu?
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_40/MatMul/ReadVariableOp?
dense_40/MatMulMatMuldense_39/Relu:activations:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_40/MatMul?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_40/BiasAdds
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_40/Relu?
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_41/MatMul/ReadVariableOp?
dense_41/MatMulMatMuldense_40/Relu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_41/MatMul?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_41/BiasAdd|
dense_41/SoftmaxSoftmaxdense_41/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_41/Softmaxn
IdentityIdentitydense_41/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????:::::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
J
dense_39_input8
 serving_default_dense_39_input:0??????????<
dense_410
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?x
?"
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
0_default_save_signature
*1&call_and_return_all_conditional_losses
2__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_39_input"}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 500]}, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_39_input"}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 500]}, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 500]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 500]}, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*5&call_and_return_all_conditional_losses
6__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*7&call_and_return_all_conditional_losses
8__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
"
	optimizer
J

0
1
2
3
4
5"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

layers
metrics
	variables
layer_regularization_losses
layer_metrics
 non_trainable_variables
regularization_losses
2__call__
0_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
,
9serving_default"
signature_map
": 	? 2dense_39/kernel
: 2dense_39/bias
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

!layers
"metrics
	variables
#layer_regularization_losses
$layer_metrics
%non_trainable_variables
regularization_losses
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_40/kernel
: 2dense_40/bias
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
trainable_variables

&layers
'metrics
	variables
(layer_regularization_losses
)layer_metrics
*non_trainable_variables
regularization_losses
6__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_41/kernel
:2dense_41/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

+layers
,metrics
	variables
-layer_regularization_losses
.layer_metrics
/non_trainable_variables
regularization_losses
8__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
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
?2?
!__inference__wrapped_model_120358?
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
annotations? *.?+
)?&
dense_39_input??????????
?2?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120605
I__inference_sequential_17_layer_call_and_return_conditional_losses_120463
I__inference_sequential_17_layer_call_and_return_conditional_losses_120444
I__inference_sequential_17_layer_call_and_return_conditional_losses_120580?
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
.__inference_sequential_17_layer_call_fn_120639
.__inference_sequential_17_layer_call_fn_120622
.__inference_sequential_17_layer_call_fn_120536
.__inference_sequential_17_layer_call_fn_120500?
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
D__inference_dense_39_layer_call_and_return_conditional_losses_120650?
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
)__inference_dense_39_layer_call_fn_120659?
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
D__inference_dense_40_layer_call_and_return_conditional_losses_120670?
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
)__inference_dense_40_layer_call_fn_120679?
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
D__inference_dense_41_layer_call_and_return_conditional_losses_120690?
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
)__inference_dense_41_layer_call_fn_120699?
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
:B8
$__inference_signature_wrapper_120555dense_39_input?
!__inference__wrapped_model_120358w
8?5
.?+
)?&
dense_39_input??????????
? "3?0
.
dense_41"?
dense_41??????????
D__inference_dense_39_layer_call_and_return_conditional_losses_120650]
0?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? }
)__inference_dense_39_layer_call_fn_120659P
0?-
&?#
!?
inputs??????????
? "?????????? ?
D__inference_dense_40_layer_call_and_return_conditional_losses_120670\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? |
)__inference_dense_40_layer_call_fn_120679O/?,
%?"
 ?
inputs????????? 
? "?????????? ?
D__inference_dense_41_layer_call_and_return_conditional_losses_120690\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
)__inference_dense_41_layer_call_fn_120699O/?,
%?"
 ?
inputs????????? 
? "???????????
I__inference_sequential_17_layer_call_and_return_conditional_losses_120444q
@?=
6?3
)?&
dense_39_input??????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120463q
@?=
6?3
)?&
dense_39_input??????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120580i
8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_120605i
8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_17_layer_call_fn_120500d
@?=
6?3
)?&
dense_39_input??????????
p

 
? "???????????
.__inference_sequential_17_layer_call_fn_120536d
@?=
6?3
)?&
dense_39_input??????????
p 

 
? "???????????
.__inference_sequential_17_layer_call_fn_120622\
8?5
.?+
!?
inputs??????????
p

 
? "???????????
.__inference_sequential_17_layer_call_fn_120639\
8?5
.?+
!?
inputs??????????
p 

 
? "???????????
$__inference_signature_wrapper_120555?
J?G
? 
@?=
;
dense_39_input)?&
dense_39_input??????????"3?0
.
dense_41"?
dense_41?????????