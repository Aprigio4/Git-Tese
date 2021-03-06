from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, Reshape, Dense, multiply, Permute, Concatenate, Conv3D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid


def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,4,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=4, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=4, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = Concatenate(axis=4)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = Conv3D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((4, 1, 2, 3))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])