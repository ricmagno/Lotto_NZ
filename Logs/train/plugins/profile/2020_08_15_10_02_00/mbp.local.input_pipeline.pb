	!�rh��P@!�rh��P@!!�rh��P@	 �GQw�	@ �GQw�	@! �GQw�	@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'�!�rh��P@��K7AP@A���Mb�?Y�I+�@*	     ,�@2h
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle��G�z��?!��\��K@)�G�z��?1��\��K@:Preprocessing2t
=Iterator::Model::ForeverRepeat::BatchV2::Shuffle::MemoryCache0���S��?!��̰{�3@)���S��?1��̰{�3@:Preprocessing2x
AIterator::Model::ForeverRepeat::BatchV2::Shuffle::MemoryCacheImpl2D�l����?!4&�)@)��K7��?1�����(@:Preprocessing2�
NIterator::Model::ForeverRepeat::BatchV2::Shuffle::MemoryCacheImpl::TensorSliceZ�S㥛��?!��� �'@)�S㥛��?1��� �'@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B95.7 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��K7AP@��K7AP@!��K7AP@      ��!       "      ��!       *      ��!       2	���Mb�?���Mb�?!���Mb�?:      ��!       B      ��!       J	�I+�@�I+�@!�I+�@R      ��!       Z	�I+�@�I+�@!�I+�@JCPU_ONLY