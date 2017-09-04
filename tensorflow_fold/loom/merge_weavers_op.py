from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# import google3
import tensorflow as tf

# This is a gross hack that (apparently) prevents Python from
# occasionally segfaulting at shutdown when unlinking dynamic
# libraries, possibly related to <https://goo.gl/aSx6Bi>.  We need to
# call some function tf.pywrap_tensorflow that munges protos, *before*
# we dlopen the deserializing weaver library (which also munges protos).
tf.pywrap_tensorflow.list_devices()

_merge_weavers = tf.load_op_library(os.path.join(
    tf.resource_loader.get_data_files_path(), '_merge_weavers_op.so'))
merge_weavers = _merge_weavers.merge_weavers

tf.NoGradient('MergeWeavers')
